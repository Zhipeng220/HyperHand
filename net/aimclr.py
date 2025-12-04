import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class


class AimCLR(nn.Module):
    def __init__(self, base_encoder=None, pretrain=True, feature_dim=128, queue_size=32768,
                 momentum=0.999, Temperature=0.07, mlp=True, **kwargs):
        """
        AimCLR Model - Dataset Agnostic
        """
        super().__init__()

        base_encoder = import_class(base_encoder)
        self.pretrain = pretrain

        encoder_args = kwargs.copy()

        if not self.pretrain:
            self.encoder_q = base_encoder(**encoder_args)
        else:
            self.K = queue_size
            self.m = momentum
            self.T = Temperature

            encoder_args['num_class'] = feature_dim

            self.encoder_q = base_encoder(**encoder_args)
            self.encoder_k = base_encoder(**encoder_args)

            if mlp:
                # [ MODIFIED ] 检查 fc 是否存在 (我们的 ctrgcn.py 有 fc)
                if hasattr(self.encoder_q, 'fc'):
                    dim_mlp = self.encoder_q.fc.weight.shape[1]
                    self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                      nn.ReLU(),
                                                      self.encoder_q.fc)
                    self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                      nn.ReLU(),
                                                      self.encoder_k.fc)
                else:
                    # 备用方案 (如果编码器没有 fc)
                    dim_mlp = feature_dim
                    self.encoder_q = nn.Sequential(self.encoder_q,
                                                   nn.Linear(dim_mlp, dim_mlp),
                                                   nn.ReLU(),
                                                   nn.Linear(dim_mlp, feature_dim))
                    self.encoder_k = nn.Sequential(self.encoder_k,
                                                   nn.Linear(dim_mlp, dim_mlp),
                                                   nn.ReLU(),
                                                   nn.Linear(dim_mlp, feature_dim))

            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

            self.register_buffer("queue", torch.randn(feature_dim, queue_size))
            self.queue = F.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        gpu_index = keys.device.index if keys.device.type == 'cuda' else 0
        self.queue[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys.T

    @torch.no_grad()
    def update_ptr(self, batch_size):
        assert self.K % batch_size == 0
        self.queue_ptr[0] = (self.queue_ptr[0] + batch_size) % self.K

    # [ MODIFIED ] --------------------------------------------------------
    def forward(self, im_q_extreme, im_q, im_k=None, nnm=False, topk=1, return_features=False):
        # [ END MODIFIED ] ----------------------------------------------------
        """
        Input:
            im_q: a batch of query sequences (normally augmented)
            im_k: a batch of key sequences
            im_q_extreme: a batch of extremely augmented query sequences
            return_features: (bool) HGC-MAE 标志，用于返回 z 特征图
        """

        if nnm:
            # KNN 挖掘 (目前未使用)
            return self.nearest_neighbors_mining(im_q, im_k, im_q_extreme, topk)

        if not self.pretrain:
            # 线性评估模式
            return self.encoder_q(im_q)

        device = im_q.device

        # [ MODIFIED ] --------------------------------------------------------
        # 为 MAE 路径获取特征图 z
        # 我们从 'im_q' (正常增强) 获取 z，并要求返回特征
        q, z_features = self.encoder_q(im_q, return_features=True)  # q=(N,C), z=(N*M,C,T,V)

        # 为 CL 路径获取特征
        q_extreme, q_extreme_drop = self.encoder_q(im_q_extreme, drop=True)  # (N,C), (N,C)
        # [ END MODIFIED ] ----------------------------------------------------

        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)

        # Compute key features
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

        # Compute logits of normally augmented query using Einstein sum
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device)

        # Compute logits_e of extremely augmented query
        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1)
        logits_e /= self.T
        logits_e = torch.softmax(logits_e, dim=1)

        # Compute logits_ed of dropped extremely augmented query
        l_pos_ed = torch.einsum('nc,nc->n', [q_extreme_drop, k]).unsqueeze(-1)
        l_neg_ed = torch.einsum('nc,ck->nk', [q_extreme_drop, self.queue.clone().detach()])
        logits_ed = torch.cat([l_pos_ed, l_neg_ed], dim=1)
        logits_ed /= self.T
        logits_ed = torch.softmax(logits_ed, dim=1)

        # Use the distribution of normally augmented view as supervision label
        labels_ddm = logits.clone().detach()
        labels_ddm = torch.softmax(labels_ddm, dim=1)
        labels_ddm = labels_ddm.detach()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels, logits_e, logits_ed, labels_ddm, z_features

    def nearest_neighbors_mining(self, im_q, im_k, im_q_extreme, topk=1):
        """
        Nearest Neighbors Mining (NNM)
        """
        # ✅ 获取设备
        device = im_q.device

        # Obtain features
        q = self.encoder_q(im_q)
        q_extreme, q_extreme_drop = self.encoder_q(im_q_extreme, drop=True)

        # Normalize
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)

        # Compute key features
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)
            k = F.normalize(k, dim=1)

        # Compute similarities
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])

        l_pos_ed = torch.einsum('nc,nc->n', [q_extreme_drop, k]).unsqueeze(-1)
        l_neg_ed = torch.einsum('nc,ck->nk', [q_extreme_drop, self.queue.clone().detach()])

        # Concatenate logits
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1)
        logits_ed = torch.cat([l_pos_ed, l_neg_ed], dim=1)

        # Apply temperature
        logits /= self.T
        logits_e /= self.T
        logits_ed /= self.T

        # Softmax
        logits_e = torch.softmax(logits_e, dim=1)
        logits_ed = torch.softmax(logits_ed, dim=1)

        # Use the distribution of normally augmented view as supervision label
        labels_ddm = logits.clone().detach()
        labels_ddm = torch.softmax(labels_ddm, dim=1)
        labels_ddm = labels_ddm.detach()

        # Nearest neighbors mining
        _, topkdix = torch.topk(l_neg, topk, dim=1)
        _, topkdix_e = torch.topk(l_neg_e, topk, dim=1)
        _, topkdix_ed = torch.topk(l_neg_ed, topk, dim=1)

        # Create positive mask
        topk_onehot = torch.zeros_like(l_neg)
        topk_onehot.scatter_(1, topkdix, 1)
        topk_onehot.scatter_(1, topkdix_e, 1)
        topk_onehot.scatter_(1, topkdix_ed, 1)

        # ✅ 修改：使用动态设备
        pos_mask = torch.cat([torch.ones(topk_onehot.size(0), 1, device=device), topk_onehot], dim=1)

        # Dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, pos_mask, logits_e, logits_ed, labels_ddm