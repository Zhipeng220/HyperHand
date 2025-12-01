import sys
import argparse
import yaml
import math
import random
import numpy as np
from itertools import chain

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor
from .knn_monitor import knn_monitor

# [ NEW ] HGC-MAE 导入
from feeder.masking import perform_masking


class AimCLR_Processor(Processor):
    """
        Processor for Pretraining HGC-MAE (AimCLR + MAE)
    """

    def load_model(self):
        # 1. 加载 HGC 编码器 (self.model)
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(self.init_weights)

        # 2. [NEW] 加载 MAE 解码器 (self.decoder)
        Decoder = import_class('net.decoder.Decoder')
        # 从编码器获取图 A
        if hasattr(self.model, 'module'):
            A = self.model.module.encoder_q.graph.A
        else:
            A = self.model.encoder_q.graph.A

        # 从 config 获取参数
        enc_args = self.arg.model_args

        self.decoder = Decoder(
            in_channels=enc_args.get('feature_dim', 256),
            out_channels=enc_args.get('in_channels', 3),
            A=A,
            adaptive=enc_args.get('adaptive', True),
            num_person=enc_args.get('num_person', 2)
        ).to(self.dev)
        self.decoder.apply(self.init_weights)  # [NEW] 初始化解码器权重

        # 3. [NEW] 加载 MAE 损失
        # 重建损失
        self.reconstruction_loss = nn.MSELoss().to(self.dev)

        # 物理损失 (创新点 2)
        AnatomicalLoss = import_class('net.physics_loss.AnatomicalLoss')
        self.physics_loss = AnatomicalLoss(
            num_joints=enc_args.get('num_point', 21),
            dataset='egogesture'  # 假设，如果跑NTU需修改此处
        ).to(self.dev)

        # 对齐损失 (创新点 4)
        self.alignment_loss = CKA_loss  # [NEW] 使用 CKA Loss

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):  # [NEW] 初始化 Conv2d
            conv_init(m)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def load_optimizer(self):
        # [MODIFIED] 优化器需要训练编码器和解码器
        all_params = chain(self.model.parameters(), self.decoder.parameters())

        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                all_params,
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                all_params,
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def load_data(self):
        super().load_data()
        if self.arg.knn_monitor:
            if hasattr(self.arg, 'memory_feeder_args'):
                self.io.print_log('Loading memory data for KNN monitor...')
                if hasattr(self.arg, 'memory_feeder') and self.arg.memory_feeder:
                    memory_feeder = import_class(self.arg.memory_feeder)
                else:
                    memory_feeder = import_class(self.arg.train_feeder)
                self.data_loader['memory'] = torch.utils.data.DataLoader(
                    dataset=memory_feeder(**self.arg.memory_feeder_args),
                    batch_size=self.arg.test_batch_size,
                    shuffle=False,
                    num_workers=self.arg.num_worker,
                    drop_last=False,
                    worker_init_fn=self.init_seed if hasattr(self, 'init_seed') else None
                )
                self.io.print_log(f'Memory data loaded: {len(self.data_loader["memory"].dataset)} samples')
            else:
                self.io.print_log('Warning: knn_monitor is True but memory_feeder_args not found!')
                self.io.print_log('KNN monitor will be disabled.')
                self.arg.knn_monitor = False

    def adjust_lr(self):
        lr = self.arg.base_lr
        epoch = self.meta_info['epoch']  # 当前 epoch (从1开始)
        num_epoch = self.arg.num_epoch

        # [MODIFIED] 支持 Cosine Annealing
        if hasattr(self.arg, 'lr_scheduler') and self.arg.lr_scheduler == 'cosine':
            # Warmup 策略
            warmup_epoch = getattr(self.arg, 'warmup_epoch', 5)

            if epoch <= warmup_epoch:
                # 线性预热
                lr = self.arg.base_lr * (epoch / warmup_epoch)
            else:
                # 余弦退火
                curr_epoch = epoch - warmup_epoch
                total_epoch = num_epoch - warmup_epoch
                lr = self.arg.base_lr * 0.5 * (1 + math.cos(math.pi * curr_epoch / total_epoch))

        # [保留] 原有的 Step 逻辑 (兼容)
        elif self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                    self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
        else:
            lr = self.arg.base_lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.lr = lr

    def train(self, epoch):
        self.model.train()
        self.decoder.train()
        self.adjust_lr()
        loader = self.data_loader['train']

        loss_value = []
        cl_loss_value = []
        rec_loss_value = []
        phy_loss_value = []
        align_loss_value = []

        cl_criterion = nn.CrossEntropyLoss()
        kl_weight = self.arg.kl_weight

        # ---------------------------------------------------------------------
        # 创新点 3: "强到弱" 物理先验调度 (Strong-to-Weak Anatomical Scheduling)
        # ---------------------------------------------------------------------
        # 余弦退火：从 lambda_phy_max 衰减到 0
        lambda_anat = self.arg.lambda_phy_max * 0.5 * (1 + math.cos(math.pi * (epoch - 1) / self.arg.num_epoch))

        # ---------------------------------------------------------------------
        # [NEW] 创新点 5: 动态 Mask Ratio 调度 (Curriculum Masking)
        # ---------------------------------------------------------------------
        target_mask_ratio = self.arg.mask_ratio
        progress = (epoch - 1) / self.arg.num_epoch
        current_mask_ratio = 0.1 + (target_mask_ratio - 0.1) * progress
        current_mask_ratio = min(current_mask_ratio, target_mask_ratio)

        if epoch == 1 and self.global_step == 0:
            self.io.print_log(f'Using KL weight (lambda_cl): {self.arg.lambda_cl}')
            self.io.print_log(f'MAE Enabled: {self.arg.use_mae}')  # Log Switch
            if self.arg.use_mae:
                self.io.print_log(f'  - Using MAE Reconstruction weight: {self.arg.lambda_rec}')
                self.io.print_log(f'  - Physics Constraints Enabled: {self.arg.use_physics}')
                if self.arg.use_physics:
                    self.io.print_log(f'    - Using Max Physics weight: {self.arg.lambda_phy_max}')
                self.io.print_log(f'  - Using MAE Alignment weight: {self.arg.lambda_align}')
                self.io.print_log(
                    f'  - Using Masking Strategy: {self.arg.mask_strategy} (Target Ratio: {target_mask_ratio})')
            self.io.print_log(f'Gradient Accumulation Steps: {self.arg.grad_accum_steps}')

        if self.arg.use_mae and self.arg.use_physics:
            self.io.print_log(f'Epoch {epoch}: Physics Loss Weight (lambda_anat) = {lambda_anat:.4f}')

        if self.arg.use_mae:
            self.io.print_log(f'Epoch {epoch}: Current Mask Ratio (Curriculum) = {current_mask_ratio:.4f}')

        self.optimizer.zero_grad()  # 确保开始前梯度清零

        for data, label in loader:
            self.global_step += 1

            data = data.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)

            im_q = data
            im_k = data
            im_q_extreme = data

            x_original = data.clone()
            N, C, T, V, M = x_original.shape

            # -----------------------------------------------------------------
            # 1. CL 路径 (HGC + AimCLR)
            # -----------------------------------------------------------------
            logits, labels_ce, logits_e, logits_ed, labels_ddm, z_features_cl = self.model(
                im_q_extreme, im_q, im_k, return_features=True
            )

            loss_ce = cl_criterion(logits, labels_ce)
            loss_kl_e = F.kl_div(F.log_softmax(logits_e, dim=1), labels_ddm, reduction='batchmean')
            loss_kl_ed = F.kl_div(F.log_softmax(logits_ed, dim=1), labels_ddm, reduction='batchmean')
            loss_cl = loss_ce + (loss_kl_e + loss_kl_ed) * kl_weight

            if hasattr(self.model, 'module'):
                self.model.module.update_ptr(im_q.size(0))
            else:
                self.model.update_ptr(im_q.size(0))

            # 初始化 MAE 相关 Loss 为 0
            loss_rec = torch.tensor(0.0).to(self.dev)
            loss_phy = torch.tensor(0.0).to(self.dev)
            loss_align = torch.tensor(0.0).to(self.dev)

            # -----------------------------------------------------------------
            # 2. MAE 路径 (HGC-MAE) - 仅当 use_mae=True 时执行
            # -----------------------------------------------------------------
            if self.arg.use_mae:
                mask, masked_indices, visible_indices = perform_masking(
                    x_original,
                    mask_ratio=current_mask_ratio,
                    strategy=self.arg.mask_strategy,
                    num_joints=V
                )
                mask = mask.to(self.dev, non_blocking=True)

                x_hat_T_out = self.decoder(z_features_cl, mask)
                x_hat = x_hat_T_out[:, :, :T, :, :]

                loss_rec = self.reconstruction_loss(
                    x_hat * mask.view(1, 1, 1, V, 1),
                    x_original * mask.view(1, 1, 1, V, 1)
                )

                # 3. 物理约束 (仅当 use_physics=True 且 use_mae=True 时)
                if self.arg.use_physics:
                    loss_phy = self.physics_loss(x_hat) * lambda_anat

                # 4. 对齐损失 (依赖 x_hat，因此也在 use_mae 下)
                # 只有在有重建任务时，对齐特征才有意义
                if self.arg.lambda_align > 0:
                    loss_align = self.alignment_loss(z_features_cl, x_hat.detach())

            # -----------------------------------------------------------------
            # 5. 汇总损失
            # -----------------------------------------------------------------
            total_loss = (loss_cl * self.arg.lambda_cl) + \
                         (loss_rec * self.arg.lambda_rec) + \
                         (loss_phy) + \
                         (loss_align * self.arg.lambda_align)

            # [ FIX ] 梯度累积逻辑
            total_loss = total_loss / self.arg.grad_accum_steps
            total_loss.backward()

            # 仅在特定的步数更新参数
            if self.global_step % self.arg.grad_accum_steps == 0:
                if self.arg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.arg.grad_clip_norm)
                    torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.arg.grad_clip_norm)

                self.optimizer.step()
                self.optimizer.zero_grad()

            # -----------------------------------------------------------------
            # 6. 统计
            # -----------------------------------------------------------------
            self.iter_info['loss'] = total_loss.data.item() * self.arg.grad_accum_steps
            self.iter_info['loss_cl'] = loss_cl.data.item()
            self.iter_info['loss_rec'] = loss_rec.data.item()
            self.iter_info['loss_phy'] = loss_phy.data.item()
            self.iter_info['loss_align'] = loss_align.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)

            loss_value.append(self.iter_info['loss'])
            cl_loss_value.append(self.iter_info['loss_cl'])
            rec_loss_value.append(self.iter_info['loss_rec'])
            phy_loss_value.append(self.iter_info['loss_phy'])
            align_loss_value.append(self.iter_info['loss_align'])

            self.show_iter_info()
            self.meta_info['iter'] += 1
            self.train_log_writer(epoch)

        self.epoch_info['train_mean_loss'] = np.mean(loss_value)
        self.epoch_info['train_mean_cl'] = np.mean(cl_loss_value)
        self.epoch_info['train_mean_rec'] = np.mean(rec_loss_value)
        self.epoch_info['train_mean_phy'] = np.mean(phy_loss_value)
        self.epoch_info['train_mean_align'] = np.mean(align_loss_value)

        self.show_epoch_info()

    def test(self, epoch):
        self.model.eval()
        if self.arg.knn_monitor:
            if 'memory' not in self.data_loader:
                self.io.print_log('Warning: memory data loader not found, skipping KNN monitor.')
                self.current_result = 0.0
            else:
                self.io.print_log('Running KNN monitor...')
                if hasattr(self.model, 'module'):
                    feature_extractor = self.model.module.encoder_q
                else:
                    feature_extractor = self.model.encoder_q

                acc = knn_monitor(
                    feature_extractor,
                    self.data_loader['memory'],
                    self.data_loader['test'],
                    epoch,
                    k=self.arg.knn_k,
                    t=self.arg.knn_t,
                    hide_progress=True
                )
                self.current_result = acc
                self.io.print_log(f'KNN accuracy: {acc:.2f}%')
        else:
            self.current_result = 0.0

        self.eval_info['test_acc'] = self.current_result
        self.eval_info['eval_mean_loss'] = 0.0
        self.show_eval_info()
        self.eval_log_writer(epoch)

    @staticmethod
    def get_parser(add_help=False):
        # parameter priority: command line > config > default
        parser = argparse.ArgumentParser(
            parents=[Processor.get_parser(add_help=False)],
            add_help=add_help,
            description='Pretrain HGC-MAE (AimCLR + MAE)')

        # Learning rate and optimizer
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--lr_scheduler', type=str, default='step', help='learning rate scheduler (step or cosine)')
        parser.add_argument('--warmup_epoch', type=int, default=5, help='warmup epochs for cosine annealing')
        parser.add_argument('--step', type=int, default=[], nargs='+',
                            help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')

        # KL 权重参数
        parser.add_argument('--kl_weight', type=float, default=0.5, help='weight for KL divergence loss')

        # KNN monitor
        parser.add_argument('--knn_monitor', type=str2bool, default=True, help='knn monitor')
        parser.add_argument('--knn_k', type=int, default=200, help='knn k')
        parser.add_argument('--knn_t', type=float, default=0.1, help='knn t')

        # Data augmentation
        parser.add_argument('--aug_method', type=str, default='aimclr', help='augmentation method')
        parser.add_argument('--shear_amplitude', type=float, default=0.5, help='amplitude of shear augmentation')
        parser.add_argument('--temperal_padding_ratio', type=int, default=6, help='temporal padding ratio')

        # Standard params
        parser.add_argument('--stream', type=str, default='joint', help='stream type: joint, bone, or motion')
        parser.add_argument('--seed', type=int, default=1, help='random seed')
        parser.add_argument('--feeder', type=str, default='feeder.feeder', help='data feeder')
        parser.add_argument('--memory_feeder', type=str, default=None, help='data feeder for memory bank')
        parser.add_argument('--memory_feeder_args', action=DictAction, default=dict(), help='memory data loader args')

        # [ NEW ] HGC-MAE 损失权重
        parser.add_argument('--lambda_cl', type=float, default=1.0, help='Weight for CL loss')
        parser.add_argument('--lambda_rec', type=float, default=1.0, help='Weight for Reconstruction loss')
        parser.add_argument('--lambda_phy_max', type=float, default=1.0, help='Max weight for Physics loss')
        parser.add_argument('--lambda_align', type=float, default=0.1, help='Weight for Alignment loss')

        # [ NEW ] HGC-MAE 掩码参数
        parser.add_argument('--mask_ratio', type=float, default=0.7, help='Ratio of joints to mask for MAE')
        parser.add_argument('--mask_strategy', type=str, default='structured', help='Masking strategy')

        # [ NEW ] 梯度裁剪和累积
        parser.add_argument('--grad_clip_norm', type=float, default=1.0, help='Max norm for gradient clipping')
        parser.add_argument('--grad_accum_steps', type=int, default=1, help='number of steps to accumulate gradients')

        # [ NEW ] 模块开关 (布尔值)
        parser.add_argument('--use_mae', type=str2bool, default=True, help='Enable MAE branch (Masked Reconstruction)')
        parser.add_argument('--use_physics', type=str2bool, default=True,
                            help='Enable Physics constraints (Dependent on MAE)')

        return parser


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


# -----------------------------------------------------------------------------
# [New] CKA Loss Implementation
# -----------------------------------------------------------------------------

def centered_gram_matrix(x):
    """
    计算中心化 Gram 矩阵
    x: (N, D)
    """
    n = x.size(0)
    # 中心化 (Centering)
    x_centered = x - x.mean(dim=0, keepdim=True)
    # 计算 Gram 矩阵 (N, N)
    gram = torch.mm(x_centered, x_centered.t())
    return gram


def CKA_loss(x, y):
    """
    计算 Linear CKA 损失
    Args:
        x: Encoder 特征, 形状通常为 (N*M, C, T, V)
        y: Decoder 重建或特征, 形状通常为 (N, C, T, V, M) 或 (N*M, C, T, V)
    Returns:
        loss: 1 - CKA_score (范围 0~1)
    """
    # 1. 维度预处理与对齐

    # 如果 y 是 (N, C, T, V, M) 格式 (如 x_hat)，转换为 (N*M, C, T, V)
    if y.dim() == 5:
        N, C, T, V, M = y.shape
        # Permute to (N, M, C, T, V) -> View (N*M, C, T, V)
        y = y.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)

    # 如果 x 是 (N, C, T, V, M) 格式，同样转换
    if x.dim() == 5:
        N, C, T, V, M = x.shape
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)

    # 2. 全局平均池化 (Global Average Pooling)
    # 将时空特征 (N*M, C, T, V) 压缩为样本语义向量 (N*M, C)
    if x.dim() == 4:
        x = x.mean(dim=[2, 3])  # (Batch, Feature_Dim_X)
    if y.dim() == 4:
        y = y.mean(dim=[2, 3])  # (Batch, Feature_Dim_Y)

    # 确保 Batch Size 一致
    assert x.size(0) == y.size(0), f"CKA Batch size mismatch: {x.size(0)} vs {y.size(0)}"

    # 3. 计算 Gram 矩阵
    gram_x = centered_gram_matrix(x)
    gram_y = centered_gram_matrix(y)

    # 4. 计算 Linear CKA
    # CKA(X, Y) = HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y))
    scaled_hsic = torch.sum(gram_x * gram_y)
    norm_x = torch.sqrt(torch.sum(gram_x * gram_x))
    norm_y = torch.sqrt(torch.sum(gram_y * gram_y))

    cka_score = scaled_hsic / (norm_x * norm_y + 1e-6)

    # 5. 返回 Loss (最大化相似度 = 最小化 1 - CKA)
    return 1.0 - cka_score