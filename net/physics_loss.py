import torch
import torch.nn as nn
import numpy as np


class AnatomicalLoss(nn.Module):
    """
    HGC-MAE 物理约束损失模块 (Anatomical Constraints)
    包含:
    1. 骨骼长度一致性 (Bone Length Consistency) - [修复] 使用时间一致性，解决单位/尺度问题
    2. 关节角度极限 (Joint Angle Limits) - [新增] 防止反向弯折和过度卷曲
    3. 手指平面性约束 (Finger Planarity Constraint) - [修复] 使用归一化体积，解决数值爆炸
    """

    def __init__(self, num_joints=21, dataset='egogesture'):
        super().__init__()

        self.dataset = dataset
        self.num_joints = num_joints

        if dataset != 'egogesture' or num_joints != 21:
            print(f"警告: PhysicsLoss 未为 {dataset} (关节数={num_joints}) 完整定义。将跳过特定约束。")
            self.bone_pairs = []
            self.angle_triplets = []
            self.finger_chains = []
        else:
            # 1. 定义骨骼连接 (父关节, 子关节)
            self.bone_pairs = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
                (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
                (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
            ]

            # 2. 定义关节角度三元组 (Parent, Middle, Child)
            # 用于计算中间关节的弯曲角度
            self.angle_triplets = [
                # Thumb
                (1, 2, 3), (2, 3, 4),
                # Index
                (0, 5, 6), (5, 6, 7), (6, 7, 8),
                # Middle
                (0, 9, 10), (9, 10, 11), (10, 11, 12),
                # Ring
                (0, 13, 14), (13, 14, 15), (14, 15, 16),
                # Pinky
                (0, 17, 18), (17, 18, 19), (18, 19, 20)
            ]

            # 最大允许弯曲角度 (约130度)，防止过度卷曲穿模
            self.max_angle = np.pi * (130 / 180)

            # 3. 定义手指链 (用于计算平面性)
            # 除了拇指，其他四指的 MCP-PIP-DIP-TIP 应近似共面
            self.finger_chains = [
                [0, 5, 6, 7, 8],  # Index
                [0, 9, 10, 11, 12],  # Middle
                [0, 13, 14, 15, 16],  # Ring
                [0, 17, 18, 19, 20]  # Pinky
            ]

    def compute_bone_length_loss(self, x_hat):
        """
        计算骨骼长度一致性损失 (Scale-Invariant)
        核心思想：同一根骨头在不同帧之间长度应该保持不变 (方差为0)。
        这比强制固定长度更鲁棒，且不受数据单位(米/像素)影响。
        """
        if not self.bone_pairs:
            return torch.tensor(0.0, device=x_hat.device)

        # 调整维度 -> (N, M, T, V, C)
        N, C, T, V, M = x_hat.shape
        x = x_hat.permute(0, 4, 2, 3, 1).contiguous()

        p_indices = [p for p, c in self.bone_pairs]
        c_indices = [c for p, c in self.bone_pairs]

        parents = x[..., p_indices, :]
        children = x[..., c_indices, :]

        # 计算当前长度: (N, M, T, Num_Bones)
        current_lengths = torch.norm(children - parents, dim=-1)

        # --- [修复核心] 使用时间维度的一致性 ---

        # 1. 计算每根骨头在时间轴上的平均长度 (N, M, 1, Num_Bones)
        mean_lengths = current_lengths.mean(dim=2, keepdim=True)

        # 2. 损失 = |当前长度 - 平均长度| / (平均长度 + epsilon)
        # 这样无论数据是像素(100)还是米(0.1)，相对误差都在同一量级
        consistency_loss = torch.abs(current_lengths - mean_lengths) / (mean_lengths + 1e-4)

        return consistency_loss.mean()

    def compute_joint_angle_loss(self, x_hat):
        """
        计算关节角度损失
        防止反向弯折 (Hyperextension) 和过度卷曲 (Hyperflexion)
        """
        if not self.angle_triplets:
            return torch.tensor(0.0, device=x_hat.device)

        N, C, T, V, M = x_hat.shape
        x = x_hat.permute(0, 4, 2, 3, 1).contiguous()  # (N, M, T, V, C)

        p_idx = [t[0] for t in self.angle_triplets]
        m_idx = [t[1] for t in self.angle_triplets]
        c_idx = [t[2] for t in self.angle_triplets]

        # 获取三个关节坐标
        pos_p = x[..., p_idx, :]  # Parent
        pos_m = x[..., m_idx, :]  # Middle (Joint)
        pos_c = x[..., c_idx, :]  # Child

        # 向量: Middle->Parent (u) 和 Middle->Child (v)
        vec_u = pos_p - pos_m
        vec_v = pos_c - pos_m

        # [关键] 归一化向量，消除尺度影响
        vec_u = vec_u / (torch.norm(vec_u, dim=-1, keepdim=True) + 1e-6)
        vec_v = vec_v / (torch.norm(vec_v, dim=-1, keepdim=True) + 1e-6)

        # 计算夹角
        cosine = torch.sum(vec_u * vec_v, dim=-1)
        cosine = torch.clamp(cosine, -0.999, 0.999)  # 防止数值误差越界
        angle = torch.acos(cosine)  # 0=折叠, pi=伸直(180度)

        # 转换为弯曲度: 0=伸直, >0=弯曲
        flexion_angle = np.pi - angle

        # 惩罚项 1: 过度弯曲 (Angle > limit)
        loss_upper = torch.relu(flexion_angle - self.max_angle)

        # 惩罚项 2: 反向弯折 (flexion < 0, 给予 -0.1 弧度的容忍度)
        loss_lower = torch.relu(-0.1 - flexion_angle)

        total_angle_loss = loss_upper + loss_lower
        return total_angle_loss.mean()

    def compute_finger_plane_loss(self, x_hat):
        """
        计算手指平面性损失 (Normalized Volume)
        [修复] 之前直接计算体积导致数值随尺度爆炸(L^3)。
        现在先对向量进行归一化，计算单位四面体的体积，范围限制在 [0, 1]。
        """
        if not self.finger_chains:
            return torch.tensor(0.0, device=x_hat.device)

        N, C, T, V, M = x_hat.shape
        x = x_hat.permute(0, 4, 2, 3, 1).contiguous()

        loss = 0.0

        for chain in self.finger_chains:
            # 选取 MCP, PIP, DIP, TIP 四个点
            p0 = x[..., chain[1], :]
            p1 = x[..., chain[2], :]
            p2 = x[..., chain[3], :]
            p3 = x[..., chain[4], :]

            # 边向量
            v01 = p1 - p0
            v02 = p2 - p0
            v03 = p3 - p0

            # [关键修复] 归一化向量
            # 这样计算出的标量积不再受骨骼长度 L 的影响 (原本是 L^3)
            v01 = v01 / (torch.norm(v01, dim=-1, keepdim=True) + 1e-6)
            v02 = v02 / (torch.norm(v02, dim=-1, keepdim=True) + 1e-6)
            v03 = v03 / (torch.norm(v03, dim=-1, keepdim=True) + 1e-6)

            # 计算归一化后的标量三重积 (Normalized Scalar Triple Product)
            # 物理意义：单位向量围成的体积。如果共面，体积为0。
            cross_prod = torch.cross(v01, v02, dim=-1)
            scalar_triple = torch.sum(cross_prod * v03, dim=-1)

            plane_loss = torch.abs(scalar_triple)
            loss += plane_loss.mean()

        return loss / len(self.finger_chains)

    def forward(self, x_hat_reconstructed):
        """
        计算总物理解剖损失
        """
        # 1. 骨骼长度一致性 (权重 1.0)
        loss_bone = self.compute_bone_length_loss(x_hat_reconstructed)

        # 2. 关节角度 (权重 1.0)
        loss_angle = self.compute_joint_angle_loss(x_hat_reconstructed)

        # 3. 手指平面性 (权重 0.5)
        loss_plane = self.compute_finger_plane_loss(x_hat_reconstructed)

        # 加权求和
        total_anat_loss = loss_bone + loss_angle + 0.5 * loss_plane

        return total_anat_loss