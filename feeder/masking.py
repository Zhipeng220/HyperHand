import torch
import numpy as np


# ---------------------------------------------------------------------------------
# 创新点 5: 语义感知掩码 (Semantic-Aware Masking)
# ---------------------------------------------------------------------------------

def random_joint_mask(num_joints, mask_ratio):
    """
    随机选择要遮挡的关节索引 (标准 MAE 方法)
    """
    num_masked_joints = int(num_joints * mask_ratio)
    masked_indices = np.random.permutation(num_joints)[:num_masked_joints]
    visible_indices = np.setdiff1d(np.arange(num_joints), masked_indices)
    return masked_indices, visible_indices


def kinematic_joint_mask(x_data, mask_ratio):
    """
    基于运动学的遮挡 (优先遮挡运动最剧烈的关节)
    x_data: (N, C, T, V, M)
    """
    N, C, T, V, M = x_data.shape
    num_masked_joints = int(V * mask_ratio)

    # 1. 计算每个关节的总运动量
    # (N, C, T, V, M) -> (N, V, T)
    motion = torch.sum(torch.abs(x_data[:, :, 1:, :, :] - x_data[:, :, :-1, :, :]), dim=(1, 4))
    # (N, V, T) -> (N, V)
    joint_motion = torch.sum(motion, dim=2)

    # 2. 在 batch 维度上平均运动量，得到 (V,)
    avg_joint_motion = torch.mean(joint_motion, dim=0)

    # 3. 找到运动最剧烈 (top-k) 的关节
    # (V,)
    _, topk_indices = torch.topk(avg_joint_motion, num_masked_joints)

    masked_indices = topk_indices.numpy()
    visible_indices = np.setdiff1d(np.arange(V), masked_indices)
    return masked_indices, visible_indices


def structured_joint_mask(num_joints, mask_ratio, joint_groups):
    """
    结构化遮挡 (一次性遮挡整个身体部位)
    """
    num_masked_joints = int(num_joints * mask_ratio)

    masked_indices = []
    # 随机打乱部位顺序
    np.random.shuffle(joint_groups)

    # 贪婪地添加部位，直到达到 mask 比例
    for group in joint_groups:
        if len(masked_indices) < num_masked_joints:
            masked_indices.extend(group)
        else:
            break

    masked_indices = np.array(list(set(masked_indices)))  # 去重
    visible_indices = np.setdiff1d(np.arange(num_joints), masked_indices)
    return masked_indices, visible_indices


def perform_masking(x_data, mask_ratio, strategy='random', num_joints=21):
    """
    主函数，用于执行遮挡策略
    x_data: (N, C, T, V, M)
    """

    # EgoGesture 21 关节的手部结构 (示例)
    # 0: 掌根 (Wrist)
    # 1-4: 拇指 (Thumb)
    # 5-8: 食指 (Index)
    # 9-12: 中指 (Middle)
    # 13-16: 无名指 (Ring)
    # 17-20: 小指 (Pinky)
    EGOGESTURE_GROUPS = [
        [1, 2, 3, 4],  # 拇指
        [5, 6, 7, 8],  # 食指
        [9, 10, 11, 12],  # 中指
        [13, 14, 15, 16],  # 无名指
        [17, 18, 19, 20],  # 小指
        [0]  # 掌根
    ]

    if strategy == 'random':
        masked_indices, visible_indices = random_joint_mask(num_joints, mask_ratio)

    elif strategy == 'kinematic':
        masked_indices, visible_indices = kinematic_joint_mask(x_data, mask_ratio)

    elif strategy == 'structured':
        # 假设我们使用的是 EgoGesture (21 关节)
        groups = EGOGESTURE_GROUPS if num_joints == 21 else []
        masked_indices, visible_indices = structured_joint_mask(num_joints, mask_ratio, groups)

    else:
        raise ValueError(f"Unknown masking strategy: {strategy}")

    # 创建一个 (V,) 的布尔掩码
    # mask: True 表示被遮挡, False 表示可见
    mask = torch.zeros(num_joints, dtype=torch.bool)
    mask[masked_indices] = True

    return mask, masked_indices, visible_indices