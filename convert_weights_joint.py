import torch

# --- 1. 修改配置 ---
# 🔴 指向您新的“关节点”最佳模型
original_file = 'work_dir/egogesture/aimclr_pretext_joint/epoch070_acc32.54_model.pt'

# 🔴 我们将创建一个新的“关节点”骨干文件
new_file = 'work_dir/egogesture/aimclr_pretext_joint/epoch070_JOINT_BACKBONE_ONLY.pt'
prefix_to_remove = 'encoder_q.'
# ---------------------

print(f"Loading weights from: {original_file}")
full_checkpoint = torch.load(original_file, map_location='cpu')

new_state_dict = {}

# 循环所有权重
for key, value in full_checkpoint.items():

    # 保持我们的修复：只保存骨干网络 (encoder_q)，并丢弃预训练头 (fc)
    if key.startswith(prefix_to_remove) and not key.startswith("encoder_q.fc."):

        # 移除 "encoder_q." 前缀
        new_key = key.replace(prefix_to_remove, "")

        new_state_dict[new_key] = value
        print(f"Converting: {key}  --->  {new_key}")

print("\nAll other keys (encoder_k, queue, fc) successfully ignored.")

# 保存新的文件
torch.save(new_state_dict, new_file)
print(f"\n✅ Success! JOINT backbone weights saved to: {new_file}")