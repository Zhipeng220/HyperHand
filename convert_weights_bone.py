import torch

# --- CONFIGURATION ---
original_file = 'work_dir/egogesture/pretext_bone/epoch080_acc33.66_model.pt'
new_file = 'work_dir/egogesture/pretext_bone/epoch080_BACKBONE_ONLY.pt'
prefix_to_remove = 'encoder_q.'
# ---------------------

print(f"Loading weights from: {original_file}")
full_checkpoint = torch.load(original_file, map_location='cpu')

new_state_dict = {}

# Loop through all keys in the original checkpoint
for key, value in full_checkpoint.items():

    # 🔴 THIS IS THE FIX:
    # We only save keys that start with 'encoder_q.'
    # AND do NOT start with 'encoder_q.fc.' (the old classification head)
    if key.startswith(prefix_to_remove) and not key.startswith("encoder_q.fc."):
        # Remove the prefix
        new_key = key.replace(prefix_to_remove, "")

        # Add to our new dictionary
        new_state_dict[new_key] = value
        print(f"Converting: {key}  --->  {new_key}")

print("\nAll other keys (encoder_k, queue, fc) successfully ignored.")

# Save the new file, overwriting the old one
torch.save(new_state_dict, new_file)
print(f"\n✅ Success! Backbone-only weights saved to: {new_file}")