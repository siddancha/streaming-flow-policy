import torch
from collections import OrderedDict
def strip_module_prefix(state_dict):
    new_state = OrderedDict()
    for k, v in state_dict.items():
        # remove only the first occurrence of "module."
        new_key = k.replace("module.", "", 1)
        new_state[new_key] = v
    return new_state

def main(args):
    # --- 1. Load your original checkpoint ---
    ckpt = torch.load(args.orig_ckpt_path, map_location="cpu", weights_only=False)

    # --- 2. Strip the 'module.' prefix from the model state_dict ---
    old_state = ckpt['state_dicts']['model']
    new_state = strip_module_prefix(old_state)
    old_state_ema = ckpt['state_dicts']['ema_model']
    new_state_ema = strip_module_prefix(old_state_ema)

    # --- 3. Replace in the checkpoint dict ---
    ckpt['state_dicts']['model'] = new_state
    ckpt['state_dicts']['ema_model'] = new_state_ema

    # --- 4. Save out the new checkpoint ---
    torch.save(ckpt, args.new_ckpt_path)

    print(f"Saved stripped checkpoint to {args.new_ckpt_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--orig_ckpt_path', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--new_ckpt_path', type=str, required=True, help='Path to the checkpoint file')
    args = parser.parse_args()
    main(args)