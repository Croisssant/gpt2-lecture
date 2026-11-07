import os
import torch
import time

from torch.nn import functional as F
from gpt.model import GPT

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Helper function for HellaSwag eval
# Takes tokens, mask, and logits, returns the index of the completion with the lowest loss
def get_most_likely_row(tokens, mask, logits):
    # Evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)

    # Now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # We must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask

    # Sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)

    # Now we have a loss for each of the 4 completions
    # The one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Helpers for writing to /data, an NFS mount that sometimes fails writes with "OSError: [Errno 22] Invalid argument"
# A failed write should not crash a run that has been training for hours, so retry, then warn and carry on
def append_to_log(log_file, line):
    try:
        with open(log_file, "a") as f:
            f.write(line)
    except OSError as e:
        print(f"WARNING: could not write to {log_file}: {e}")


def save_checkpoint(checkpoint, checkpoint_path, attempts=3):
    tmp_path = checkpoint_path + ".tmp"
    for attempt in range(1, attempts + 1):
        try:
            # Save to a temporary file and rename it after, so a failed save can't leave a broken "latest" checkpoint
            torch.save(checkpoint, tmp_path)
            os.replace(tmp_path, checkpoint_path)
            return
        except (OSError, RuntimeError) as e: # torch.save reports write failures as RuntimeError
            print(f"WARNING: checkpoint save to {checkpoint_path} failed (attempt {attempt}/{attempts}): {e}")
            time.sleep(10)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    print(f"WARNING: giving up on {checkpoint_path}, training continues")


def load_checkpoint(checkpoint_path, device):
    """Rebuild a GPT from a saved checkpoint and move it to device.

    Args:
        checkpoint_path: a .pt file saved with {"model": state_dict, "config": GPTConfig, ...}.
        device: where to put the model and its tensors.

    Returns:
        (model, checkpoint): the GPT with the saved weights, and the full checkpoint dict, so the
        caller can also use checkpoint["optimizer"], ["step"] and so on.
        The model is in train mode; call model.eval() for inference.

    Pickle gotcha: pretraining checkpoints were saved from train_gpt2.py run as a script, so their
    config is stored as __main__.GPTConfig. Any script that loads one must therefore import
    GPTConfig at the top level (from gpt.model import GPT, GPTConfig), or unpickling fails with
    "Can't get attribute 'GPTConfig'". SFT checkpoints store gpt.model.GPTConfig and load anywhere.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = GPT(checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    return model, checkpoint

if __name__ == "__main__":
    import tempfile
    from gpt.model import GPTConfig

    # Save a tiny model, then check that loading gives back the same model
    torch.manual_seed(0)
    original = GPT(GPTConfig(block_size=64, vocab_size=50304, n_layer=2, n_head=2, n_embd=32)).eval()
    path = os.path.join(tempfile.mkdtemp(), "model_test.pt")
    torch.save({"model": original.state_dict(), "config": original.config, "step": 123, "optimizer": {"state": {}, "param_groups": []}}, path)

    model, checkpoint = load_checkpoint(path, "cpu")
    model.eval()

    # The same input gives exactly the same output
    x = torch.randint(0, 50304, (2, 10))
    assert torch.equal(model(x)[0], original(x)[0])

    # The config and other entries come back too
    assert model.config == original.config
    assert checkpoint["step"] == 123 and "optimizer" in checkpoint

    # Weight tying is still in place
    assert model.transformer.wte.weight is model.lm_head.weight

    print("all tests passed")
