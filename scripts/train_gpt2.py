import math
import os
import time
import torch
import tiktoken

import torch.distributed as dist

from torch.nn import functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from gpt.model import GPT, GPTConfig
from gpt.dataloader import DataLoaderLite
from gpt.utils import get_most_likely_row, append_to_log, save_checkpoint
from tasks.hellaswag import render_example, iterate_examples



# Setup Distributed Data Parallel (DDP)
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get("RANK", -1)) != -1 # Is this a DDP run?
if ddp:
    # Use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "For now I think we need CUDA for DDP"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # This process will do logging, checkpointing etc.

else:
    # Vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    # Attempt to auto-detect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    print(f"using device: {device}")

device_type = "cuda" if device.startswith("cuda") else device
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 524288 # 2**19, ~0.5M in number of tokens
B = 32 # Micro Batch Size
T = 1024 # Sequence Length
assert total_batch_size % (B * T * ddp_world_size) == 0, "Make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"Total desired batch size: {total_batch_size}")
    print(f"=> Calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train", master_process=master_process)
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val", master_process=master_process)

# Setting data type
torch.set_float32_matmul_precision('high')

use_compile = True # Compiled model is only used for training and val loss, HellaSwag and sampling use core_model
enc = tiktoken.get_encoding("gpt2")

# Create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")

# Resume from the latest checkpoint in log_dir when RESUME=1 is set, e.g. RESUME=1 uv run torchrun ...
resume = os.environ.get("RESUME", "0") == "1"
checkpoint = None
start_step = 0
if resume:
    checkpoints = sorted(f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt"))
    assert len(checkpoints) > 0, f"RESUME=1 but no checkpoints found in {log_dir}"
    checkpoint_path = os.path.join(log_dir, checkpoints[-1])
    # weights_only=False because the checkpoint also contains GPTConfig, safe since we wrote this file ourselves
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    start_step = checkpoint['step']
    if master_process:
        print(f"Resuming from {checkpoint_path} at step {start_step}")

if master_process:
    if resume and os.path.exists(log_file):
        # Drop log lines from the steps we are about to redo, so they don't show up twice
        with open(log_file, "r") as f:
            lines = [line for line in f if int(line.split()[0]) < start_step]
        with open(log_file, "w") as f:
            f.writelines(lines)
    else:
        with open(log_file, "w") as f: # Open for writing to clear the file
            pass

# Get model
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
if checkpoint is not None:
    model.load_state_dict(checkpoint['model']) # Load before compile and DDP so the keys match the plain GPT module
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # Always contains the "raw" unwrapped model
core_model = raw_model._orig_mod if use_compile else raw_model # Plain GPT module (not compiled), shares the same weights

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073 * 4

def get_lr(it):
    
    # 1) Linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps

    # 2) If it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    
    # 3) In between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# Optimize!
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device=device)
if checkpoint is not None:
    optimizer.load_state_dict(checkpoint['optimizer'])
    train_loader.load_state(**checkpoint['train_loader'])
    del checkpoint # Free the memory, everything has been restored

for step in range(start_step, max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # Once in a while, evaluate our validatio loss
    if step % 100 == 0 or last_step:
        model.eval()
        val_loader.reset()
        
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20

            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

        if master_process:
            print(f"Validation loss: {val_loss_accum.item():.4f}")
            append_to_log(log_file, f"{step} val {val_loss_accum.item():.4f}\n")
            if step > start_step and (step % 2000 == 0 or last_step): # step > start_step so a resumed run doesn't re-save the checkpoint it loaded
                # Write model checkpoints, with enough state to resume training later
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': core_model.state_dict(), # core_model so the keys have no "_orig_mod." prefix
                    'config': core_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item(),
                    'optimizer': optimizer.state_dict(),
                    'train_loader': {
                        'current_shard': train_loader.current_shard,
                        'current_position': train_loader.current_position,
                    },
                }
                save_checkpoint(checkpoint, checkpoint_path)

    # Once in a while evaluate hellaswag
    if step % 1000 == 0 or last_step:
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = core_model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            append_to_log(log_file, f"{step} hella {acc_norm:.4f}\n")

    # Once in awhile generate from the model (except step 0, which is noise)
    if (step > 0 and step % 250 == 0) or last_step:
        model.eval()
        num_return_sequences = 4
        max_length = 32

        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
        x = tokens.to(device)

        # Generate! Right now x is (B, T) where B = 5, T = 8
        # Set the seed to 42
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)

        while x.size(1) < max_length:
            # Forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = core_model(x) # (B, T, vocab_size)

                # Take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)

                # Get the probabilities
                probs = F.softmax(logits, dim=-1)

                # Do top-k sampling of 50 (hugging face pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

                # Select a token from the top-k probabilities
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)

                # Gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)

                # Append to the sequence
                x = torch.cat((x, xcol), dim=1)


        # Print the generated text
        for i in range(num_return_sequences):
            tokens = x[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # Training Loop
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0

    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)

        # We have to sacle the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # Addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    optimizer.step()

    torch.cuda.synchronize() # Waiting for GPU to stop this iteration so we get the accurate time
    t1 = time.time()
    dt = (t1 - t0) # Time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt

    if master_process:
        print(f"Step {step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

if ddp:
    destroy_process_group()