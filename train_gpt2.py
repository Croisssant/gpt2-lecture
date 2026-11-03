import math
import torch
import inspect
import time
import tiktoken
import os

import torch.nn as nn
import torch.distributed as dist
import numpy as np

from dataclasses import dataclass
from torch.nn import functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from hellaswag import render_example, iterate_examples

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class CausalSelfAttention (nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Key, Query, Value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # Batch Size, Sequence Length, Embedding Dimensionality (n_embd)

        # Calculate Query, Key and Value for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head = 12, hs=64, so nh * hs = C = 768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Attention (materializes the large (T, T) matrix for all the queries and keys)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        # Replaced the above 4 lines with Flash Attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # Re-assemble all head outputs side by side

        # Output Projection
        y = self.c_proj(y)

        return y


class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity then projection"""

    def __init__(self, config):
        super().__init__()

        # Convolutional Fully Connected
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd) # From paper, inner-layer has dimensionality of 4 times the dimensionality of input and output
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd) # Projection layer going back to the residual pathway
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        
        return x


class Block(nn.Module):
    """ Transformer Block: Communication followed by Computation"""

    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention (config)
       
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(self, x):
        # The residual pathway (also called the residual stream) is the main path that x takes from the embeddings straight to lm_head.
        # Fork off, compute some computation and come back
        """
        token_emb + pos_emb
        │
        x ──────────────┐
        │               ▼
        │           layer_norm_1
        │               ▼
        │        self_attention
        ▼               │
       (+) ◄────────────┘      Block 1
        │
        x ──────────────┐
        │               ▼
        │          layer_norm_2
        │               ▼
        │         feed_forward
        ▼               │
       (+) ◄────────────┘
        │
       ... Block 2, Block 3 ...
        │
     layer_norm_final
        │
     lm_head

        """
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024 # Context Length
    vocab_size: int = 50257 # Number of unique tokens
    n_layer: int = 12 # Number of Transformer block
    n_head: int = 12 # Number of attention head in an attention layer
    n_embd: int = 768 # How many floats in a vector to represent 1 token


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            # wte: Weights Token Embeddings
            wte = nn.Embedding(config.vocab_size, config.n_embd), # (B, T, C)
            # wpe: Weights Positional Embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # h: Hidden Layers
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # ln_f: Layer Norm Final
            ln_f = nn.LayerNorm(config.n_embd),
        ))
    
        # Language Modeling Head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # Init parameters
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02

        if isinstance(module, nn.Linear):
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # Start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # Create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # e.g. all weight tensrs in matmuls + embeddings decay, all biases and layernorms don't
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        non_decay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': non_decay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_non_decay_params = sum(p.numel() for p in non_decay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(non_decay_params)}, with {num_non_decay_params:,} parameters")

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

# Load and get batch data
class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {"train", "val"}

        # Get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"No Shards found for split {split}"
        if master_process:
            print(f"Found {len(shards)} shards for split {split}")
        
        self.reset()
    
    def reset(self):
        # State, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def load_state(self, current_shard, current_position):
        # Restore the position from a checkpoint
        # current_position is rank 0's position (only the master saves it), so offset it for this rank
        self.current_shard = current_shard
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = current_position + self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T) 
        y = buf[1:].view(B, T)

        # Advance the position in the tensor
        self.current_position += B * T * self.num_processes

        # If loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank

        return x, y

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

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
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

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

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