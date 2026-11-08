import inspect
import torch

import torch.nn as nn

from torch.nn import functional as F
from dataclasses import dataclass

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

    def forward(self, x, kv_cache=None, layer_idx=None):
        """Multi-head causal self-attention, optionally reading earlier keys and values from a KV cache.

        Args:
            x: (B, T, C). Without a cache, the whole sequence (training, rl_loss). With one, the
                prompt on the first call (prefill), then T = 1, just the newest token.
            kv_cache: a KVCache from gpt.engine, or None. If given, this call's k and v are stored
                in it, and attention runs against every key and value cached so far.
            layer_idx: this block's index, so the cache uses the right slot.

        Returns:
            (B, T, C), the same as without a cache.

        The mask: is_causal=T > 1, not True. With one new token, q has 1 row but k has pos+1, and
        PyTorch lines a causal mask up top-left, so that row would only see position 0. One token
        may see everything before it, so it needs no mask. T > 1 only happens with an empty cache
        or none, where q and k have the same length and the usual causal mask is right.

        Example (150-token prompt, then one step, 16 samples):
            prefill: x (16,150,768) -> k,v after insert (16,12,150,64) -> causal       -> (16,150,768)
            step 1:  x (16,1,768)   -> k,v after insert (16,12,151,64) -> no mask      -> (16,1,768)
        """

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
    
        if kv_cache is not None:
            k, v = kv_cache.insert(layer_idx, k, v)

        # Replaced the above 4 lines with Flash Attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1)

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

    def forward(self, x, kv_cache=None, layer_idx=None):
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
        x = x + self.attn(self.ln_1(x), kv_cache, layer_idx)
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

    def forward(self, idx, targets=None, loss_reduction="mean", kv_cache=None):
        """Run the model on idx and return logits, plus the loss if targets are given.

        Args:
            idx: (B, T) token ids. Without a cache, the whole sequence. With one, the prompt on the
                first call, then one new token per call.
            targets: (B, T) next-token ids, -1 = ignored (prompt, padding). None = no loss.
            loss_reduction: "mean" (training) or "none" (one loss per token, (B, T), for RL).
            kv_cache: a KVCache from gpt.engine, or None (training, rl_loss, evaluate_categorical).

        With a cache:
            - Positions start at kv_cache.pos, not 0, so a new token gets the same position
              embedding it would have had in the full sequence.
            - Each block gets its index, so it stores its keys and values in its own slot.
            - kv_cache.pos moves forward by T once, after the last block. All blocks insert at
              the same position, so none of them can move it.

        Returns:
            logits (B, T, vocab_size), and loss: None, a scalar ("mean"), or (B, T) ("none").

        Example (150-token prompt, 16 samples):
            model(prompt (16,150), kv_cache=cache) -> logits (16,150,V), positions 0..149,   pos 0 -> 150
            model(token  (16,1),   kv_cache=cache) -> logits (16,1,V),   position 150,       pos 150 -> 151

        Gotcha: start + T may not pass block_size (1024), because wpe has no row for it. Without a
        cache, generate used to crop instead; with one, generation has to stop there.
        """

        start = kv_cache.pos if kv_cache is not None else 0
        # idx is of shape (B, T)
        B, T = idx.size()
        assert start + T <= self.config.block_size, f"Cannot forward to position {start + T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(start, start + T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for i, block in enumerate(self.transformer.h):
            x = block(x, kv_cache=kv_cache, layer_idx=i)

        if kv_cache is not None:
            kv_cache.pos += T

        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # ignore_index = -1: targets set to -1 (prompt tokens, padding) don't count towards loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            if loss_reduction == "none":
                loss = loss.view(B, T) # One loss per token for RL
                
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