
import torch

from torch.nn import functional as F
from gpt.chat_tokenizer import enc, bos, special_token_ids

PY_START = special_token_ids["<|python_start|>"]
PY_END = special_token_ids["<|python_end|>"]
OUT_START = special_token_ids["<|output_start|>"]
OUT_END = special_token_ids["<|output_end|>"]
ASSISTANT_END = special_token_ids["<|assistant_end|>"]

def sample_next_token(logits, rng, temperature=1.0, top_k=50):
    """Pick the next token for each row from the model's logits.

    Args:
        logits: (B, V) float tensor, the logits at the last position only.
        rng: a torch.Generator on the same device, so sampling is repeatable with a seed.
        temperature: 0.0 = greedy (always the most likely token). Below 1 sharpens the
            distribution (safer, more repetitive), above 1 flattens it (more varied, more mistakes).
        top_k: only sample from the k most likely tokens (None = the whole vocabulary). This cuts
            off the long tail of unlikely tokens that would otherwise be picked now and then.

    Returns:
        (B, 1) long tensor of token ids.

    Order matters: the logits are divided by the temperature *before* the softmax. top-k is then
    taken on the probabilities, and multinomial renormalizes the kept ones, which gives the same
    distribution as taking top-k of the logits first.

    Example (one row, logits [2.0, 1.0, 0.5, 3.0, -1.0], top_k=2, temperature 1.0):
        probs [0.229, 0.084, 0.051, 0.624, 0.011] -> top-2 ids [3, 0] with [0.624, 0.229]
        -> multinomial picks position 1 (27% chance after renormalizing) -> gather -> token id 0
    """
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)

  
    probs = F.softmax(logits / temperature, dim=-1)
    if top_k is not None:
        top_k = min(top_k, logits.size(-1))
      
        topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
    
        # Select a token from the top-k probabilities
        ix = torch.multinomial(topk_probs, 1, generator=rng) # (B, 1)
    
        # Gather the corresponding indices
        sampled = torch.gather(topk_indices, -1, ix) # (B, 1)

    else:
        sampled = torch.multinomial(probs, 1, generator=rng)

    return sampled


def use_calculator(expr):
    """Safely evaluate a simple arithmetic expression the model wrote between python tokens.

    Args:
        expr: a string, e.g. "48/3" or "1,000*3".

    Returns:
        The result as a string, e.g. "16" or "3000", or None if the expression is refused or fails.

    Safety: only the characters "0123456789+-*/.() " are allowed (all 23,716 GSM8K training calls
    use only these), "**" is refused (9**9**9 would hang), and eval runs with no builtins.
    Anything else, such as letters, names or __import__, returns None before eval is reached.

    A float with no fractional part becomes an int, so "48/3" gives "16" rather than "16.0",
    matching the training data. This reproduces 97.1% of GSM8K's results exactly; the rest
    differ only in formatting, e.g. "20" where the data has "20.00".

    Examples:
        "2*12" -> "24"    "48/3" -> "16"    "5*2.5" -> "12.5"    "1,000*3" -> "3000"
        "5/0" -> None     "2**100" -> None  "x+1" -> None        "()" -> None (not a number)
    """
    expr = expr.replace(",", "")
    allowed = set("0123456789+-*/.() ")
    if not set(expr) <= allowed or "**" in expr:
        return None    

    try:
        result = eval(expr, {"__builtins__": {}}, {})

    except Exception:
        return None

    if not isinstance(result, (int, float)):
        return None 

    if isinstance(result, float) and result.is_integer():
        result = int(result)

    return str(result)


@torch.no_grad()
def generate(model, prompt_ids, num_samples=1, max_new_tokens=256, temperature=1.0, top_k=50, seed=42):
    """Sample num_samples replies to one prompt, yielding one new token per reply at each step.

    Used by chat_cli (1 sample, printed as it streams), rollout (16 samples per RL question)
    and evaluate_gsm8k.

    Args:
        model: the uncompiled model; the sequence grows by one token every step.
        prompt_ids: a list of ints, usually ending with <|assistant_start|>.
        num_samples: how many independent replies (rows) to generate together in one batch.
        max_new_tokens: the most tokens per reply; rows still running are cut off here.
        temperature, top_k: passed to sample_next_token.
        seed: for the sampling generator, so the same seed gives the same replies.

    Each step:
        1. Run the model on all rows, cropped to the last block_size (1024) tokens. There is no
           KV cache, so the whole sequence is recomputed each time.
        2. Sample one token per row, then choose each row's actual token:
           - row finished: <|assistant_end|> as padding                        -> mask 0
           - calculator result waiting: the next forced token, sample ignored -> mask 0
           - otherwise: the sampled token                                     -> mask 1
        3. Update each row's state from its sampled token:
           - <|assistant_end|> or BOS: the row is done.
           - <|python_start|>: start collecting a calculator expression.
           - <|python_end|>: decode the collected tokens, call use_calculator, and queue
             <|output_start|> result <|output_end|> to be forced in over the next steps
             (nothing is queued if it returns None).
           - any other token while collecting: add it to the expression.
        4. Yield, then stop if every row is done; otherwise append the tokens and continue.

    Yields:
        (tokens, masks): two lists of num_samples ints, this step's token for each row and whether
        the model chose it (1) or it was forced or padding (0). Only the new tokens are yielded,
        not the whole sequence; the caller collects them. RL trains only on mask-1 tokens.

    Example (one row, the model writes a calculator call):
        sampled  <|python_start|> 48 / 3 <|python_end|>       mask 1 1 1 1 1
        forced   <|output_start|> 16 <|output_end|>           mask 0 0 0
        sampled  " #### 16" <|assistant_end|>                 mask 1 ... 1, then the row is done
    """
    device = next(model.parameters()).device
    rng = torch.Generator(device=device).manual_seed(seed)
    x = torch.tensor([prompt_ids] * num_samples, dtype=torch.long, device=device)
    forced = [[] for _ in range(num_samples)]   # tokens waiting to be injected, per row
    python_buf = [None] * num_samples           # None = not inside a calculator call
    done = [False] * num_samples                # has this row finished?

    for _ in range(max_new_tokens):
        logits, _ = model(x[:, -model.config.block_size:])
        sampled = sample_next_token(logits[:, -1, :], rng, temperature, top_k).view(-1).tolist()

        tokens, masks = [], []
        for i in range(len(sampled)):
            if done[i]:
                tokens.append(ASSISTANT_END)
                masks.append(0)

            elif forced[i]:
                # A calculator result is waiting: inject its next token instead of the model's choice
                tokens.append(forced[i].pop(0))
                masks.append(0)

            else:
                # The model chose this token
                token = sampled[i]
                tokens.append(token)
                masks.append(1)

                if token == ASSISTANT_END or token == bos:
                    # The reply is finished
                    done[i] = True

                elif token == PY_START:
                    # A calculator call starts: begin collecting its tokens
                    python_buf[i] = []

                elif token == PY_END and python_buf[i] is not None:
                    # The call is complete: evaluate it and queue the result to be injected
                    result = use_calculator(enc.decode(python_buf[i]))
                    if result is not None:
                        forced[i] = [OUT_START] + enc.encode_ordinary(result) + [OUT_END]
                    python_buf[i] = None

                elif python_buf[i] is not None:
                    # Inside a call: collect the expression's tokens
                    python_buf[i].append(token)

        yield tokens, masks

        # Stop once every row has finished
        if all(done):
            break

        # Add this step's tokens as a new column, so the next forward pass sees them
        x = torch.cat([x, torch.tensor(tokens, device=device).view(-1, 1)], dim=1)


class KVCache:
    """Stores every layer's keys and values, so generate only computes the newest token each step.

    Without a cache, each step re-runs the whole sequence, even though the keys and values of
    earlier tokens never change. With one, the prompt is processed once ("prefill"), and each later
    step feeds in a single token, whose k and v are appended here.

    Args:
        num_layers: one slot per transformer block (12 for GPT-2 124M).
        max_len: the most tokens the cache can hold, prompt included. GPT-2's learned positions
            stop at block_size (1024), so it can never be more than that.

    Attributes:
        pos: how many tokens are stored. insert writes at pos but does not move it, because all
            layers insert at the same position; GPT.forward adds T after the last layer.
        k, v: lists of per-layer tensors (B, n_head, max_len, head_size), allocated on a layer's
            first insert, so they take that call's dtype (bf16 under autocast) and device.

    insert(layer_idx, k, v), with k and v of shape (B, n_head, T, head_size):
        1. Several tokens at once (T > 1) are only allowed into an empty cache, i.e. the prompt.
            Afterwards it's one token per step, which keeps the attention mask simple.
        2. Writes k and v into positions pos .. pos+T-1 of that layer.
        3. Returns that layer's k and v for positions 0 .. pos+T-1, everything the new tokens
            may attend to.

    Example (150-token prompt, 16 samples, max_len 406):
        prefill: insert(0, k (16,12,150,64), v) -> (16,12,150,64); GPT.forward then sets pos = 150
        step 1:  insert(0, k (16,12,1,64), v)   -> (16,12,151,64)

    Gotcha: once full, it can't slide like the old cropping did. Each cached k and v was computed
    with its token's position embedding built in, so shifting them would make every one wrong.
    """

    def __init__(self, num_layers, max_len):
        self.pos = 0
        self.k = [None] * num_layers
        self.v = [None] * num_layers
        self.max_len = max_len
        

    def insert(self, layer_idx, k, v):
        B, num_heads, T, head_size = k.shape
        assert T == 1 or self.pos == 0, "Only one token at a time is allowed to be inserted after the prompt"
        assert self.pos + T <= self.max_len, f"Context length has exceeded the allowed max length, ({self.pos} + {T} > {self.max_len})"

        if self.k[layer_idx] is None:
            self.k[layer_idx] = torch.zeros((B, num_heads, self.max_len, head_size), dtype=k.dtype, device=k.device)
            self.v[layer_idx] = torch.zeros((B, num_heads, self.max_len, head_size), dtype=v.dtype, device=v.device)

    
        self.k[layer_idx][:, :, self.pos:self.pos + T] = k
        self.v[layer_idx][:, :, self.pos:self.pos + T] = v

        return self.k[layer_idx][:, :, 0:self.pos + T], self.v[layer_idx][:, :, 0:self.pos + T]

    
if __name__ == "__main__":
    logits = torch.tensor([[2.0, 1.0, 0.5, 3.0, -1.0],
                           [0.0, 5.0, 0.0, 0.0, 4.0]])
    rng = torch.Generator().manual_seed(42)

    # Shape and dtype: one token per row
    out = sample_next_token(logits, rng)
    assert out.shape == (2, 1) and out.dtype == torch.long

    # Temperature 0 is greedy: always the largest logit
    assert sample_next_token(logits, rng, temperature=0.0).tolist() == [[3], [1]]

    # top_k=1 is also greedy, whatever the temperature
    assert sample_next_token(logits, rng, temperature=2.0, top_k=1).tolist() == [[3], [1]]

    # top_k=2 only ever picks from the two largest logits
    row0 = {sample_next_token(logits[:1], rng, top_k=2).item() for _ in range(500)}
    assert row0 == {0, 3}, row0

    # top_k larger than the vocabulary doesn't crash
    sample_next_token(logits, rng, top_k=1000)

    # Low temperature almost always picks the largest logit
    picks = [sample_next_token(logits[:1], rng, temperature=0.1).item() for _ in range(200)]
    assert picks.count(3) >= 195

    # Same seed gives the same samples
    a = sample_next_token(logits, torch.Generator().manual_seed(0), top_k=None)
    b = sample_next_token(logits, torch.Generator().manual_seed(0), top_k=None)
    assert torch.equal(a, b)

    
    # Real GSM8K calls: the result matches the python_output in the data
    assert use_calculator("2*12") == "24"
    assert use_calculator("2+4+3+10") == "19"
    assert use_calculator("48/3") == "16"           # not "16.0"
    assert use_calculator("5*2.5") == "12.5"
    assert use_calculator("(10+5)*2") == "30"
    assert use_calculator("1,000*3") == "3000"      # commas are removed
    assert use_calculator(" 6 - 2 ") == "4"

    # Bad maths returns None instead of crashing
    assert use_calculator("5/0") is None
    assert use_calculator("2+") is None
    assert use_calculator("") is None
    assert use_calculator("()") is None             # valid Python, but not a number

    # Anything that isn't arithmetic is refused
    assert use_calculator("2**100") is None
    assert use_calculator("__import__('os')") is None
    assert use_calculator("x+1") is None


    from types import SimpleNamespace

    class ScriptedModel(torch.nn.Module):
        def __init__(self, scripts, block_size=1024):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))  # so generate can find the device
            self.config = SimpleNamespace(block_size=block_size)
            self.scripts, self.calls = scripts, 0

        def forward(self, x):
            assert x.size(1) <= self.config.block_size, "generate must crop to block_size"
            logits = torch.full((x.size(0), x.size(1), 50304), -1e9)
            for i, script in enumerate(self.scripts):
                logits[i, -1, script[min(self.calls, len(script) - 1)]] = 0.0
            self.calls += 1
            return logits, None

    def run(scripts, prompt=[bos], **kwargs):
        steps = list(generate(ScriptedModel(scripts), prompt, num_samples=len(scripts), **kwargs))
        tokens = [[step[0][i] for step in steps] for i in range(len(scripts))]
        masks = [[step[1][i] for step in steps] for i in range(len(scripts))]
        return tokens, masks

    PY_START, PY_END = special_token_ids["<|python_start|>"], special_token_ids["<|python_end|>"]
    ASSISTANT_END = special_token_ids["<|assistant_end|>"]
    FILLER = 0  # the model's prediction during forced steps is ignored
    hello = enc.encode_ordinary("Hello!") + [ASSISTANT_END]

    # 1. Stops at <|assistant_end|>, and every sampled token has mask 1
    tokens, masks = run([hello + [FILLER] * 5])
    assert tokens[0] == hello and masks[0] == [1] * len(hello)

    # 2. Calculator: the result is forced in with mask 0, then sampling continues
    call = [PY_START] + enc.encode_ordinary("48/3") + [PY_END]
    after = enc.encode_ordinary(" #### 16") + [ASSISTANT_END]
    tokens, masks = run([call + [FILLER] * 3 + after])
    assert enc.decode(tokens[0]) == "<|python_start|>48/3<|python_end|><|output_start|>16<|output_end|> #### 16<|assistant_end|>", enc.decode(tokens[0])
    assert masks[0] == [1] * len(call) + [0] * 3 + [1] * len(after)

    # 3. An expression the calculator refuses: nothing is forced
    bad = [PY_START] + enc.encode_ordinary("x+1") + [PY_END, ASSISTANT_END]
    tokens, masks = run([bad])
    assert tokens[0] == bad and 0 not in masks[0]

    # 4. Never ends: stops after max_new_tokens
    tokens, _ = run([[FILLER]], max_new_tokens=7)
    assert len(tokens[0]) == 7

    # 5. Two samples: the finished one is padded with <|assistant_end|>, mask 0, until the other ends
    short = enc.encode_ordinary("Hi") + [ASSISTANT_END]
    tokens, masks = run([short, hello])
    n = len(hello) - len(short)
    assert tokens[0] == short + [ASSISTANT_END] * n and masks[0] == [1] * len(short) + [0] * n
    assert tokens[1] == hello and masks[1] == [1] * len(hello)

    # 6. A prompt longer than block_size is cropped instead of crashing
    tokens, _ = run([hello], prompt=[bos] * 2000)
    assert tokens[0] == hello

    # 7. BOS also stops generation
    tokens, _ = run([[FILLER, bos, FILLER]])
    assert tokens[0] == [FILLER, bos]

    
    # --- KVCache ---
    torch.manual_seed(0)
    cache = KVCache(num_layers=2, max_len=10)
    assert cache.pos == 0 and cache.k == [None, None]

    # The prompt: 5 tokens, 3 rows, 4 heads, head size 8
    k, v = torch.randn(3, 4, 5, 8), torch.randn(3, 4, 5, 8)
    k_all, v_all = cache.insert(0, k, v)
    assert k_all.shape == (3, 4, 5, 8) and torch.equal(k_all, k) and torch.equal(v_all, v)
    assert cache.k[0].shape == (3, 4, 10, 8)    # room for max_len tokens is allocated up front
    assert cache.k[1] is None                    # each layer allocates on its own first insert
    assert cache.pos == 0                        # insert doesn't move pos; GPT.forward does
    cache.insert(1, k, v)
    cache.pos += 5

    # One new token: returns the prompt's keys followed by the new one
    k1, v1 = torch.randn(3, 4, 1, 8), torch.randn(3, 4, 1, 8)
    k_all, v_all = cache.insert(0, k1, v1)
    assert k_all.shape == (3, 4, 6, 8)
    assert torch.equal(k_all[:, :, :5], k) and torch.equal(k_all[:, :, 5:], k1)
    assert torch.equal(v_all[:, :, :5], v) and torch.equal(v_all[:, :, 5:], v1)

    # Layers are kept separately: layer 1 still holds only what was inserted into it
    k_all1, _ = cache.insert(1, k1 * 0, v1 * 0)
    assert torch.equal(k_all1[:, :, :5], k) and torch.equal(k_all1[:, :, 5:], k1 * 0)
    cache.pos += 1

    # More than one token into a cache that isn't empty is refused
    try:
        cache.insert(0, torch.randn(3, 4, 2, 8), torch.randn(3, 4, 2, 8))
        raise RuntimeError("should have failed")
    except AssertionError:
        pass

    # A prompt longer than max_len is refused
    try:
        KVCache(1, 4).insert(0, torch.randn(1, 1, 5, 2), torch.randn(1, 1, 5, 2))
        raise RuntimeError("should have failed")
    except AssertionError:
        pass

    # Filling up one token at a time stops exactly at max_len
    small = KVCache(1, 3)
    for _ in range(3):
        small.insert(0, torch.randn(1, 1, 1, 2), torch.randn(1, 1, 1, 2))
        small.pos += 1
    try:
        small.insert(0, torch.randn(1, 1, 1, 2), torch.randn(1, 1, 1, 2))
        raise RuntimeError("should have failed")
    except AssertionError:
        pass

    # The cache takes the dtype of the first keys it sees (bf16 under autocast)
    bf = KVCache(1, 4)
    k_all, _ = bf.insert(0, torch.randn(1, 2, 3, 4, dtype=torch.bfloat16), torch.randn(1, 2, 3, 4, dtype=torch.bfloat16))
    assert bf.k[0].dtype == torch.bfloat16 and k_all.dtype == torch.bfloat16


    # --- CausalSelfAttention with a KV cache ---
    from gpt.model import GPTConfig, CausalSelfAttention

    torch.manual_seed(0)
    attn = CausalSelfAttention(GPTConfig(block_size=16, n_head=2, n_embd=8)).eval()
    x = torch.randn(3, 6, 8)    # 3 rows, 6 tokens
    full = attn(x)               # no cache: the whole sequence at once, as in training

    # Prefill: the first 4 tokens through an empty cache match the no-cache output
    cache = KVCache(num_layers=1, max_len=16)
    out = attn(x[:, :4], kv_cache=cache, layer_idx=0)
    cache.pos += 4
    assert out.shape == (3, 4, 8) and torch.allclose(out, full[:, :4], atol=1e-6)

    # Then one token at a time: each matches its row of the full output, so it saw every earlier token
    for t in range(4, 6):
        out = attn(x[:, t:t + 1], kv_cache=cache, layer_idx=0)
        cache.pos += 1
        assert out.shape == (3, 1, 8) and torch.allclose(out, full[:, t:t + 1], atol=1e-6), t

    # The cache now holds keys for all 6 positions, and nothing after them
    assert cache.k[0][:, :, :6].abs().sum() > 0 and cache.k[0][:, :, 6:].abs().sum() == 0


    print("all tests passed")
