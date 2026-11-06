
import torch

from torch.nn import functional as F
from gpt.chat_tokenizer import enc, bos, special_token_ids

PY_START = special_token_ids["<|python_start|>"]
PY_END = special_token_ids["<|python_end|>"]
OUT_START = special_token_ids["<|output_start|>"]
OUT_END = special_token_ids["<|output_end|>"]
ASSISTANT_END = special_token_ids["<|assistant_end|>"]

def sample_next_token(logits, rng, temperature=1.0, top_k=50):   
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


    print("all tests passed")
