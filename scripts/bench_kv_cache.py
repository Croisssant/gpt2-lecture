import os
import time
import argparse
import subprocess
import tempfile
import importlib.util
import torch

from gpt.utils import load_checkpoint, append_to_log
from gpt.engine import generate
from gpt.chat_tokenizer import render_for_completion
from tasks.gsm8k import GSM8K

NO_CACHE_COMMIT = "262586f" # the last commit before the KV cache was added to gpt/engine.py

def load_no_cache_generate(commit=NO_CACHE_COMMIT):
    """Load generate() as it was before the KV cache, straight from git history.

    The old version isn't kept in the repo. Instead, `git show <commit>:gpt/engine.py` writes that
    commit's engine.py to a temporary file, which is imported as its own module next to the current
    gpt.engine. Its generate calls model(x) without a kv_cache, which the current GPT.forward still
    supports (kv_cache=None), so it runs the same weights the old way: the whole sequence every step.

    Returns:
        The old generate function, with the same arguments and yields as the current one.
    """
    source = subprocess.run(["git", "show", f"{commit}:gpt/engine.py"], capture_output=True, text=True, check=True).stdout
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(source)
    spec = importlib.util.spec_from_file_location("engine_no_cache", f.name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    os.remove(f.name)
    return module.generate


def time_generate(generate_fn, model, prompt_ids, num_samples, max_new_tokens, device):
    """Run one generate call to the end and time it.

    GPU work is asynchronous, so the device is synchronized before starting and before stopping
    the clock; otherwise the time would only cover queueing the work, not doing it.

    Returns:
        (steps, seconds): the list of (tokens, masks) it yielded, and the wall-clock time.
    """
    sync = {"cuda": torch.cuda.synchronize, "mps": torch.mps.synchronize}.get(device, lambda: None)
    sync()
    start = time.time()
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")):
        steps = list(generate_fn(model, prompt_ids, num_samples=num_samples, max_new_tokens=max_new_tokens, temperature=0.0))
    sync()
    return steps, time.time() - start


def benchmark(model, prompts, sample_counts, max_new_tokens, device, log_file):
    """Time generate with and without the KV cache on the same prompts, and log one line per setting.

    Used to record how much the KV cache speeds up generation, e.g. on the Mac and on the H200.

    Args:
        model: a chat model in eval mode.
        prompts: lists of token ids, e.g. GSM8K test questions from render_for_completion.
        sample_counts: batch sizes to try, e.g. [1, 16] (chat_cli and rollout).
        max_new_tokens: the most tokens per reply, the same for both versions.
        device: "cuda", "mps" or "cpu". On cuda it runs under bf16 autocast, like RL and eval.
        log_file: lines are appended here, so runs on different machines collect in one file.

    What it does, for each batch size:
        1. One warm-up call of each version (the first call pays for kernel setup).
        2. For every prompt, time the old generate (no cache) and the current one (cache).
           Greedy sampling (temperature 0), so both should produce the same tokens.
        3. Add up the time, the tokens generated (steps x samples) and how many steps matched.
        4. Log: tokens, both times and tokens/s, the speed-up, and identical steps.

    Identical steps should be all of them on fp32 (Mac). Under bf16 a few may differ: the two
    versions add up the same numbers in a different order, which can flip a near-tie in argmax,
    and the replies then diverge from there.

    Returns:
        {num_samples: speed-up} for each batch size.
    """
    no_cache_generate = load_no_cache_generate()
    speedups = {}

    for num_samples in sample_counts:
        for fn in [no_cache_generate, generate]:
            time_generate(fn, model, prompts[0], num_samples, 16, device)

        total_tokens, same, total_steps = 0, 0, 0
        seconds = {"no_cache": 0.0, "cache": 0.0}
        for prompt_ids in prompts:
            old_steps, old_time = time_generate(no_cache_generate, model, prompt_ids, num_samples, max_new_tokens, device)
            new_steps, new_time = time_generate(generate, model, prompt_ids, num_samples, max_new_tokens, device)
            seconds["no_cache"] += old_time
            seconds["cache"] += new_time
            total_tokens += len(new_steps) * num_samples
            same += sum(a == b for a, b in zip(old_steps, new_steps))
            total_steps += max(len(old_steps), len(new_steps))

        speedups[num_samples] = seconds["no_cache"] / seconds["cache"]
        line = (f"{device} samples {num_samples} prompts {len(prompts)} tokens {total_tokens} | "
                f"no_cache {seconds['no_cache']:.2f}s {total_tokens / seconds['no_cache']:.0f} tok/s | "
                f"cache {seconds['cache']:.2f}s {total_tokens / seconds['cache']:.0f} tok/s | "
                f"speedup {speedups[num_samples]:.1f}x | identical steps {same}/{total_steps}\n")
        print(line, end="")
        append_to_log(log_file, line)

    return speedups


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="run_logs/rl_lr5e-6/model_rl.pt", help="chat model path")
    parser.add_argument("--log-dir", default="run_logs/kv_cache")
    parser.add_argument("--num-prompts", type=int, default=5, help="GSM8K test questions to time")
    parser.add_argument("--samples", default="1,16", help="comma-separated batch sizes")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    model, _ = load_checkpoint(args.checkpoint, device)
    model.eval()
    task = GSM8K("test")
    prompts = [render_for_completion(task[i]) for i in range(args.num_prompts)]

    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, "log.txt")
    header = (f"# {time.strftime('%Y-%m-%d %H:%M')} {device} {args.checkpoint}, {args.num_prompts} GSM8K prompts, "
              f"greedy, max_new_tokens {args.max_new_tokens}, no_cache = generate from {NO_CACHE_COMMIT}\n")
    print(header, end="")
    append_to_log(log_file, header)
    benchmark(model, prompts, [int(n) for n in args.samples.split(",")], args.max_new_tokens, device, log_file)
