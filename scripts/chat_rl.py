import torch
import os
import time

import torch.nn.functional as F    # only the tests use it

from gpt.engine import generate
from gpt.chat_dataloader import build_xy
from gpt.chat_tokenizer import enc, render_for_completion
from gpt.utils import load_checkpoint, save_checkpoint, append_to_log
from scripts.chat_eval import evaluate_gsm8k
from tasks.gsm8k import GSM8K


def rollout(model, task, index, num_samples=16, max_new_tokens=256, temperature=1.0, top_k=50, seed=0):
    """Answer one question num_samples times, score each answer, and build training tensors.

    This is the "experience" half of RL: the model tries, and the reward says how well each try
    went. Each sample is judged against the *group* of samples for the same question (GRPO-style),
    so no separate value model is needed.

    Args:
        model: the uncompiled model (generate's shapes change every step).
        task: has [index] and reward(conversation, completion), e.g. GSM8K("train").
        index: which question.
        num_samples: samples per question, the group size (16, as in nanochat).
        max_new_tokens, temperature, top_k, seed: passed to generate. Temperature 1.0 samples from
            the model's own distribution, so the group explores different answers.

    What it does:
        1. Build the prompt with render_for_completion and generate num_samples replies.
        2. For each sample, decode its tokens and get its reward from task.reward.
        3. Make each row prompt + reply, with mask 0 for the prompt and generate's masks for the
           reply, and turn the rows into x and y with build_xy. All rows are the same length,
           because generate pads finished rows.
        4. advantages = rewards - rewards.mean().

    Returns:
        x, y: (B, L-1) long tensors on the model's device. y is -1 for the prompt, forced
            calculator output and padding, so only tokens the model chose get credit or blame.
        rewards: (B,) float tensor, e.g. 1.0 = correct.
        advantages: (B,) float tensor that sums to 0. Positive = better than this group's average.

    Example: rewards [1, 0, 0, 1] -> mean 0.5 -> advantages [+0.5, -0.5, -0.5, +0.5].
    If every sample gets the same reward (all wrong, or all right), every advantage is 0 and the
    question teaches nothing. The SFT model scores 0 on GSM8K, so every advantage is 0 there.
    """
    device = next(model.parameters()).device
    conversation = task[index]
    prompt_ids = render_for_completion(conversation)

    steps = list(generate(model, prompt_ids, num_samples=num_samples, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, seed=seed))
    rows = []
    rewards = []
    for i in range(num_samples):
        gen_tokens = [tokens[i] for tokens, _ in steps]
        gen_masks = [masks[i] for _, masks in steps]
        reward = task.reward(conversation, enc.decode(gen_tokens))
        rewards.append(reward)
        rows.append((prompt_ids + gen_tokens, [0] * len(prompt_ids) + gen_masks))

    x, y = build_xy(rows)
    x = x.to(device)
    y = y.to(device)
    rewards = torch.tensor(rewards, dtype=torch.float, device=device)
    advantages = rewards - rewards.mean()

    return x, y, rewards, advantages


def rl_loss(model, x, y, advantages):
    """The policy-gradient (REINFORCE) loss for one rollout.

    Minimizing it raises the probability of every token in samples with a positive advantage and
    lowers it in samples with a negative advantage, in proportion to the advantage. This is
    nanochat's simplified GRPO: no KL penalty, no reference model and no PPO clipping, because
    each batch of samples is used for exactly one update (fully on-policy).

    Args:
        model: the model being trained.
        x, y: (B, T) from rollout, with y = -1 where tokens weren't chosen by the model.
        advantages: (B,) from rollout.

    What it does:
        1. nll = model(x, targets=y, loss_reduction="none"): (B, T) of -log p(chosen token),
           already 0 wherever y == -1 (ignore_index).
        2. logp = -nll.
        3. loss = -sum(logp * advantage of that row) / number of valid tokens (y != -1).
           Each row's advantage multiplies all of its tokens. The minus sign is because the
           optimizer minimizes, while we want to maximize logp * advantage.

    Returns:
        A scalar tensor to call .backward() on.

    The value itself isn't meaningful to track: it can be negative, and it's exactly 0 when every
    advantage is 0. Track the mean reward instead.

    Example:
        logp row 0 [-1.0, -2.0] (advantage +0.5), row 1 [-0.5, ignored] (advantage -0.5)
        sum = 0.5*(-3.0) + (-0.5)*(-0.5) = -1.25; 3 valid tokens -> loss = 1.25 / 3 = 0.417
    """

    # nll: negative log likelihood
    _, nll = model(x, targets=y, loss_reduction="none")
    logp = -nll
    num_valid = (y != -1).sum().clamp(min=1)
    loss = -(logp * advantages.unsqueeze(1)).sum() / num_valid

    return loss


def get_rl_lr(step, num_steps, max_lr=3e-5):
    """RL learning rate: decreases in a straight line from max_lr to 0, with no warmup.

    Args:
        step: from 0 to num_steps - 1.
        num_steps: the length of the run.
        max_lr: 3e-5, i.e. 0.05 x the pretraining peak of 6e-4, the same fraction nanochat uses.
            RL should only nudge a model that can already write well, so this is about 16x
            smaller than SFT's.

    Returns:
        max_lr * (1 - step / num_steps), e.g. with num_steps=100: step 0 -> 3e-5,
        step 50 -> 1.5e-5, step 99 -> 3e-7.
    """
    return max_lr * (1 - step / num_steps)


def train_rl(checkpoint_path, train_task, eval_task, log_dir, device, questions_per_step=16, 
             num_samples=16, max_new_tokens=256, temperature=1.0, top_k=50, max_lr=3e-5, 
             num_steps=None, eval_every=60, eval_examples=400, eval_samples=8):
    """RL on a task with a checkable answer (GSM8K): sample answers, reward correct ones, update.

    nanochat's simplified GRPO. Each step, for questions_per_step new questions, rollout samples
    num_samples answers and scores them; rl_loss then raises the probability of answers that beat
    their group's average and lowers the rest. One pass over the data, single GPU.

    Args:
        checkpoint_path: the SFT checkpoint to start from (it has no optimizer state).
        train_task: has len(), [i] and reward(), e.g. GSM8K("train").
        eval_task: has len(), [i] and evaluate(), e.g. GSM8K("test").
        log_dir: a new folder; must not already contain log.txt.
        device: "cuda", "mps" or "cpu".
        questions_per_step: questions per optimizer step (16, as in nanochat).
        num_samples: answers per question, the group size (16).
        max_new_tokens, temperature, top_k: sampling settings for training rollouts.
        max_lr: peak learning rate, decaying linearly to 0 (get_rl_lr).
        num_steps: None = one epoch, len(train_task) // questions_per_step (467 for GSM8K).
        eval_every, eval_examples, eval_samples: pass@k on eval_task every eval_every steps.

    What it does:
        1. Load the model and create a fresh AdamW with weight decay 0.
        2. Each step: every eval_every steps, evaluate and save log_dir/model_rl.pt; set the lr; for
           each question, run rollout, skip it if every advantage is 0 (no signal), otherwise backward
           rl_loss / questions_per_step; clip the gradient norm to 1.0; step the optimizer; print and log.
        3. Final eval, then save log_dir/model_rl.pt.

    model_rl.pt is overwritten at every eval, so if the run crashes (e.g. out of memory on a shared
    GPU), the weights from the last eval survive; its "step" says how many steps they include.
    It holds no optimizer state, so a restart from it begins with a fresh AdamW.

    Log lines (log_dir/log.txt):
        "<step> reward <mean>"   mean reward over the step's samples: should rise
        "<step> signal <frac>"   fraction of questions whose samples didn't all get the same reward;
                                 below ~0.1, the step learns almost nothing
        "<step> pass@1 <acc>" and "<step> pass@<eval_samples> <acc>" at each eval

    Returns:
        (model, results), where results is the final {k: pass@k} dict.
    """
    # Set up logging
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "log.txt")
    assert not os.path.exists(log_file), f"{log_file} already exists, choose a new log_dir"

    # Loading model
    device_type = "cuda" if "cuda" in device else device
    core_model, checkpoint = load_checkpoint(checkpoint_path, device)
    optimizer = core_model.configure_optimizers(weight_decay=0.0, learning_rate=max_lr, device=device)
    if num_steps is None: num_steps = len(train_task) // questions_per_step

    def log(step, name, value):
        line = f"{step} {name} {value:.4f}\n"
        print(line, end="")
        append_to_log(log_file, line)

    def run_eval(step):
        # pass@1 and pass@eval_samples on the eval task, logged
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            k_passes = evaluate_gsm8k(core_model, eval_task, num_samples=eval_samples, max_examples=eval_examples,
                                      max_new_tokens=max_new_tokens, top_k=top_k)
        for k in (1, eval_samples):
            log(step, f"pass@{k}", k_passes[k])
        return k_passes

    def save(step, k_passes):
        # Overwrites the previous save; "step" = how many training steps these weights include
        save_checkpoint({"model": core_model.state_dict(), "config": core_model.config, "step": step, "results": k_passes},
                        os.path.join(log_dir, "model_rl.pt"))

    # Loop steps
    for step in range(num_steps):
        if step % eval_every == 0:
            k_passes = run_eval(step)
            if step > 0:
                save(step, k_passes)   # so a crash later in the run keeps the progress up to here

        lr = get_rl_lr(step, num_steps, max_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        t0 = time.time()
        optimizer.zero_grad()
        rewards_sum, num_with_signal = 0.0, 0

        for q in range(questions_per_step):
            index = step * questions_per_step + q   # every question is used once
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                x, y, rewards, advantages = rollout(core_model, train_task, index, num_samples=num_samples,
                                                    max_new_tokens=max_new_tokens, temperature=temperature,
                                                    top_k=top_k, seed=index)
            rewards_sum += rewards.mean().item()

            # All samples got the same reward: zero gradient, so skip the forward and backward
            if (advantages == 0).all():
                continue

            num_with_signal += 1
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                loss = rl_loss(core_model, x, y, advantages) / questions_per_step
            loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(core_model.parameters(), 1.0)
        optimizer.step()

        dt = time.time() - t0
        mean_reward = rewards_sum / questions_per_step
        signal = num_with_signal / questions_per_step
        print(f"Step {step:4d} | reward: {mean_reward:.4f} | signal: {signal:.2f} | lr: {lr:.4e} | norm: {norm.item():.4f} | dt: {dt:.1f}s")
        log(step, "reward", mean_reward)
        log(step, "signal", signal)

    # Finish
    k_passes = run_eval(num_steps)
    save(num_steps, k_passes)

    return core_model, k_passes


def run_tests():
    """Test every RL function on tiny models and fake tasks, on the CPU. Run with --test."""
    from gpt.model import GPT, GPTConfig

    torch.manual_seed(0)
    tiny = GPT(GPTConfig(block_size=128, vocab_size=50304, n_layer=1, n_head=1, n_embd=16)).eval()

    class FakeTask:
        """One fixed question; the reward is 1.0 when the completion has an even number of characters"""
        def __init__(self):
            self.conversation = {"messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ]}
        def __getitem__(self, index):
            return self.conversation
        def reward(self, conversation, completion):
            return float(len(completion) % 2 == 0)

    task = FakeTask()
    prompt = render_for_completion(task[0])
    x, y, rewards, advantages = rollout(tiny, task, 0, num_samples=8, max_new_tokens=10, seed=1)

    # Shapes: B rows; prompt + 10 new tokens, minus 1 for the shift
    assert x.shape == y.shape == (8, len(prompt) + 10 - 1), x.shape
    assert rewards.shape == advantages.shape == (8,)

    # Every row starts with the same prompt, and the prompt is never trained on
    assert (x[:, :len(prompt)] == torch.tensor(prompt)).all()
    assert (y[:, :len(prompt) - 1] == -1).all()

    # The generated tokens are trained on (a random tiny model never stops early, so no padding)
    assert (y[:, len(prompt) - 1:] != -1).all()

    # Advantages are rewards minus the group mean, so they sum to 0
    assert torch.allclose(advantages, rewards - rewards.mean())
    assert abs(advantages.sum().item()) < 1e-5
    assert 0 < rewards.sum() < 8, f"expected mixed rewards, got {rewards.tolist()}"

    # The same seed gives the same rollout
    x2, _, rewards2, _ = rollout(tiny, task, 0, num_samples=8, max_new_tokens=10, seed=1)
    assert torch.equal(x, x2) and torch.equal(rewards, rewards2)

    # All rewards equal → all advantages 0 (nothing to learn from this question)
    task.reward = lambda conversation, completion: 1.0
    _, _, rewards, advantages = rollout(tiny, task, 0, num_samples=4, max_new_tokens=5)
    assert rewards.tolist() == [1.0] * 4 and advantages.tolist() == [0.0] * 4


    # --- rl_loss ---
    torch.manual_seed(0)
    tiny = GPT(GPTConfig(block_size=128, vocab_size=50304, n_layer=1, n_head=1, n_embd=16))

    x = torch.randint(0, 50257, (2, 6))
    y = torch.randint(0, 50257, (2, 6))
    y[:, :3] = -1                          # the prompt
    y[1, 5] = -1                           # padding at the end of row 1
    advantages = torch.tensor([0.5, -0.5])
    valid = y != -1

    # The loss is a single number
    loss = rl_loss(tiny, x, y, advantages)
    assert loss.dim() == 0

    # It matches -sum(logp * advantage) / number of valid tokens, computed by hand
    logits, _ = tiny(x)
    logp = F.log_softmax(logits, dim=-1).gather(-1, y.clamp(min=0).unsqueeze(-1)).squeeze(-1) * valid
    expected = -(logp * advantages.unsqueeze(1)).sum() / valid.sum()
    assert torch.allclose(loss, expected, atol=1e-5), (loss.item(), expected.item())

    # Zero advantages: zero loss and zero gradients (nothing to learn)
    tiny.zero_grad()
    zero = rl_loss(tiny, x, y, torch.zeros(2))
    zero.backward()
    assert zero.item() == 0.0
    assert all(p.grad is None or p.grad.abs().max() == 0 for p in tiny.parameters())

    # One gradient step makes row 0 (advantage +) more likely and row 1 (advantage −) less likely
    def row_logp():
        with torch.no_grad():
            logits, _ = tiny(x)
            lp = F.log_softmax(logits, dim=-1).gather(-1, y.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            return (lp * valid).sum(dim=1)
    before = row_logp()
    optimizer = torch.optim.SGD(tiny.parameters(), lr=0.1)
    optimizer.zero_grad()
    rl_loss(tiny, x, y, advantages).backward()
    optimizer.step()
    after = row_logp()
    assert after[0] > before[0] and after[1] < before[1], (before.tolist(), after.tolist())

    # --- get_rl_lr ---
    assert get_rl_lr(0, 100, 3e-5) == 3e-5                        # starts at max_lr
    assert abs(get_rl_lr(50, 100, 3e-5) - 1.5e-5) < 1e-12         # halfway: half
    assert 0 < get_rl_lr(99, 100, 3e-5) < 1e-6                    # last step: tiny but positive
    lrs = [get_rl_lr(s, 100, 3e-5) for s in range(100)]
    assert all(a > b for a, b in zip(lrs, lrs[1:]))               # always decreasing


    # --- train_rl ---
    import os
    import tempfile
    from gpt.utils import load_checkpoint

    class FakeGSM8K:
        """n identical questions; `rule` decides whether a completion is correct"""
        def __init__(self, rule, n=8):
            self.rule, self.n = rule, n
        def __len__(self):
            return self.n
        def __getitem__(self, index):
            return {"messages": [{"role": "user", "content": "What is 2+2?"},
                                 {"role": "assistant", "content": "4"}]}
        def evaluate(self, conversation, completion):
            return int(self.rule(completion))
        def reward(self, conversation, completion):
            return float(self.rule(completion))

    torch.manual_seed(0)
    base = GPT(GPTConfig(block_size=128, vocab_size=50304, n_layer=1, n_head=1, n_embd=16))
    folder = tempfile.mkdtemp()
    base_path = os.path.join(folder, "model_sft.pt")
    torch.save({"model": base.state_dict(), "config": base.config, "step": 0}, base_path)   # no optimizer, like model_sft.pt
    settings = dict(questions_per_step=2, num_samples=4, max_new_tokens=6, max_lr=1e-2, eval_every=2, eval_examples=2, eval_samples=2)

    # Mixed rewards: runs, logs, learns and saves
    parity = lambda c: len(c) % 2 == 0
    log_dir = os.path.join(folder, "log_rl")
    model, results = train_rl(base_path, FakeGSM8K(parity), FakeGSM8K(parity), log_dir, "cpu", **settings)
    assert set(results) == {1, 2}
    lines = open(os.path.join(log_dir, "log.txt")).read().splitlines()
    assert sum(" reward " in line for line in lines) == 4, "8 questions / 2 per step = 4 steps"
    assert lines[0].startswith("0 pass@1") and lines[-1].startswith("4 pass@2"), (lines[0], lines[-1])
    assert any(float(line.split()[-1]) > 0 for line in lines if " signal " in line), "some groups should have mixed rewards"
    assert not torch.equal(model.transformer.wte.weight, base.transformer.wte.weight), "weights should change"

    # The saved model reloads with the same weights
    reloaded, saved = load_checkpoint(os.path.join(log_dir, "model_rl.pt"), "cpu")
    x = torch.randint(0, 50304, (1, 8))
    assert torch.equal(reloaded.eval()(x)[0], model.eval()(x)[0]) and saved["step"] == 4

    # A crash mid-run keeps the checkpoint from the last eval (here: saved at step 2, crash during step 3)
    class CrashingGSM8K(FakeGSM8K):
        def __getitem__(self, index):
            if index >= 6:
                raise RuntimeError("simulated crash")
            return super().__getitem__(index)
    log_dir = os.path.join(folder, "log_rl_crash")
    try:
        train_rl(base_path, CrashingGSM8K(parity), FakeGSM8K(parity), log_dir, "cpu", **settings)
        raise AssertionError("expected the simulated crash")
    except RuntimeError:
        pass
    _, saved = load_checkpoint(os.path.join(log_dir, "model_rl.pt"), "cpu")
    assert saved["step"] == 2 and set(saved["results"]) == {1, 2}

    # Refuses to overwrite an existing log
    try:
        train_rl(base_path, FakeGSM8K(parity), FakeGSM8K(parity), log_dir, "cpu", **settings)
        raise RuntimeError("expected AssertionError")
    except AssertionError:
        pass

    # Every reward the same: no signal, so the weights must not change at all
    log_dir = os.path.join(folder, "log_rl_flat")
    model, _ = train_rl(base_path, FakeGSM8K(lambda c: False), FakeGSM8K(lambda c: False), log_dir, "cpu", **settings)
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), base.state_dict().values()))
    lines = open(os.path.join(log_dir, "log.txt")).read().splitlines()
    assert all(float(line.split()[-1]) == 0 for line in lines if " signal " in line or " reward " in line)

    # num_steps limits the run
    log_dir = os.path.join(folder, "log_rl_short")
    train_rl(base_path, FakeGSM8K(parity), FakeGSM8K(parity), log_dir, "cpu", **{**settings, "num_steps": 1})
    assert sum(" reward " in line for line in open(os.path.join(log_dir, "log.txt"))) == 1

    print("all tests passed")


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="run the tests and stop")
    parser.add_argument("--checkpoint", help="SFT model path to start from")
    parser.add_argument("--log-dir", help="a new folder for this run")
    parser.add_argument("--max-lr", type=float, default=3e-5)
    parser.add_argument("--num-steps", type=int, default=None, help="default: one epoch of GSM8K train")
    parser.add_argument("--questions-per-step", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=16, help="answers per question")
    parser.add_argument("--eval-every", type=int, default=60)
    parser.add_argument("--eval-examples", type=int, default=400, help="GSM8K test questions per eval")
    args = parser.parse_args()

    if args.test:
        run_tests()
        sys.exit(0)

    if args.checkpoint is None or args.log_dir is None:
        parser.error("--checkpoint and --log-dir are required unless --test is set")

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

    torch.set_float32_matmul_precision("high")
    train_rl(args.checkpoint, GSM8K("train"), GSM8K("test"), args.log_dir, device,
             questions_per_step=args.questions_per_step, num_samples=args.num_samples,
             max_lr=args.max_lr, num_steps=args.num_steps,
             eval_every=args.eval_every, eval_examples=args.eval_examples)
