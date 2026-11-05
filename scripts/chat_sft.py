import os
import torch
import time

from gpt.model import GPT, GPTConfig
from gpt.chat_dataloader import chat_batches
from gpt.utils import load_checkpoint, save_checkpoint, append_to_log
from scripts.chat_eval import evaluate_categorical

from tasks.common import TaskMixture
from tasks.smoltalk import SmolTalk
from tasks.mmlu import MMLU
from tasks.arc import ARC
from tasks.gsm8k import GSM8K

def get_sft_lr(progress, init_lr, warmdown_ratio=0.5):
    return init_lr * (1 - progress) / warmdown_ratio if progress > 1 - warmdown_ratio else init_lr


@torch.no_grad()
def evaluate_val_loss(model, task, B, T, device, num_batches=20):
    model.eval()
    device_type = "cuda" if "cuda" in device else device
    total_loss = 0.0
    count = 0

    for x, y in chat_batches(task, B, T):
        x = x.to(device)
        y = y.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            _, loss = model(x, y)

        total_loss += loss.item()
        count += 1

        if count == num_batches:
            break

    model.train()

    return total_loss / count


def sft_step(model, optimizer, batches, grad_accum_steps, lr, device):
    device_type = "cuda" if "cuda" in device else device

    for group in optimizer.param_groups:
        group["lr"] = lr

    optimizer.zero_grad()
    loss_accum = 0.0

    for _ in range(grad_accum_steps):
        x, y = next(batches)
        x = x.to(device)
        y = y.to(device)

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            _, loss = model(x, y)
            loss /= grad_accum_steps
        loss_accum += loss.item()
        loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss_accum, norm.item()


def run_evals(model, core_model, val_task, eval_tasks, B, T, device, max_examples=None):
    results = {"val_loss": evaluate_val_loss(model, val_task, B, T, device)}
    centered = []

    for name, task in eval_tasks.items():
        accuracy = evaluate_categorical(core_model, task, device, max_examples=max_examples)
        results[name] = accuracy
        centered.append((accuracy - 0.25) / 0.75)

    results["chatcore"] = sum(centered) / len(centered)

    return results


def train_sft(checkpoint_path, train_task, val_task, eval_tasks, log_dir, device,
              B=32, T=1024, total_batch_size=524288, max_lr=6e-4, init_lr_frac=0.8,
              eval_every=100, max_examples=None, use_compile=False):

    # Set up logging
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "log.txt")
    assert not os.path.exists(log_file), f"{log_file} already exists, choose a new log_dir"

    # Load the model and optimizer
    init_lr = max_lr * init_lr_frac
    core_model, checkpoint = load_checkpoint(checkpoint_path, device)
    optimizer = core_model.configure_optimizers(weight_decay=0.0, learning_rate=init_lr, device=device)
    optimizer.load_state_dict(checkpoint["optimizer"])

    for group in optimizer.param_groups:
        group["weight_decay"]= 0.0

    del checkpoint
    model = torch.compile(core_model) if use_compile else core_model

    # Work out the length
    grad_accum_steps = total_batch_size // (B * T)
    num_steps = sum(1 for _ in chat_batches(train_task, B, T)) // grad_accum_steps

    # Train
    batches = chat_batches(train_task, B, T)
    for step in range(num_steps):
        if step % eval_every == 0:
            results = run_evals(model, core_model, val_task, eval_tasks, B, T, device, max_examples)
            
            for name, value in results.items():
                to_print = f"{step} {name} {value:.4f}\n"
                print(to_print, end="")
                append_to_log(log_file, to_print)

        lr = get_sft_lr(step / num_steps, init_lr)
        t0 = time.time()
        loss, norm = sft_step(model, optimizer, batches, grad_accum_steps, lr, device)

        t1 = time.time()
        dt = (t1 - t0) # Time difference in seconds
        tokens_processed = B * T * grad_accum_steps
        tokens_per_sec = tokens_processed / dt
        print(f"Step {step:4d} | loss: {loss:.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        append_to_log(log_file, f"{step} train {loss:.4f}\n")

    # Finish
    results = run_evals(model, core_model, val_task, eval_tasks, B, T, device, max_examples=None)
    for name, value in results.items():
        to_print = f"{num_steps} {name} {value:.4f}\n"
        print(to_print, end="")
        append_to_log(log_file, to_print)

    save_checkpoint({"model": core_model.state_dict(), "config": core_model.config, "step": num_steps, "results": results}, os.path.join(log_dir, "model_sft.pt"))

    return core_model, optimizer, results


def build_tasks():
    smol = SmolTalk("train")
    mmlu = MMLU("auxiliary_train")
    gsm = GSM8K("train")

    train_task = TaskMixture([smol] + [mmlu] * 3 + [gsm] * 4)
    val_task = TaskMixture([SmolTalk("test"), MMLU("test"), GSM8K("test")])
    eval_tasks = {"MMLU": MMLU("test"), "ARC-Easy": ARC("ARC-Easy", "test"), "ARC-Challenge": ARC("ARC-Challenge", "test")}

    return train_task, val_task, eval_tasks


def run_tests():
    import math
    import copy
    import itertools
    import tempfile
    
    # Constant for the first half
    assert get_sft_lr(0.0, 1.0) == 1.0
    assert get_sft_lr(0.49, 1.0) == 1.0

    # Decreases in a straight line, with no jump where it starts
    assert get_sft_lr(0.5, 1.0) == 1.0
    assert math.isclose(get_sft_lr(0.75, 1.0), 0.5)
    assert get_sft_lr(1.0, 1.0) == 0.0

    # A different warmdown_ratio: the last 20% decreases
    assert math.isclose(get_sft_lr(0.9, 1.0, warmdown_ratio=0.2), 0.5)

    # evaluate_val_loss
    conv_of = lambda u, a: {"messages": [{"role": "user", "content": u}, {"role": "assistant", "content": a}]}
    task = [conv_of(f"Question {i}", f"Answer {i}.") for i in range(20)]

    class FakeModel:
        def __init__(self):
            self.losses = iter([1.0, 2.0, 3.0])
            self.calls = 0
            self.modes = []
        def eval(self): self.modes.append("eval")
        def train(self): self.modes.append("train")
        def __call__(self, x, y):
            self.calls += 1
            return None, torch.tensor(next(self.losses))

    # Stops after num_batches, and switches eval mode on and back off
    fake = FakeModel()
    assert evaluate_val_loss(fake, task, 2, 31, "cpu", num_batches=2) == 1.5
    assert fake.calls == 2 and fake.modes == ["eval", "train"]

    # Handles a task with fewer batches than num_batches
    fake = FakeModel()
    assert evaluate_val_loss(fake, task, 2, 31, "cpu", num_batches=100) == 2.0
    assert fake.calls == 3

    # A real, untrained GPT should score close to ln(vocab_size), about 10.83
    torch.manual_seed(0)
    tiny = GPT(GPTConfig(block_size=64, vocab_size=50304, n_layer=2, n_head=2, n_embd=32))
    loss = evaluate_val_loss(tiny, task, 2, 31, "cpu")
    assert abs(loss - math.log(50304)) < 0.5, loss
    assert tiny.training


    # sft_step
    fixed = next(chat_batches(task, 2, 31))
    torch.manual_seed(0)
    tiny = GPT(GPTConfig(block_size=64, vocab_size=50304, n_layer=2, n_head=2, n_embd=32))
    optimizer = tiny.configure_optimizers(weight_decay=0.0, learning_rate=1e-3, device="cpu")

    # Uses exactly grad_accum_steps batches, sets the lr and updates the weights
    batches = iter([fixed, fixed, fixed, fixed])
    before = tiny.transformer.wte.weight.clone()
    loss, norm = sft_step(tiny, optimizer, batches, 2, 1e-3, "cpu")
    assert abs(loss - 10.83) < 0.5 and norm > 0
    assert all(group["lr"] == 1e-3 for group in optimizer.param_groups)
    assert len(list(batches)) == 2, "should use exactly 2 of the 4 batches"
    assert not torch.equal(before, tiny.transformer.wte.weight)

    # An lr of 0 leaves the weights unchanged
    frozen = {k: v.clone() for k, v in tiny.state_dict().items()}
    sft_step(tiny, optimizer, itertools.repeat(fixed), 1, 0.0, "cpu")
    assert all(torch.equal(frozen[k], v) for k, v in tiny.state_dict().items())

    # Repeating one batch should memorize it, so the loss falls a lot
    first = None
    for _ in range(60):
        loss, norm = sft_step(tiny, optimizer, itertools.repeat(fixed), 2, 3e-3, "cpu")
        first = first or loss
    assert loss < first - 1, (first, loss)

    # Accumulating the same batch twice must give the same gradient norm as using it once
    torch.manual_seed(0)
    base = GPT(GPTConfig(block_size=64, vocab_size=50304, n_layer=2, n_head=2, n_embd=32))
    norms = []
    for steps in [1, 2]:
        model = copy.deepcopy(base)
        optimizer = model.configure_optimizers(weight_decay=0.0, learning_rate=0.0, device="cpu")
        _, norm = sft_step(model, optimizer, itertools.repeat(fixed), steps, 0.0, "cpu")
        norms.append(norm)
    assert math.isclose(norms[0], norms[1], rel_tol=1e-3), norms

    
    # run_evals
    class ScoreTask:
        def __init__(self, n, score):
            self.n = n
            self.score = score
        def __len__(self):
            return self.n
        def __getitem__(self, i):
            return {"messages": [{"role": "user", "content": f"Q{i}?"}, {"role": "assistant", "content": "A"}], "letters": "ABCD"}
        def evaluate(self, conversation, completion):
            return self.score

    torch.manual_seed(0)
    tiny = GPT(GPTConfig(block_size=64, vocab_size=50304, n_layer=2, n_head=2, n_embd=32))
    results = run_evals(tiny, tiny, task, {"all_right": ScoreTask(4, 1), "all_wrong": ScoreTask(4, 0)}, 2, 31, "cpu")
    assert list(results.keys()) == ["val_loss", "all_right", "all_wrong", "chatcore"]
    assert abs(results["val_loss"] - math.log(50304)) < 0.5
    assert results["all_right"] == 1.0 and results["all_wrong"] == 0.0
    assert math.isclose(results["chatcore"], (1.0 + (-1 / 3)) / 2)
    assert tiny.training


    # train_sft
    # Build a small checkpoint whose optimizer was pretrained with weight decay
    torch.manual_seed(0)
    original = GPT(GPTConfig(block_size=64, vocab_size=50304, n_layer=2, n_head=2, n_embd=32))
    pre_optimizer = original.configure_optimizers(weight_decay=0.1, learning_rate=1e-3, device="cpu")
    _, loss = original(*fixed)
    loss.backward()
    pre_optimizer.step()

    folder = tempfile.mkdtemp()
    base_path = os.path.join(folder, "model_base.pt")
    torch.save({"model": original.state_dict(), "config": original.config, "step": 1, "optimizer": pre_optimizer.state_dict()}, base_path)

    train_task = [conv_of("Hi", "Hello there.") for _ in range(100)]
    log_dir = os.path.join(folder, "log_sft")
    model, optimizer, results = train_sft(base_path, train_task, train_task, {"all_right": ScoreTask(4, 1)}, log_dir, "cpu", B=2, T=31, total_batch_size=124, max_lr=3e-3, eval_every=5)
    num_steps = sum(1 for _ in chat_batches(train_task, 2, 31)) // 2

    # Weight decay is 0, and the optimizer state continued from the checkpoint
    assert all(group["weight_decay"] == 0.0 for group in optimizer.param_groups)
    first_param = optimizer.param_groups[0]["params"][0]
    assert optimizer.state[first_param]["step"].item() == 1 + num_steps

    # The model learned something
    assert results["val_loss"] < math.log(50304) - 1, results

    # The log has evals at the start and end, and one train line per step
    lines = open(os.path.join(log_dir, "log.txt")).read().splitlines()
    assert lines[0].startswith("0 val_loss") and lines[-1].startswith(f"{num_steps} chatcore")
    assert sum(" train " in line for line in lines) == num_steps

    # The saved model reloads with the same weights
    reloaded, saved = load_checkpoint(os.path.join(log_dir, "model_sft.pt"), "cpu")
    x = torch.randint(0, 50304, (2, 10))
    assert torch.equal(reloaded.eval()(x)[0], model.eval()(x)[0])
    assert saved["step"] == num_steps and saved["results"] == results


    print("all tests passed")
    

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="run the tests and stop")
    parser.add_argument("--checkpoint", help="pretrained model path")
    parser.add_argument("--log-dir", help="a new folder for this run")
    parser.add_argument("--init-lr-frac", type=float, default=0.8)
    parser.add_argument("--max-examples", type=int, default=1000, help="questions per task during training")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N train conversations")
    args = parser.parse_args()

    if not args.test and (args.checkpoint is None or args.log_dir is None):
        parser.error("--checkpoint and --log-dir are required unless --test is set")

    if args.test:
        run_tests()
        sys.exit(0)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    print(f"using device: {device}")

    torch.set_float32_matmul_precision("high")
    train_task, val_task, eval_tasks = build_tasks()
    if args.limit:
        train_task.index_map = train_task.index_map[:args.limit]

    train_sft(args.checkpoint, train_task, val_task, eval_tasks, args.log_dir, device,
            init_lr_frac=args.init_lr_frac, max_examples=args.max_examples, use_compile=(device == "cuda"))
