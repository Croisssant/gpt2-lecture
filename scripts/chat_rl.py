import torch

import torch.nn.functional as F    # only the tests use it

from gpt.engine import generate
from gpt.chat_dataloader import build_xy
from gpt.chat_tokenizer import enc, render_for_completion


def rollout(model, task, index, num_samples=16, max_new_tokens=256, temperature=1.0, top_k=50, seed=0):
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

    # nll: negative log likelihood
    _, nll = model(x, targets=y, loss_reduction="none")
    logp = -nll
    num_valid = (y != -1).sum().clamp(min=1)
    loss = -(logp * advantages.unsqueeze(1)).sum() / num_valid

    return loss


def get_rl_lr(step, num_steps, max_lr=3e-5):
    return max_lr * (1 - step / num_steps)


if __name__ == "__main__":
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

    print("all tests passed")
