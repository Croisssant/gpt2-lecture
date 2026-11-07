import torch

from gpt.engine import generate
from gpt.chat_tokenizer import render_for_completion, enc, bos, special_token_ids

@torch.no_grad()
def evaluate_categorical(model, task, device, batch_size=8, max_examples=None):
    """Multiple-choice accuracy (MMLU, ARC): pick the answer letter the model scores highest.

    Rather than letting the model generate freely (it might write "The answer is B" or
    something invalid), this reads the model's prediction for the very first answer token and
    compares only the allowed letters. One forward pass per batch, no sampling.

    Args:
        model: the *uncompiled* model. Prompt lengths change from batch to batch, which would
            make a compiled model recompile.
        task: a task whose conversations include "letters", with an evaluate method.
        device: where to run.
        batch_size: questions per forward pass.
        max_examples: only use the first N questions (None = all). SFT uses 1000 during
            training to keep evals quick.

    What it does, for each batch:
        1. Build each prompt with render_for_completion (it ends with <|assistant_start|>).
        2. Pad the shorter prompts on the right with BOS, so they stack into one (B, T) tensor.
           Padding on the right is safe: attention is causal, so a prompt's last real position
           never sees the padding after it.
        3. Run the model, and for each row take the logits at position len(prompt) - 1, i.e. the
           prediction for the token after <|assistant_start|>.
        4. Keep only the logits of that question's letter tokens and take the argmax.
        5. task.evaluate(conversation, predicted_letter) gives 1 or 0.

    Returns:
        Accuracy, a float between 0 and 1. Random guessing scores about 0.25.
    """
    num_examples = len(task) if max_examples is None else min(max_examples, len(task))
    num_correct = 0

    for i in range(0, num_examples, batch_size):
        conversations, prompts = [], []

        for j in range(i, min(i + batch_size, num_examples)):
            conversations.append(task[j])
            prompts.append(render_for_completion(task[j]))

        max_length = max(len(prompt) for prompt in prompts)
        padded_prompts = [prompt + [bos] * (max_length - len(prompt)) for prompt in prompts]
        x = torch.tensor(padded_prompts, dtype=torch.long, device=device)

        logits, _ = model(x)

        for row, (conversation, prompt) in enumerate(zip(conversations, prompts)):
            letters = conversation["letters"]
            answer_position = len(prompt) - 1
            letter_ids = [enc.encode_ordinary(letter)[0] for letter in letters]
            scores = logits[row, answer_position, letter_ids]
            best = scores.argmax().item()
            predicted = letters[best]

            num_correct += task.evaluate(conversation, predicted)

    return num_correct / num_examples


def evaluate_gsm8k(model, task, num_samples=8, max_examples=400, max_new_tokens=256, temperature=1.0, top_k=50, seed=0):
    """GSM8K accuracy as pass@k: the fraction of questions solved by at least one of the first k samples.

    This is the number RL is meant to improve. train_rl calls it every eval_every steps, and it
    can also be run on its own on any checkpoint.

    Args:
        model: the uncompiled model (generate's shapes change every step).
        task: has len(), [i] and evaluate(conversation, completion), e.g. GSM8K("test").
        num_samples: samples per question; pass@k is reported for every k from 1 to this.
        max_examples: only use the first N questions (None = all 1,319). 400 keeps it to a few minutes.
        max_new_tokens, temperature, top_k: passed to generate.
        seed: question i uses seed + i, so each question gets different samples but the whole
            run is repeatable.

    What it does, for each question:
        1. Generate num_samples answers from render_for_completion(conversation).
        2. Decode each answer and score it with task.evaluate: 1 = correct, 0 = wrong.
        3. For each k, count the question as passed if any of its first k answers is correct.

    Returns:
        {k: passes / num_examples for k in 1..num_samples}, e.g. {1: 0.02, 2: 0.035, ..., 8: 0.08}.
        pass@k never decreases as k grows.

    The two ends mean different things:
        - pass@1 is how good the model is.
        - pass@num_samples is how often RL has something to learn from: a question only gives a
          gradient when its group contains at least one correct answer.

    Example (3 questions, 4 samples):
        correct = [0, 1, 0, 0], [0, 0, 0, 0], [1, 1, 0, 1]
        -> pass@1 = 1/3 (only q3's first sample), pass@2 = 2/3, pass@4 = 2/3

    The SFT model (log_sft_0.8) scores 0.0 for every k: it never writes the "#### <answer>" line
    that GSM8K.evaluate looks for.
    """

    num_examples = len(task) if max_examples is None else min(len(task), max_examples)
    passes = [0] * num_samples

    for i in range(num_examples):
        conversation = task[i]
        ids = render_for_completion(conversation=conversation)
        steps = list(generate(model, ids, num_samples=num_samples, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, seed=seed + i))

        correct_results = []
        for j in range(num_samples):
            gen_tokens = [tokens[j] for tokens, _ in steps]
            correct_results.append(task.evaluate(conversation, enc.decode(gen_tokens)))

        for k in range(1, num_samples + 1):
            passes[k - 1] += any(correct_results[:k])
            
    
    return {k: passes[k - 1] / num_examples for k in range(1, num_samples + 1)}


if __name__ == "__main__":
    from collections import Counter
    from tasks.mmlu import MMLU

    A, B = enc.encode_ordinary("A")[0], enc.encode_ordinary("B")[0]

    class FakeModel:
        def __call__(self, idx):
            logits = torch.zeros(*idx.shape, 100)
            logits[..., B] = 1.0
            at_answer = idx == special_token_ids["<|assistant_start|>"]
            logits[at_answer] = 0.0
            logits[at_answer, A] = 1.0
            return logits, None

    class ListTask:
        def __init__(self, conversations): self.conversations = conversations
        def __len__(self): return len(self.conversations)
        def __getitem__(self, i): return self.conversations[i]
        def evaluate(self, conversation, completion): return int(conversation["messages"][-1]["content"] == completion)

    def mc(prompt, answer):
        return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}], "letters": "ABCD"}

    # Different prompt lengths: the batch size must not change the result
    small = ListTask([mc("Short?", "A"), mc("A much longer question with quite a few more words in it?", "B"), mc("Medium length one?", "A"), mc("Q?", "C")])
    for batch_size in [1, 2, 8]:
        assert evaluate_categorical(FakeModel(), small, "cpu", batch_size=batch_size) == 0.5, batch_size

    # max_examples only uses the first N questions
    assert evaluate_categorical(FakeModel(), small, "cpu", max_examples=1) == 1.0

    # Real data: always answering A should score exactly the share of questions whose answer is A
    test = MMLU("test")
    share_of_a = Counter(test.ds["answer"])[0] / len(test)
    assert evaluate_categorical(FakeModel(), test, "cpu") == share_of_a
    print(f"always-A accuracy: {share_of_a:.4f}") # 0.2295

    # --- evaluate_gsm8k ---
    from gpt.model import GPT, GPTConfig

    torch.manual_seed(0)
    tiny = GPT(GPTConfig(block_size=128, vocab_size=50304, n_layer=1, n_head=1, n_embd=16)).eval()

    class FakeGSM8K:
        """10 identical questions; `rule` decides whether a completion is correct"""
        def __init__(self, rule):
            self.rule, self.calls = rule, 0
        def __len__(self):
            return 10
        def __getitem__(self, index):
            return {"messages": [{"role": "user", "content": "What is 2+2?"},
                                 {"role": "assistant", "content": "4"}]}
        def evaluate(self, conversation, completion):
            self.calls += 1
            return self.rule(completion)

    # Keys are 1..num_samples and values are fractions between 0 and 1
    task = FakeGSM8K(lambda c: int(len(c) % 2 == 0))
    result = evaluate_gsm8k(tiny, task, num_samples=4, max_examples=5, max_new_tokens=6)
    assert list(result) == [1, 2, 3, 4] and all(0 <= v <= 1 for v in result.values()), result

    # pass@k never decreases as k grows: more tries can only help
    assert all(result[k] <= result[k + 1] for k in range(1, 4)), result

    # max_examples limits the questions: 5 questions × 4 samples = 20 evaluations
    assert task.calls == 20, task.calls

    # A task that's always right scores 1.0 everywhere; one that's never right scores 0.0
    assert set(evaluate_gsm8k(tiny, FakeGSM8K(lambda c: 1), num_samples=3, max_examples=2, max_new_tokens=4).values()) == {1.0}
    assert set(evaluate_gsm8k(tiny, FakeGSM8K(lambda c: 0), num_samples=3, max_examples=2, max_new_tokens=4).values()) == {0.0}

    # The same seed gives the same result
    again = evaluate_gsm8k(tiny, FakeGSM8K(lambda c: int(len(c) % 2 == 0)), num_samples=4, max_examples=5, max_new_tokens=6)
    assert again == result

    # max_examples=None uses every question
    task = FakeGSM8K(lambda c: 0)
    evaluate_gsm8k(tiny, task, num_samples=2, max_examples=None, max_new_tokens=3)
    assert task.calls == 20


    print("all tests passed")
