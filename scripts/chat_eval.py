import torch
from gpt.chat_tokenizer import render_for_completion, enc, bos, special_token_ids

@torch.no_grad()
def evaluate_categorical(model, task, device, batch_size=8, max_examples=None):
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

    print("all tests passed")
