import random

def render_mc(question:str, letters:str|list[str], choices:list[str]) -> str:
    rendered = [f"Multiple Choice question: {question}\n"]

    for letter, choice in zip(letters, choices):
        rendered.append(f"- {choice}={letter}\n")

    rendered.append("\nRespond only with the letter of the correct answer.")

    return "".join(rendered)


class TaskMixture:
    def __init__(self, tasks):
        self.tasks = tasks
        self.index_map = [(task_idx, local_idx) for task_idx, task in enumerate(tasks) for local_idx in range(len(task))]
        random.Random(42).shuffle(self.index_map)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        task_idx, local_idx = self.index_map[index]
        return self.tasks[task_idx][local_idx]


if __name__ == "__main__":
    from collections import Counter
    from tasks.smoltalk import SmolTalk
    from tasks.mmlu import MMLU
    from tasks.gsm8k import GSM8K
    from gpt.chat_tokenizer import render_conversation

    question = "Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q."
    expected = "Multiple Choice question: Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.\n- 0=A\n- 4=B\n- 2=C\n- 6=D\n\nRespond only with the letter of the correct answer."
    assert render_mc(question, "ABCD", ["0", "4", "2", "6"]) == expected

    # Small lists: every item appears once, in a fixed shuffled order
    a, b = ["a0", "a1", "a2"], ["b0", "b1"]
    m = TaskMixture([a, b])
    items = [m[i] for i in range(len(m))]
    assert len(m) == 5
    assert items == ["b0", "a1", "a2", "b1", "a0"], items
    assert items == [TaskMixture([a, b])[i] for i in range(5)], "same order every run"

    # Listing a task twice repeats its items
    r = TaskMixture([a, a, b])
    assert Counter(r[i] for i in range(len(r))) == {"a0": 2, "a1": 2, "a2": 2, "b0": 1, "b1": 1}

    # The real SFT training mix
    smol, mmlu, gsm = SmolTalk("train"), MMLU("auxiliary_train"), GSM8K("train")
    train = TaskMixture([smol] + [mmlu] * 3 + [gsm] * 4)
    assert len(train) == 789759
    for i in range(2000):
        render_conversation(train[i])

    print("all tests passed")
