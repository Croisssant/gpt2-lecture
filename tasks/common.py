import random

def render_mc(question:str, letters:str|list[str], choices:list[str]) -> str:
    """Format a multiple-choice question as the user message for MMLU and ARC.

    Args:
        question: the question text.
        letters: one label per choice, e.g. "ABCD" or ARC's ["1", "2", "3", "4"]. Each must be
            a single token, because evaluate_categorical reads one logit per label.
        choices: the answer texts, in the same order as letters.

    Returns:
        A string in nanochat's format. Each choice is written as "choice=letter", with the letter
        *after* the text, so the model has read the choice before it sees which letter it belongs to.

    Example:
        render_mc("Find the degree for ... over Q.", "ABCD", ["0", "4", "2", "6"])
        ->  Multiple Choice question: Find the degree for ... over Q.
            - 0=A
            - 4=B
            - 2=C
            - 6=D

            Respond only with the letter of the correct answer.
    """
    rendered = [f"Multiple Choice question: {question}\n"]

    for letter, choice in zip(letters, choices):
        rendered.append(f"- {choice}={letter}\n")

    rendered.append("\nRespond only with the letter of the correct answer.")

    return "".join(rendered)


class TaskMixture:
    """Combine several tasks into one shuffled task, so SFT sees a mix rather than one dataset at a time.

    Args:
        tasks: a list of tasks (anything with len() and [i]). Listing a task several times repeats
            it, which is how the SFT mix gives MMLU 3 epochs and GSM8K 4:
            TaskMixture([smoltalk] + [mmlu] * 3 + [gsm8k] * 4) -> 789,759 conversations.

    How it works:
        index_map lists every (task_idx, local_idx) pair once, then shuffles it with a fixed seed
        (42), so the order is the same on every run and every machine. mixture[i] looks up pair i
        and returns that task's conversation. Nothing is copied or loaded in advance.

    Example:
        TaskMixture([["a0", "a1", "a2"], ["b0", "b1"]]) -> items in order: b0, a1, a2, b1, a0

    To train on a subset, cut index_map, e.g. mixture.index_map = mixture.index_map[:20000]
    (this is what chat_sft's --limit does).
    """

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
