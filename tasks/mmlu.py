import datasets
from tasks.common import render_mc

class MMLU:
    def __init__(self, split):
        assert split in ["auxiliary_train", "test"], "Split can only be either auxiliary_train or test"
        data = datasets.load_dataset("cais/mmlu", "all", split=split)
        self.ds = data.shuffle(seed=42)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        ds_row = self.ds[index]
        letters = "ABCD"
        correct_letter = letters[ds_row["answer"]]

        return {
            "messages": [
                {"role": "user", "content": render_mc(ds_row["question"], letters, ds_row["choices"])},
                {"role": "assistant", "content": correct_letter}
            ],
            "letters": letters
        }

    def evaluate(self, conversation, completion):
        return int(conversation["messages"][-1]["content"] == completion)

if __name__ == "__main__":
    from gpt.chat_tokenizer import render_conversation, enc
    
    train, test = MMLU("auxiliary_train"), MMLU("test")
    assert len(train) == 99842 and len(test) == 14042

    # The raw row has the fields __getitem__ will need
    row = test.ds[0]
    assert set(row.keys()) == {"question", "subject", "choices", "answer"}
    assert len(row["choices"]) == 4 and row["answer"] in [0, 1, 2, 3]

    # Invalid splits are rejected
    try:
        MMLU("dev")
        raise RuntimeError("expected AssertionError")
    except AssertionError:
        pass

    for task in [train, test]:
        for i in range(1000):
            conv = task[i]
            row = task.ds[i]
            assert [m["role"] for m in conv["messages"]] == ["user", "assistant"]
            assert conv["letters"] == "ABCD"
            assert conv["messages"][1]["content"] == "ABCD"[row["answer"]]
            assert row["question"] in conv["messages"][0]["content"]
            # Only the letter and <|assistant_end|> are trained on
            ids, mask = render_conversation(conv, max_tokens=10**7)
            trained = enc.decode([t for t, m in zip(ids, mask) if m == 1])
            assert trained == conv["messages"][1]["content"] + "<|assistant_end|>", trained

    # The eval relies on each letter being a single token
    for letter in "ABCD":
        assert len(enc.encode_ordinary(letter)) == 1

    # evaluate
    conv = test[0]
    assert test.evaluate(conv, "A") == 1
    for wrong in ["B", "C", "D", "a", " A", ""]:
        assert test.evaluate(conv, wrong) == 0, wrong

    # Exactly one of the four letters is correct for every question
    for i in range(len(test)):
        c = test[i]
        assert sum(test.evaluate(c, letter) for letter in c["letters"]) == 1, i

    print("all tests passed")