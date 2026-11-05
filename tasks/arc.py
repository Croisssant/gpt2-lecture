import datasets
from tasks.common import render_mc

class ARC:
    def __init__(self, subset, split):
        assert split in ["train", "validation", "test"], "Split can only be either train, validation or test"
        assert subset in ["ARC-Easy", "ARC-Challenge"], "Only ARC-Easy or ARC-Challenge allowed in subset"
        data = datasets.load_dataset("allenai/ai2_arc", subset, split=split)
        self.ds = data.shuffle(seed=42)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        ds_row = self.ds[index]
        labels = ds_row["choices"]["label"]
        choices = ds_row["choices"]["text"]
        ans = ds_row["answerKey"]

        assert ans in labels, "Answer for this question is not found in the labels"

        return {
            "messages": [
                {"role": "user", "content": render_mc(ds_row["question"], labels, choices)},
                {"role": "assistant", "content": ans}
            ],
            "letters": labels
        }

    def evaluate(self, conversation, completion):
        return int(conversation["messages"][-1]["content"] == completion)
    
if __name__ == "__main__":
    from gpt.chat_tokenizer import render_conversation, enc

    easy, challenge = ARC("ARC-Easy", "test"), ARC("ARC-Challenge", "test")
    assert len(easy) == 2376 and len(challenge) == 1172

    # The raw row has the fields __getitem__ will need
    row = easy.ds[0]
    assert set(row.keys()) == {"id", "question", "choices", "answerKey"}
    assert set(row["choices"].keys()) == {"text", "label"}

    # Invalid arguments are rejected
    for bad in [("ARC-Hard", "test"), ("ARC-Easy", "dev")]:
        try:
            ARC(*bad)
            raise RuntimeError("expected AssertionError")
        except AssertionError:
            pass

    for task in [easy, challenge]:
        for i in range(len(task)):
            conv = task[i]
            row = task.ds[i]
            assert conv["messages"][1]["content"] == row["answerKey"]
            assert conv["letters"] == row["choices"]["label"]
            # The eval relies on each label being a single token
            for letter in conv["letters"]:
                assert len(enc.encode_ordinary(letter)) == 1
            # Only the answer and <|assistant_end|> are trained on
            ids, mask = render_conversation(conv)
            trained = enc.decode([t for t, m in zip(ids, mask) if m == 1])
            assert trained == row["answerKey"] + "<|assistant_end|>", trained

    # evaluate
    conv = easy[0]
    assert easy.evaluate(conv, "B") == 1
    for wrong in ["A", "C", "D", "b", ""]:
        assert easy.evaluate(conv, wrong) == 0, wrong

    # Numeric labels
    conv = easy[50]
    assert easy.evaluate(conv, "2") == 1
    assert easy.evaluate(conv, "B") == 0

    # Exactly one label is correct for every question
    for task in [easy, challenge]:
        for i in range(len(task)):
            c = task[i]
            assert sum(task.evaluate(c, letter) for letter in c["letters"]) == 1, i
            
    print("all tests passed")
