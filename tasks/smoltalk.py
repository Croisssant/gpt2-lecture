import datasets

class SmolTalk:
    def __init__(self, split):
        assert split in ["train", "test"], "Split can only be either train or test"
        data = datasets.load_dataset("HuggingFaceTB/smol-smoltalk", split=split)
        self.ds = data.shuffle(seed=42)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        return {"messages": self.ds[index]["messages"]}


if __name__ == "__main__":
    from gpt.chat_tokenizer import render_conversation, enc

    train, test = SmolTalk("train"), SmolTalk("test")
    assert len(train) == 460341 and len(test) == 24229

    # __getitem__: only the messages are returned
    conv = train[0]
    assert list(conv.keys()) == ["messages"]

    # Check 1000 conversations: they all render, and only assistant replies are trained on
    for i in range(1000):
        conv = train[i]
        ids, mask = render_conversation(conv, max_tokens=10**7)
        trained = enc.decode([t for t, m in zip(ids, mask) if m == 1])
        num_replies = sum(m["role"] == "assistant" for m in conv["messages"])
        assert len(ids) == len(mask)
        assert trained.count("<|assistant_end|>") == num_replies, i
        assert trained.endswith("<|assistant_end|>"), i

    print("all tests passed")
