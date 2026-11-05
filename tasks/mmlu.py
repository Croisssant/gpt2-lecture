import datasets

class MMLU:
    def __init__(self, split):
        assert split in ["auxiliary_train", "test"], "Split can only be either train or test"
        data = datasets.load_dataset("cais/mmlu", "all", split=split)
        self.ds = data.shuffle(seed=42)

    def __len__(self):
        return len(self.ds)


if __name__ == "__main__":
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

    print("all tests passed")
