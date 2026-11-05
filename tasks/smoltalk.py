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