import datasets
import re

def extract_answer(text):
    pattern = r"#### (\-?[0-9\.\,]+)"
    match = re.search(pattern, text)

    if match:
        result = match.group(1).replace(",", "")
        return result
        
    return None

class GSM8K:
    def __init__(self, split):
        assert split in ["train", "test"], "Split can only be either train or test"
        data = datasets.load_dataset("openai/gsm8k", "main", split=split)
        self.ds = data.shuffle(seed=42)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        ds_data = self.ds[index]
        ds_answer = ds_data["answer"]
        ds_question = ds_data["question"]

        matches = re.split(r"(<<[^>]+>>)", ds_answer)
        create_content = lambda part_type, text: {"type": part_type, "text": text}
        content = []

        for piece in matches:
            if piece.startswith("<<"):
                piece = piece[2:-2]
                expr, result = piece.rsplit("=", 1)

                content.append(create_content("python", expr))
                content.append(create_content("python_output", result))

            else:
                content.append(create_content("text", piece))

        return {"messages": [
            {"role": "user", "content": ds_question},
            {"role": "assistant", "content": content}
        ]}
    
    def evaluate(self, conversation, completion):
        return int(extract_answer(completion) == extract_answer(conversation["messages"][-1]["content"][-1]["text"]))

    def reward(self, conversation, completion):
        return float(self.evaluate(conversation, completion))


if __name__ == "__main__":
    from gpt.chat_tokenizer import render_conversation, render_for_completion, enc

    # extract_answer
    assert extract_answer("so she sold #### 72") == "72"
    assert extract_answer("#### 1,234") == "1234"
    assert extract_answer("#### -5") == "-5"
    assert extract_answer("#### 3.5") == "3.5"
    assert extract_answer("the answer is 72") is None

    train, test = GSM8K("train"), GSM8K("test")
    assert len(train) == 7473 and len(test) == 1319

    # __getitem__: the conversation has the right shape
    conv = train[0]
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant"]
    assert conv["messages"][0]["content"] == train.ds[0]["question"]

    # Only the assistant's text and python calls are trained on, not the calculator results
    ids, mask = render_conversation(conv)
    trained = enc.decode([t for t, m in zip(ids, mask) if m == 1])
    assert "<|python_start|>2*12<|python_end|>" in trained, trained
    assert "<|output_start|>" not in trained and trained.endswith("#### 16<|assistant_end|>"), trained
    assert enc.decode(render_for_completion(conv)).endswith("<|user_end|><|assistant_start|>")

    # evaluate and reward
    assert train.evaluate(conv, "blah #### 16") == 1
    assert train.evaluate(conv, "blah #### 17") == 0
    assert train.evaluate(conv, "no answer") == 0
    assert train.reward(conv, "blah #### 16") == 1.0 and isinstance(train.reward(conv, "blah #### 16"), float)
    assert train.reward(conv, "blah #### 17") == 0.0

    # Every reference solution should score correct against itself
    for task in [train, test]:
        for i in range(len(task)):
            c = task[i]
            assert task.evaluate(c, c["messages"][-1]["content"][-1]["text"]) == 1, i

    print("all tests passed")
