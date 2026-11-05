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