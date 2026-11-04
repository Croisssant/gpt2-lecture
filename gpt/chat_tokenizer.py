import tiktoken
import copy

base = tiktoken.get_encoding("gpt2")
CHAT_TOKENS = [
    "<|user_start|>", "<|user_end|>", "<|assistant_start|>", "<|assistant_end|>",
    "<|python_start|>", "<|python_end|>", "<|output_start|>", "<|output_end|>"
]
enc = tiktoken.Encoding(
    name="gpt2_chat",
    pat_str=base._pat_str,
    mergeable_ranks=base._mergeable_ranks,
    special_tokens={**base._special_tokens, **{t: base.n_vocab + i for i, t in enumerate(CHAT_TOKENS)}}
)

# beginning of sequence token
bos = enc.eot_token # reuse <|endoftext|> (50256) as BOS, since pretraining used it to separate documents
special_token_ids = {token: enc.encode_single_token(token) for token in CHAT_TOKENS}
assert enc.n_vocab <= 50304


def render_conversation(conversation, max_tokens=1025):
    messages = conversation["messages"]

    # Check if message begins with System Message
    if messages[0]["role"] == "system":
        messages = copy.deepcopy(messages)
        assert messages[1]["role"] == "user", "System message must be followed by a user message"
        messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
        messages = messages[1:]

    # Check if User and Assitant message alternates
    for idx, message in enumerate(messages):
        expected_role = "user" if idx % 2 == 0 else "assistant"
        assert message["role"] == expected_role, f"Message {idx} is from {message['role']}, expected {expected_role}"
    
    def add_tokens(token_ids, mask_value):

        if isinstance(token_ids, int):
            token_ids = [token_ids]

        assert isinstance(token_ids, list), f"token_ids must be int or list, received {token_ids} of type {type(token_ids)}"
        
        ids.extend(token_ids)
        mask.extend(len(token_ids) * [mask_value])

    ids, mask = [], []
    add_tokens(bos, 0)

    for message in messages:

        if message["role"] == "user":
            assert isinstance(message["content"], str), "Message must be of type string"
            add_tokens(special_token_ids["<|user_start|>"], 0)
            add_tokens(enc.encode_ordinary(message["content"]), 0)
            add_tokens(special_token_ids["<|user_end|>"], 0)

        elif message["role"] == "assistant":
            add_tokens(special_token_ids["<|assistant_start|>"], 0)

            if isinstance(message["content"], str):
                add_tokens(enc.encode_ordinary(message["content"]), 1)

            elif isinstance(message["content"], list):
                for content in message["content"]:
                    if content["type"] == "text":
                        add_tokens(enc.encode_ordinary(content["text"]), 1)

                    elif content["type"] == "python":
                        add_tokens(special_token_ids["<|python_start|>"], 1)
                        add_tokens(enc.encode_ordinary(content["text"]), 1)
                        add_tokens(special_token_ids["<|python_end|>"], 1)

                    elif content["type"] == "python_output":
                        add_tokens(special_token_ids["<|output_start|>"], 0)
                        add_tokens(enc.encode_ordinary(content["text"]), 0)
                        add_tokens(special_token_ids["<|output_end|>"], 0)

                    else:
                        raise ValueError(f"Assistant message is of unsupported type: {content['type']}. Only text, python and python output is supported for now.")

            else:
                raise ValueError(f"Assistant message['content'] is of unsupported type: {message['content']}. Only text and list is supported for now.")
            
            add_tokens(special_token_ids["<|assistant_end|>"], 1)


    return ids[:max_tokens], mask[:max_tokens]


if __name__ == "__main__":
    conversation = {
        "messages": [
            {"role": "system",    "content": "You are a helpful assistant who answers briefly."},
            {"role": "user",      "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris."},
        ]
    }

    ids, mask = render_conversation(conversation)
    print(len(ids) == len(mask))
    print(enc.decode(ids))
