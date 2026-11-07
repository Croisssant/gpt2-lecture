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
    """Turn a conversation into token ids, plus a mask saying which tokens to train on.

    Used by SFT (via chat_batches) to build training rows, and by render_for_completion and
    chat_cli to build prompts.

    Args:
        conversation: {"messages": [{"role": ..., "content": ...}, ...]}. Roles must alternate
            user, assistant, user, ..., optionally preceded by one system message.
            - user content: a string.
            - assistant content: a string, or a list of parts (GSM8K), each
              {"type": "text" | "python" | "python_output", "text": str}.
        max_tokens: cut both lists to this length. The default of 1025 is one SFT row
            (T + 1). Pass a large number such as 10**7 to keep everything.

    Returns:
        (ids, mask): two lists of ints of equal length.
        - mask 1: assistant text, python tool calls (<|python_start|>...<|python_end|>) and
          <|assistant_end|>, i.e. everything the model must learn to produce, including when to stop.
        - mask 0: BOS, user turns, <|assistant_start|> and calculator results
          (<|output_start|>...<|output_end|>), which come from the user or the tool.

    Layout:
        <|endoftext|> <|user_start|> question <|user_end|> <|assistant_start|> reply <|assistant_end|> ...

    Notes:
        - A system message is merged into the first user message as system + "\\n\\n" + user,
          on a deep copy, so the input is never modified.
        - Text is encoded with encode_ordinary, so a literal "<|endoftext|>" typed by a user stays
          plain text and is never read as a special token.

    Example:
        {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
        -> tokens: <|endoftext|> <|user_start|> Hi <|user_end|> <|assistant_start|> Hello ! <|assistant_end|>
           mask:   0             0              0  0            0                   1     1 1
    """
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
        """Append one id or a list of ids to `ids`, and the same number of `mask_value`s to `mask`."""

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


def render_for_completion(conversation):
    """Build the prompt the model answers from: the conversation without its last reply.

    Used at eval time (evaluate_categorical, evaluate_gsm8k) and in RL (rollout), where the
    task's conversation contains the reference answer that the model must not see.

    Args:
        conversation: as for render_conversation. The last message must be from the assistant.

    Returns:
        A list of token ids: the rendered conversation minus its last assistant message,
        followed by <|assistant_start|>, so the model's next token begins its answer.
        The input is not modified, and there is no mask (nothing is trained on here).

    Example:
        {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
        -> decoded: <|endoftext|><|user_start|>Hi<|user_end|><|assistant_start|>
    """
    assert conversation["messages"][-1]["role"] == "assistant", "Role for last message should be assistant"

    conversation = copy.deepcopy(conversation)
    conversation["messages"] = conversation["messages"][:-1]
    ids, _ = render_conversation(conversation)
    ids.append(special_token_ids["<|assistant_start|>"])
    return ids


if __name__ == "__main__":

    def check_raises(conversation, error):
        try:
            render_conversation(conversation)
        except error:
            return
        raise AssertionError(f"expected {error.__name__}")

    # 1. Multi-turn: only the assistant replies are trained on
    conv = {"messages": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "Bye"},
        {"role": "assistant", "content": "See you."},
    ]}
    ids, mask = render_conversation(conv)
    trained = enc.decode([t for t, m in zip(ids, mask) if m == 1])
    assert trained == "Hello!<|assistant_end|>See you.<|assistant_end|>", trained

    # 2. Literal special-token text in the content is treated as plain text
    conv = {"messages": [
        {"role": "user", "content": "What does <|endoftext|> mean?"},
        {"role": "assistant", "content": "It ends a document."},
    ]}
    ids, mask = render_conversation(conv)
    assert ids.count(bos) == 1, "only the real BOS should be token 50256"

    # 3. Rendering does not modify the input (system message merge works on a copy)
    conv = {"messages": [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Capital of France?"},
        {"role": "assistant", "content": "Paris."},
    ]}
    before = copy.deepcopy(conv)
    first = render_conversation(conv)
    second = render_conversation(conv)
    assert conv == before and first == second

    # 4. Truncation keeps ids and mask the same length
    ids, mask = render_conversation(conv, max_tokens=5)
    assert len(ids) == len(mask) == 5

    # 5. Tool use: the python call is trained on, the calculator output is not
    conv = {"messages": [
        {"role": "user", "content": "What is 12/60?"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "It is "},
            {"type": "python", "text": "12/60"},
            {"type": "python_output", "text": "0.2"},
            {"type": "text", "text": " #### 0.2"},
        ]},
    ]}
    ids, mask = render_conversation(conv)
    trained = enc.decode([t for t, m in zip(ids, mask) if m == 1])
    assert trained == "It is <|python_start|>12/60<|python_end|> #### 0.2<|assistant_end|>", trained
    assert special_token_ids["<|output_start|>"] in ids

    # 6. Invalid conversations raise
    check_raises({"messages": [{"role": "assistant", "content": "Hi"}]}, AssertionError)
    check_raises({"messages": [{"role": "user", "content": "Hi"},
                               {"role": "assistant", "content": [{"type": "image", "text": "x"}]}]}, ValueError)
    check_raises({"messages": [{"role": "user", "content": "Hi"},
                               {"role": "assistant", "content": 42}]}, ValueError)

    # 7. render_for_completion: drops the last assistant reply and primes the model to answer
    conv = {"messages": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]}
    ids = render_for_completion(conv)
    assert enc.decode(ids) == "<|endoftext|><|user_start|>Hi<|user_end|><|assistant_start|>", enc.decode(ids)
    assert len(conv["messages"]) == 2, "render_for_completion should not modify the input"

    # 8. render_for_completion: the last message must be from the assistant
    try:
        render_for_completion({"messages": [{"role": "user", "content": "Hi"}]})
        raise RuntimeError("expected AssertionError")
    except AssertionError:
        pass

    print("all tests passed")

