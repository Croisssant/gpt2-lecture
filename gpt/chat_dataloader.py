import torch
from gpt.chat_tokenizer import special_token_ids, render_conversation, bos

def trim_to_last_reply(ids, mask, max_tokens):
    """Cut a rendered conversation so it fits in max_tokens and still ends on a complete reply.

    Plain truncation at max_tokens could stop mid-reply, and in MMLU could even cut off the answer
    letter, leaving a question with no answer to learn. Instead, keep everything up to and
    including the last <|assistant_end|> that fits.

    Args:
        ids, mask: from render_conversation(..., max_tokens=10**7), so nothing is cut yet.
        max_tokens: the longest allowed result, i.e. one row (T + 1 = 1025).

    Returns:
        (ids, mask) cut just after the last <|assistant_end|> within the first max_tokens tokens,
        or None if not even the first reply fits; the caller then skips the conversation
        (1.16% of the SFT mix).

    Also removes a trailing user message that has no reply (258 SmolTalk conversations end that way).

    Example (conversation: "Hi there" / "Hello!" / "Bye" / "See you.", 18 tokens, first reply ends at index 8):
        max_tokens=100 -> all 18 tokens
        max_tokens=12  -> ids[:9], only the first exchange
        max_tokens=8   -> None
    """
    smaller = min(len(ids), max_tokens)

    for i in range(smaller - 1, 0, -1):
        if ids[i] == special_token_ids["<|assistant_end|>"]:
            return ids[:i+1], mask[:i+1]

    return None


def pack_row(buffer, row_len, pad):
    """Fill one training row with whole conversations from the buffer (best-fit packing).

    Conversations are short (often 100-500 tokens) but a row is 1025, so padding each one to a
    full row would waste most of the batch. Packing several per row keeps the GPU busy. Each
    conversation starts with its own BOS, so the model can tell where one ends and the next begins.

    Args:
        buffer: a list of (ids, mask) pairs, each at most row_len long. Modified in place:
            conversations placed in the row are popped out.
        row_len: the row length, T + 1.
        pad: the token used to fill leftover space (BOS).

    What it does:
        Repeatedly pick the longest conversation that still fits in the remaining space, pop it
        and append it. When nothing fits, pad the rest with `pad` (mask 0, so it isn't trained on).

    Returns:
        (row_ids, row_mask): two lists of exactly row_len ints.

    Example (row_len=1025, buffer lengths [600, 300, 200, 50]):
        take 600 (425 left) -> take 300, the longest that fits (125 left) -> take 50 (75 left)
        -> 200 no longer fits, so pad 75
        row = 600 + 300 + 50 + 75 padding; the 200 stays in the buffer for the next row
    """
    row_ids, row_mask = [], []

    while True:
        empty_space = row_len - len(row_ids)

        best_index, best_len = None, 0
        for i, (ids, _) in enumerate(buffer):
            if best_len < len(ids) <= empty_space:
                best_index, best_len = i, len(ids)

        if best_index is None:
            break

        ids, mask = buffer.pop(best_index)
        row_ids.extend(ids)
        row_mask.extend(mask)
  
    row_ids.extend([pad] * empty_space)
    row_mask.extend([0] * empty_space)

    return row_ids, row_mask


def build_xy(rows):
    """Turn packed rows into the model's inputs x and targets y, ignoring tokens with mask 0.

    Used by chat_batches for SFT and by rollout for RL.

    Args:
        rows: a list of B (ids, mask) pairs, all of the same length L.

    Returns:
        x: (B, L-1) long tensor, ids[:, :-1], what the model reads.
        y: (B, L-1) long tensor, ids[:, 1:], the next token at each position, set to -1 wherever
           the *target's* mask is 0. The model's cross_entropy uses ignore_index=-1, so those
           positions add nothing to the loss: the model is never trained to predict user text,
           BOS, padding or calculator output.
        Both are on the CPU; the caller moves them to the device.

    Example:
        ids  = [BOS, <|user_start|>, Hi, there, <|user_end|>, <|assistant_start|>, Hello, !, <|assistant_end|>]
        mask = [0,   0,              0,  0,     0,            0,                   1,     1, 1]
        x    = [BOS, <|user_start|>, Hi, there, <|user_end|>, <|assistant_start|>, Hello, !]
        y    = [-1,  -1,             -1, -1,    -1,           Hello,               !,     <|assistant_end|>]
        The mask is shifted with the targets, so "Hello" is predicted from <|assistant_start|>.
    """
    tensor_ids = torch.tensor([row_ids for row_ids, _ in rows], dtype=torch.long)
    tensor_masks = torch.tensor([row_mask for _, row_mask in rows], dtype=torch.long)

    x = tensor_ids[:, :-1]
    y = tensor_ids[:, 1:].clone()
    y[tensor_masks[:, 1:] == 0] = -1
    
    return x, y


def chat_batches(task, B, T, buffer_size=100):
    """Generate packed (x, y) SFT batches from a task, one pass over it in order.

    The SFT replacement for pretraining's DataLoaderLite: instead of cutting one long token stream
    into rows, it packs whole conversations into rows so that every row starts at a conversation
    boundary and only assistant tokens are trained on.

    Args:
        task: anything with len() and [i] returning a conversation (a task, TaskMixture or list).
        B: rows per batch.
        T: tokens per row seen by the model; rows are built T + 1 long so x and y are each T.
        buffer_size: how many conversations to keep ready for pack_row to choose from. Bigger
            means tighter packing but more memory; 100 gives about 55% graded tokens on the SFT mix.

    What it does:
        1. Top the buffer up to buffer_size: render each conversation in full, trim it with
           trim_to_last_reply (skipping any that return None) and add it.
        2. Build one row with pack_row.
        3. Every B rows, yield build_xy(rows).
        Stops when the task and the buffer are both empty. A final incomplete batch is dropped.

    Yields:
        (x, y): each a (B, T) long tensor on the CPU, with y = -1 where nothing is trained.

    Notes:
        - It's a generator, so nothing is computed until you loop over it; each new call starts
          again from the first conversation.
        - The task's order is used as-is, so shuffle in the task (TaskMixture does).
        - Full SFT mix with B=32, T=1024: 12,178 batches, about 399M tokens.
    """
    row_len = T + 1
    buffer, rows = [], []
    index = 0

    while True:
        while len(buffer) < buffer_size and index < len(task):
            ids, mask = render_conversation(task[index], max_tokens=10**7)
            index += 1

            trimmed = trim_to_last_reply(ids, mask, row_len)

            if trimmed is not None:
                buffer.append(trimmed)

        if not buffer:
            return

        rows.append(pack_row(buffer=buffer, row_len=row_len, pad=bos))

        if len(rows) >= B:
            yield build_xy(rows)
            rows = []



if __name__ == "__main__":
    from gpt.chat_tokenizer import enc

    conv = {"messages": [
        {"role": "user", "content": "Hi there"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "Bye"},
        {"role": "assistant", "content": "See you."},
    ]}
    ids, mask = render_conversation(conv, max_tokens=10**7)

    # Everything fits
    assert trim_to_last_reply(ids, mask, 100) == (ids, mask)
    assert trim_to_last_reply(ids, mask, len(ids)) == (ids, mask)

    # The second reply doesn't fit, so only the first is kept
    first = trim_to_last_reply(ids, mask, 12)
    assert first == (ids[:9], mask[:9])
    assert enc.decode(first[0]).endswith("Hello!<|assistant_end|>")

    # The boundary: the first reply only just fits, then no longer fits
    assert trim_to_last_reply(ids, mask, 9) == (ids[:9], mask[:9])
    assert trim_to_last_reply(ids, mask, 8) is None

    # A final user message with no reply gets removed
    trailing = {"messages": conv["messages"] + [{"role": "user", "content": "Thanks"}]}
    tids, tmask = render_conversation(trailing, max_tokens=10**7)
    assert trim_to_last_reply(tids, tmask, 100) == (ids, mask)

    # pack_row: the sample above
    buffer = [([1] * 600, [1] * 600), ([2] * 300, [1] * 300), ([3] * 200, [1] * 200), ([4] * 50, [1] * 50)]
    row_ids, row_mask = pack_row(buffer, 1025, 0)
    assert row_ids == [1] * 600 + [2] * 300 + [4] * 50 + [0] * 75
    assert row_mask == [1] * 950 + [0] * 75
    assert buffer == [([3] * 200, [1] * 200)]

    # Exact fit: no padding
    buffer = [([1] * 5, [1] * 5), ([2] * 5, [0] * 5)]
    row_ids, row_mask = pack_row(buffer, 10, 0)
    assert row_ids == [1] * 5 + [2] * 5 and row_mask == [1] * 5 + [0] * 5 and buffer == []

    # Nothing fits: the row is all padding and the buffer is untouched
    buffer = [([1] * 20, [1] * 20)]
    row_ids, row_mask = pack_row(buffer, 10, 0)
    assert row_ids == [0] * 10 and row_mask == [0] * 10 and len(buffer) == 1

    # build_xy: the sample above
    ids = [50256, 50257, 17250, 612, 50258, 50259, 15496, 0, 50260]
    mask = [0, 0, 0, 0, 0, 0, 1, 1, 1]
    x, y = build_xy([(ids, mask)])
    assert x.tolist() == [[50256, 50257, 17250, 612, 50258, 50259, 15496, 0]]
    assert y.tolist() == [[-1, -1, -1, -1, -1, 15496, 0, 50260]]

    # Two rows: the batch has two rows, and each is masked separately
    x, y = build_xy([([1, 2, 3], [0, 1, 1]), ([4, 5, 6], [1, 0, 1])])
    assert x.tolist() == [[1, 2], [4, 5]]
    assert y.tolist() == [[2, 3], [-1, 6]]

    # With pack_row: only the replies are graded, not BOS or padding
    conv = {"messages": [
        {"role": "user", "content": "Hi there"},
        {"role": "assistant", "content": "Hello!"},
    ]}
    i1, m1 = trim_to_last_reply(*render_conversation(conv, max_tokens=10**7), 1025)
    row = pack_row([(i1, m1), (i1, m1)], 30, bos)
    x, y = build_xy([row])
    assert x.shape == (1, 29) and y.shape == (1, 29)
    assert enc.decode(y[y != -1].tolist()) == "Hello!<|assistant_end|>Hello!<|assistant_end|>"


    print("all tests passed")
