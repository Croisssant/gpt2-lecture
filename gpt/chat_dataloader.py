import torch
from gpt.chat_tokenizer import special_token_ids, render_conversation, bos

def trim_to_last_reply(ids, mask, max_tokens):
    smaller = min(len(ids), max_tokens)

    for i in range(smaller - 1, 0, -1):
        if ids[i] == special_token_ids["<|assistant_end|>"]:
            return ids[:i+1], mask[:i+1]

    return None


def pack_row(buffer, row_len, pad):
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
    tensor_ids = torch.tensor([row_ids for row_ids, _ in rows], dtype=torch.long)
    tensor_masks = torch.tensor([row_mask for _, row_mask in rows], dtype=torch.long)

    x = tensor_ids[:, :-1]
    y = tensor_ids[:, 1:].clone()
    y[tensor_masks[:, 1:] == 0] = -1
    
    return x, y


def chat_batches(task, B, T, buffer_size=100):
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
