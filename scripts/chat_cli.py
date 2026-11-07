import argparse
import torch

from gpt.utils import load_checkpoint
from gpt.engine import generate, ASSISTANT_END
from gpt.chat_tokenizer import enc, render_conversation, special_token_ids

def chat(model, temperature=0.6, top_k=50, max_new_tokens=512):
    """Chat with the model in the terminal, streaming each reply as it's generated.

    Args:
        model: an SFT model in eval mode.
        temperature, top_k: sampling settings for generate. 0.6 gives fairly focused replies;
            raise it for more variety.
        max_new_tokens: the longest reply in tokens.

    Each turn:
        1. Read a line. "quit"/"exit" stops, "clear" starts a new conversation, empty input is ignored.
        2. Add it to the conversation as a user message.
        3. Render the whole conversation and append <|assistant_start|>.
        4. Print tokens from generate as they arrive, stopping at <|assistant_end|>.
        5. Add the reply to the conversation as an assistant message, so the next turn sees it.

    Known limitations:
        - The context is only 1024 tokens. Once the conversation outgrows it, generate keeps only
          the last 1024 tokens, and the model starts giving empty replies. Type "clear" to reset.
        - Tokens are decoded one at a time, so characters that span several tokens, such as
          emoji, can print as "�".
        - A calculator call in a reply is saved back into the history as plain text.
    """
    conversation = {"messages": []}

    while True:
        user_input = input("\nYou: ").strip()

        if user_input in ["quit", "exit"]:
            break

        if user_input == "clear":
            conversation = {"messages": []}
            print("Conversation Cleared!")
            continue

        if user_input == "":
            continue

        conversation["messages"].append({
            "role": "user",
            "content": user_input
        })

        ids, _ = render_conversation(conversation, max_tokens=10**7)
        ids.append(special_token_ids["<|assistant_start|>"])

        print("Assistant: ", end="", flush=True)
        reply= []
        for tokens, _ in generate(model, ids, temperature=temperature, top_k=top_k, max_new_tokens=max_new_tokens):
            token = tokens[0]
            if token == ASSISTANT_END:
                break

            
            reply.append(token)
            print(enc.decode([token]), end="", flush=True)

        print()
        conversation["messages"].append({"role": "assistant", "content": enc.decode(reply)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="log_sft_0.8/model_sft.pt", help="SFT model path")
    parser.add_argument("--temperature", type=float, default=0.6, help="0 = always the most likely token")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=512, help="longest reply")
    args = parser.parse_args()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

    model, _ = load_checkpoint(args.checkpoint, device)
    model.eval()

    print("Type 'clear' to start a new conversation, 'quit' to exit.")
    chat(model, temperature=args.temperature, top_k=args.top_k, max_new_tokens=args.max_new_tokens)
