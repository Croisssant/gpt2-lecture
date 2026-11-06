import argparse
import torch

from gpt.utils import load_checkpoint
from gpt.engine import generate, ASSISTANT_END
from gpt.chat_tokenizer import enc, render_conversation, special_token_ids

def chat(model, temperature=0.6, top_k=50, max_new_tokens=512):
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
