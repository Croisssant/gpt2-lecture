from transformers import GPT2LMHeadModel, pipeline, set_seed

model_hf = GPT2LMHeadModel.from_pretrained("gpt2") # 124M
sd_hf = model_hf.state_dict()

# for k, v in sd_hf.items():
#     print(k, v.shape)


generator = pipeline("text-generation", model="gpt2")
set_seed(42)
print(generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5))