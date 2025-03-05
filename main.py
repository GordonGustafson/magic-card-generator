
from transformers import pipeline, set_seed, GPT2LMHeadModel

saved_model_dir = "/home/ggustafson/stuff/magic-card-generator/fine_tuned_gpt2"

generator = pipeline('text-generation', model=saved_model_dir)
set_seed(42)
result = generator("name: Lightning Bolt manaCost: {R} type: Instant", max_length=50, num_return_sequences=5)

print(result)

# model = GPT2LMHeadModel.from_pretrained("/home/ggustafson/stuff/magic-card-generator/fine_tuned_gpt2")


