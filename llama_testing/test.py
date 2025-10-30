import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    pipeline
)
from datasets import load_dataset

pipe = pipeline(
    "text-generation",
    model="/home/advaithb/eecs598/tinyllama-finetuned-yelp/checkpoint-1900",
    tokenizer="/home/advaithb/eecs598/tinyllama-finetuned-yelp/checkpoint-1900",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

prompt = "Summarize this review:\nI loved the sushi and the service was amazing!"
output = pipe(prompt, max_new_tokens=50, temperature=0.7, top_p=0.9)
print("\n=== Test Output ===")
print(output[0]["generated_text"])