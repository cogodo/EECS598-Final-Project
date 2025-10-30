import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    pipeline
)
from datasets import load_dataset

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = load_dataset("yelp_review_full", split="train[:1%]")
val_dataset = load_dataset("yelp_review_full", split="test[:1%]")

def format_example(text):
    return f"Summarize this review:\n{text}"

def tokenize_function(examples):
    texts = [format_example(t) for t in examples["text"]]
    tokens = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=256
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_train = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
tokenized_val = val_dataset.map(tokenize_function, batched=True, remove_columns=val_dataset.column_names)

training_args = TrainingArguments(
    output_dir="./tinyllama-finetuned-yelp",
    eval_steps=100,
    logging_steps=50,
    save_steps=100,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    bf16=True,  # or fp16=True if your GPU supports it
    learning_rate=2e-5,
    warmup_steps=50,
    save_total_limit=2,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
)

trainer.train()

trainer.save_model("./tinyllama-finetuned-yelp")
tokenizer.save_pretrained("./tinyllama-finetuned-yelp")

pipe = pipeline(
    "text-generation",
    model="./tinyllama-finetuned-yelp",
    tokenizer="./tinyllama-finetuned-yelp",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

prompt = "Summarize this review:\nI loved the sushi and the service was amazing!"
output = pipe(prompt, max_new_tokens=50, temperature=0.7, top_p=0.9)
print("\n=== Test Output ===")
print(output[0]["generated_text"])
