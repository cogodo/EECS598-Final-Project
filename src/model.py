import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import yaml

class TinyLlama:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", dataset_name="yelp_review_full"):
        with open('config.yaml') as f:
            self.config = yaml.safe_load(f)
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    def prep_data(self):
        
        dataset = load_dataset(self.dataset_name, split="train[:1%]")
        val_dataset = load_dataset(self.dataset_name, split="test[:1%]")

        def format_example(text):
            return f"Summarize this review:\n{text}"

        def tokenize_function(examples):
            texts = [format_example(t) for t in examples["text"]]
            tokens = self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=256
            )
            tokens["labels"] = tokens["input_ids"].copy()
            return tokens

        tokenized_train = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
        tokenized_val = val_dataset.map(tokenize_function, batched=True, remove_columns=val_dataset.column_names)
        
        return tokenized_train, tokenized_val

    def setup_trainer(self, train_data, val_data):

        training_args = TrainingArguments(**self.config['training'])

        self.trainer = Trainer(
        model=self.model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=self.tokenizer,
    )

    def train(self):
        self.trainer.train()

    def generate(self, prompt, max_new_tokens=256):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)