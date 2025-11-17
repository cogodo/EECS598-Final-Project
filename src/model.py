import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import yaml
from tqdm import tqdm

class TinyLlama:
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.model_name = self.config['model']['name']
        self.dataset_name = self.config['data']['dataset']
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def load_model(self):
        print("Loading model and tokenizer...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("Model loaded successfully!")
    
    def prep_data(self):
        print("Loading datasets...")
        dataset = load_dataset(self.dataset_name, split="train[:1%]")
        val_dataset = load_dataset(self.dataset_name, split="test[:1%]")
        print(f"Loaded {len(dataset)} training examples, {len(val_dataset)} validation examples")

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

        print("Tokenizing datasets...")
        tokenized_train = dataset.map(
            tokenize_function, 
            batched=True, 
            remove_columns=dataset.column_names,
            desc="Tokenizing train data"
        )
        tokenized_val = val_dataset.map(
            tokenize_function, 
            batched=True, 
            remove_columns=val_dataset.column_names,
            desc="Tokenizing validation data"
        )
        print("Tokenization complete!")
        
        return tokenized_train, tokenized_val

    def setup_trainer(self, train_data, val_data):
        print("Setting up trainer...")
        training_args = TrainingArguments(**self.config['training'])

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            tokenizer=self.tokenizer,
        )
        print("Trainer ready!")

    def train(self):
        print("Starting training...")
        self.trainer.train()  # Trainer has built-in progress bars
        print("Training complete!")

    def generate(self, prompt, max_new_tokens=256, use_chat_template=True):
        """Generate text from a prompt.
        
        Args:
            prompt: Input text or list of messages for chat template
            max_new_tokens: Maximum tokens to generate
            use_chat_template: Whether to format as chat (for TinyLlama-Chat models)
        """
        if use_chat_template and isinstance(prompt, str):
            # Format as chat for TinyLlama-Chat models
            messages = [
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt
            
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)