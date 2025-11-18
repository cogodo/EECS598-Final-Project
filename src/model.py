import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# from datasets import load_dataset
import yaml
from tqdm import tqdm
import pandas as pd


class TinyLlama:
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # needed for loading model
        self.model_task = self.config["model"]["task"]
        self.model_name = self.config["model"]["name"]
        if self.config["model"]["torch_dtype"] == "torch.bfloat16":
            self.model_dtype = torch.bfloat16
        else:
            raise ValueError(f'Add datatype: {self.config["model"]["torch_dtype"]}')

        self.model_device_map = self.config["model"]["device_map"]

        # needed for loading dataset
        self.train_dataset_path = self.config["data"]["train"]["path"]
        self.test_dataset_path = self.config["data"]["test"]["path"]

        self.model = None
        self.tokenizer = None
        self.trainer = None

    # loading tinyllama        
    def load_model(self):
        print("Loading model and tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=self.model_dtype,
            device_map="auto"
        )


    def load_data(self):
        print("Loading datasets...")
        dataset = pd.read_json(self.train_dataset_path , lines=True)
        val_dataset = pd.read_json(self.test_dataset_path , lines=True)

        dataset_question = self.tokenizer.apply_chat_template(
            [
                [{"role": "system", "content": q}]
                for q in dataset["question"]
            ],
            tokenize=False,
            add_generation_prompt=True
        )


        dataset_answer = self.tokenizer.apply_chat_template(
            [
                [{"role": "system", "content": a}]
                for a in dataset["answer"]
            ],
            tokenize=False,
            add_generation_prompt=True
        )


        val_dataset_question = self.tokenizer.apply_chat_template(
            [
                [{"role": "system", "content": q}]
                for q in val_dataset["question"]
            ],
            tokenize=False,
            add_generation_prompt=True
        )

        val_dataset_answer = self.tokenizer.apply_chat_template(
            [
                [{"role": "system", "content": a}]
                for a in val_dataset["answer"]
            ],
            tokenize=False,
            add_generation_prompt=True
        )

        # validation dataset answer
        val_dataset_answer = [
            self.tokenizer.apply_chat_template(
                [{"role": "system", "content": a}],
                tokenize=False,
                add_generation_prompt=True
            )
            for a in val_dataset["answer"]
        ]

        print("Datasets Loaded")

        return dataset_question, val_dataset_question, dataset_answer, val_dataset_answer

    def inference(self, dataset_question):

        print("Asking Quesiton")

        # Tokenize the list of input prompts
        encodings = self.tokenizer(
            dataset_question,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)

        # Use .generate()
        outputs = self.model.generate(
            **encodings,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            return_dict_in_generate=False  # <-- faster
        )

        # Decode each output
        decoded = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )

        print("Answer Quesiton")

        return decoded


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
    

if __name__ == "__main__":

    model = TinyLlama()

    model.load_model()

    dataset_question, val_dataset_question, dataset_answer, val_dataset_answer = model.load_data()

    dataset_question_1 = dataset_question[0:20]

    answer = model.inference(dataset_question_1)

    print("answer")
    print(answer)