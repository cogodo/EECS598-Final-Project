import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Union

class AceRewardModel:
    def __init__(self, model_name: str = "nvidia/AceMath-7B-RM"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer first to handle padding logic
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            device_map=self.device,
            num_labels=1,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()

        # CRITICAL FIX: Sync model config with tokenizer to allow batch_size > 1
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def __call__(self, prompt: str, resp: str) -> List[float]:
        return self.compute_batch_reward(prompt, [resp])

    def compute_reward(self, prompt: str, resp: str) -> List[float]:
        return self.compute_batch_reward(prompt, [resp])

    def compute_batch_reward(self, prompt: str, responses: List[str]) -> List[float]:
        # Format conversation for the model
        batch_chats = [
            [
                {"role": "system", "content": "Please reason step by step, and check your final answer within \\boxed{}."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": resp}
            ] for resp in responses
        ]

        batch_strs = [
            self.tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False) 
            for c in batch_chats
        ]

        # Tokenize
        inputs = self.tokenizer(
            batch_strs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        scores = outputs.logits.squeeze(-1).tolist()
        
        # Ensure return is always a list
        return [scores] if isinstance(scores, float) else scores
