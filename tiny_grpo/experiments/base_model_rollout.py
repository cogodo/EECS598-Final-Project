"""
Quick script to see what the base TinyLlama model outputs on math problems.
Useful as a baseline before training.
"""

import sys
import json
import torch
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
from math_verifier import MathVerifier

SYSTEM_PROMPT = """You are a helpful math assistant. Please solve the problem step by step, showing your reasoning clearly. 
Once you have solved the problem, provide your final numerical answer wrapped in <answer> tags, like this: <answer>number</answer>"""


def load_base_model(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", device: str = "cuda"):
    """Load the base model without any fine-tuning."""
    print(f"Loading base model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.config.pad_token_id = tokenizer.eos_token_id
    model.eval()
    
    return model, tokenizer


def read_jsonl(file_path: str | Path):
    """Read JSONL file."""
    data = []
    with Path(file_path).open(mode="r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


@torch.no_grad()
def generate_response(
    model,
    tokenizer,
    question: str,
    num_samples: int = 1,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
):
    """Generate response(s) for a single question."""
    
    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"].repeat(num_samples, 1)
    attention_mask = inputs["attention_mask"].repeat(num_samples, 1)
    
    gen_config = GenerationConfig(
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=gen_config
    )
    
    completions = tokenizer.batch_decode(
        outputs[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    
    return completions


def main():
    parser = argparse.ArgumentParser(description="Test base model on math problems")
    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Base model name"
    )
    parser.add_argument(
        "--num_questions",
        type=int,
        default=5,
        help="Number of questions to test"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples per question"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Max tokens to generate"
    )
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_base_model(args.model)
    
    # Load test data
    test_path = Path(__file__).parent.parent / "data" / "test.jsonl"
    test_data = read_jsonl(test_path)[:args.num_questions]
    print(f"Testing on {len(test_data)} questions\n")
    
    # Initialize verifier
    verifier = MathVerifier(method="flexible", correct_reward=1.0, format_reward=0.0)
    
    correct = 0
    total = 0
    
    for i, item in enumerate(test_data):
        question = item["question"]
        oracle_answer = item["answer"]
        # Extract just the final answer if it has ####
        if "####" in oracle_answer:
            oracle_answer = oracle_answer.split("####")[-1].strip()
        
        print(f"{'='*70}")
        print(f"Question {i+1}/{len(test_data)}")
        print(f"{'='*70}")
        print(f"Q: {question[:300]}{'...' if len(question) > 300 else ''}")
        print(f"\nOracle Answer: {oracle_answer}")
        print()
        
        # Generate responses
        completions = generate_response(
            model, tokenizer, question,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        
        for j, completion in enumerate(completions):
            # Verify correctness
            result = verifier.verify(question, completion, oracle_answer)
            is_correct = result["reward"] == 1.0
            status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
            
            if is_correct:
                correct += 1
            total += 1
            
            print(f"--- Response {j+1} {status} ---")
            print(completion[:600])
            if len(completion) > 600:
                print("... (truncated)")
            print()
    
    print(f"{'='*70}")
    print(f"SUMMARY: {correct}/{total} correct ({100*correct/total:.1f}%)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

