import sys
import json
import torch
import argparse
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
from reward_model import AceRewardModel
from math_verifier import MathVerifier

SYSTEM_PROMPT = "You are a helpful math assistant. Solve the given problem step by step and provide your final answer wrapped in <answer> tags, like this: <answer>your answer here</answer>"

def load_model(checkpoint_path: str, device: str = "cuda", base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """Load model and tokenizer from checkpoint

    Handles both full checkpoints and LoRA adapter checkpoints.
    For LoRA checkpoints, loads base model first then applies adapters.
    """
    checkpoint_path = Path(checkpoint_path)
    print(f"Loading model from {checkpoint_path}...")

    # Check if this is a LoRA checkpoint (has adapter_config.json)
    is_lora = (checkpoint_path / "adapter_config.json").exists()

    if is_lora:
        print(f"Detected LoRA checkpoint, loading base model {base_model} first...")
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        # Load LoRA adapters
        model = PeftModel.from_pretrained(model, checkpoint_path)
        print("LoRA adapters loaded successfully")

        # Load tokenizer from checkpoint or base model
        try:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        except:
            tokenizer = AutoTokenizer.from_pretrained(base_model)
    else:
        print("Loading full model checkpoint...")
        # Load full model
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    model.eval()

    return model, tokenizer

def read_jsonl(file_path: str | Path) -> List[Dict]:
    """Read JSONL file"""
    data = []
    with Path(file_path).open(mode="r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

@torch.no_grad()
def generate_responses(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    questions: List[str],
    num_samples: int = 1,
    max_length: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    batch_size: int = 4,
) -> List[List[str]]:
    """Generate responses for a list of questions

    Returns:
        List of lists, where each inner list contains num_samples responses for a question
    """
    all_responses = []

    for i in tqdm(range(0, len(questions), batch_size), desc="Generating"):
        batch_questions = questions[i:i+batch_size]

        # Prepare batch of prompts, repeated for num_samples
        chat_prompts = []
        for q in batch_questions:
            chat_prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": q}],
                tokenize=False,
                add_generation_prompt=True
            )
            chat_prompts.extend([chat_prompt] * num_samples)

        # Tokenize
        model_inputs = tokenizer(
            chat_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(model.device)

        # Generate
        gen_config = GenerationConfig(
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            max_length=max_length,
            pad_token_id=tokenizer.eos_token_id,
        )

        outputs = model.generate(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            generation_config=gen_config
        )

        # Decode
        completions = tokenizer.batch_decode(
            outputs[:, model_inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        # Group by question
        for j in range(len(batch_questions)):
            start_idx = j * num_samples
            end_idx = start_idx + num_samples
            all_responses.append(completions[start_idx:end_idx])

    return all_responses

def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Compute pass@k metric
    n: total number of samples
    c: number of correct samples
    k: k in pass@k
    """
    if n - c < k:
        return 1.0
    return 1.0 - (
        sum(1.0 / (n - i) for i in range(k))
        / sum(1.0 / (n - i) for i in range(n - c, n))
        if c < n else 1.0
    )

def evaluate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    test_data: List[Dict],
    num_samples: int = 8,
    use_reward_model: bool = True,
    max_samples: int | None = None,
) -> Dict:
    """Evaluate model on test data"""

    # Initialize verifier and reward model
    verifier = MathVerifier(method="flexible", correct_reward=1.0, format_reward=0.0)
    reward_model = AceRewardModel() if use_reward_model else None

    # Prepare test data
    if max_samples:
        test_data = test_data[:max_samples]

    questions = [item["question"] for item in test_data]
    answers = [item["answer"] for item in test_data]

    # Generate responses
    print(f"Evaluating on {len(questions)} questions with {num_samples} samples each...")
    all_responses = generate_responses(
        model, tokenizer, questions,
        num_samples=num_samples,
        temperature=1.0,
        top_p=1.0
    )

    # Evaluate each question
    results = []
    total_correct = 0
    correct_counts = []

    for idx, (question, oracle_answer, responses) in enumerate(
        tqdm(zip(questions, answers, all_responses), total=len(questions), desc="Evaluating")
    ):
        question_results = {
            "question": question,
            "oracle_answer": oracle_answer,
            "responses": [],
            "num_correct": 0,
            "num_samples": num_samples,
        }

        # Evaluate each response
        for response in responses:
            # Verify correctness
            verify_result = verifier.verify(question, response, oracle_answer)
            is_correct = verify_result["reward"] == 1.0

            # Compute reward model score if available
            rm_score = None
            if reward_model:
                rm_score = reward_model.compute_batch_reward(question, [response])[0]

            question_results["responses"].append({
                "text": response,
                "correct": is_correct,
                "rm_score": rm_score,
            })

            if is_correct:
                question_results["num_correct"] += 1

        results.append(question_results)
        correct_counts.append(question_results["num_correct"])
        if question_results["num_correct"] > 0:
            total_correct += 1

    # Compute metrics
    accuracy = total_correct / len(questions)
    pass_at_1 = compute_pass_at_k(num_samples, sum(correct_counts), 1) if num_samples >= 1 else 0
    pass_at_5 = compute_pass_at_k(num_samples, sum(correct_counts), 5) if num_samples >= 5 else 0
    pass_at_8 = compute_pass_at_k(num_samples, sum(correct_counts), 8) if num_samples >= 8 else 0

    avg_correct_per_question = sum(correct_counts) / len(questions)

    # Compute average RM scores for correct vs incorrect
    if use_reward_model:
        correct_rm_scores = []
        incorrect_rm_scores = []

        for result in results:
            for resp in result["responses"]:
                if resp["rm_score"] is not None:
                    if resp["correct"]:
                        correct_rm_scores.append(resp["rm_score"])
                    else:
                        incorrect_rm_scores.append(resp["rm_score"])

        avg_rm_correct = sum(correct_rm_scores) / len(correct_rm_scores) if correct_rm_scores else 0
        avg_rm_incorrect = sum(incorrect_rm_scores) / len(incorrect_rm_scores) if incorrect_rm_scores else 0
    else:
        avg_rm_correct = avg_rm_incorrect = None

    metrics = {
        "num_questions": len(questions),
        "num_samples_per_question": num_samples,
        "accuracy": accuracy,  # At least one correct answer
        "pass@1": pass_at_1,
        "pass@5": pass_at_5,
        "pass@8": pass_at_8,
        "avg_correct_per_question": avg_correct_per_question,
        "avg_rm_score_correct": avg_rm_correct,
        "avg_rm_score_incorrect": avg_rm_incorrect,
    }

    return metrics, results

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained GRPO models")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (full model or LoRA adapters)"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Base model name (needed for LoRA checkpoints)"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="data/test_gsm8k.jsonl",
        help="Path to test data (JSONL format)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=8,
        help="Number of samples per question for pass@k"
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        default=None,
        help="Maximum number of questions to evaluate (for quick tests)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save detailed results (JSON)"
    )
    parser.add_argument(
        "--no_rm",
        action="store_true",
        help="Skip reward model evaluation (faster)"
    )

    args = parser.parse_args()

    # Check if test data exists
    test_path = Path(args.test_data)
    if not test_path.exists():
        print(f"Error: Test data not found at {test_path}")
        print("\nTo download GSM8K test set, run:")
        print("  python -c \"from datasets import load_dataset; ds = load_dataset('openai/gsm8k', 'main'); ds['test'].to_json('data/test_gsm8k.jsonl')\"")
        print("\nOr use a portion of train data for quick evaluation:")
        print(f"  python {__file__} --checkpoint {args.checkpoint} --test_data data/train_gsm8k.jsonl --max_questions 100")
        return

    # Load model
    model, tokenizer = load_model(args.checkpoint, base_model=args.base_model)

    # Load test data
    test_data = read_jsonl(args.test_data)
    print(f"Loaded {len(test_data)} test examples")

    # Evaluate
    metrics, results = evaluate(
        model,
        tokenizer,
        test_data,
        num_samples=args.num_samples,
        use_reward_model=not args.no_rm,
        max_samples=args.max_questions,
    )

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test Data: {args.test_data}")
    print(f"Questions Evaluated: {metrics['num_questions']}")
    print(f"Samples per Question: {metrics['num_samples_per_question']}")
    print("-"*60)
    print(f"Accuracy (e1 correct): {metrics['accuracy']*100:.2f}%")
    print(f"Pass@1: {metrics['pass@1']*100:.2f}%")
    if args.num_samples >= 5:
        print(f"Pass@5: {metrics['pass@5']*100:.2f}%")
    if args.num_samples >= 8:
        print(f"Pass@8: {metrics['pass@8']*100:.2f}%")
    print(f"Avg Correct/Question: {metrics['avg_correct_per_question']:.2f}")

    if not args.no_rm:
        print("-"*60)
        print(f"Avg RM Score (Correct): {metrics['avg_rm_score_correct']:.3f}")
        print(f"Avg RM Score (Incorrect): {metrics['avg_rm_score_incorrect']:.3f}")
    print("="*60)

    # Save detailed results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w") as f:
            json.dump({
                "metrics": metrics,
                "detailed_results": results,
                "config": vars(args),
            }, f, indent=2)

        print(f"\nDetailed results saved to: {output_path}")

if __name__ == "__main__":
    main()
