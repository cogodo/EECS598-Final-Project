"""Batch evaluation script for multiple checkpoints"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd

# Import from eval_model
sys.path.append(str(Path(__file__).parent))
from eval_model import load_model, read_jsonl, evaluate

def find_checkpoints(checkpoint_dir: Path, pattern: str = "step_*") -> list[Path]:
    """Find all checkpoint directories matching pattern"""
    checkpoints = sorted(checkpoint_dir.glob(pattern))
    return [cp for cp in checkpoints if cp.is_dir()]

def main():
    parser = argparse.ArgumentParser(description="Batch evaluate multiple checkpoints")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing checkpoints"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="step_*",
        help="Glob pattern for checkpoint directories (default: step_*)"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Base model name (for LoRA checkpoints)"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="data/test_gsm8k.jsonl",
        help="Path to test data"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=8,
        help="Number of samples per question"
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        default=None,
        help="Max questions to evaluate per checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--no_rm",
        action="store_true",
        help="Skip reward model evaluation"
    )

    args = parser.parse_args()

    # Setup
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find checkpoints
    checkpoints = find_checkpoints(checkpoint_dir, args.pattern)
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir} matching pattern '{args.pattern}'")
        return

    print(f"Found {len(checkpoints)} checkpoints:")
    for cp in checkpoints:
        print(f"  - {cp.name}")

    # Load test data once
    test_data = read_jsonl(args.test_data)
    print(f"\nLoaded {len(test_data)} test examples")
    if args.max_questions:
        test_data = test_data[:args.max_questions]
        print(f"Using first {len(test_data)} questions")

    # Evaluate each checkpoint
    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, checkpoint_path in enumerate(checkpoints):
        print(f"\n{'='*70}")
        print(f"Evaluating checkpoint {i+1}/{len(checkpoints)}: {checkpoint_path.name}")
        print(f"{'='*70}")

        try:
            # Load model
            model, tokenizer = load_model(
                str(checkpoint_path),
                base_model=args.base_model
            )

            # Evaluate
            metrics, detailed_results = evaluate(
                model,
                tokenizer,
                test_data,
                num_samples=args.num_samples,
                use_reward_model=not args.no_rm,
                max_samples=args.max_questions,
            )

            # Add checkpoint info to metrics
            metrics["checkpoint"] = checkpoint_path.name
            metrics["checkpoint_path"] = str(checkpoint_path)
            all_results.append(metrics)

            # Save individual checkpoint results
            checkpoint_output = output_dir / f"{checkpoint_path.name}_{timestamp}.json"
            with checkpoint_output.open("w") as f:
                json.dump({
                    "metrics": metrics,
                    "detailed_results": detailed_results,
                    "config": vars(args),
                }, f, indent=2)
            print(f"\nSaved to: {checkpoint_output}")

            # Print summary
            print(f"\n{'='*70}")
            print(f"RESULTS for {checkpoint_path.name}")
            print(f"{'='*70}")
            print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
            print(f"Pass@1: {metrics['pass@1']*100:.2f}%")
            if args.num_samples >= 5:
                print(f"Pass@5: {metrics['pass@5']*100:.2f}%")
            if args.num_samples >= 8:
                print(f"Pass@8: {metrics['pass@8']*100:.2f}%")
            print(f"{'='*70}\n")

            # Free memory
            del model
            del tokenizer
            import torch
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\nERROR evaluating {checkpoint_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save summary CSV
    if all_results:
        df = pd.DataFrame(all_results)
        summary_csv = output_dir / f"batch_eval_summary_{timestamp}.csv"
        df.to_csv(summary_csv, index=False)
        print(f"\n{'='*70}")
        print("BATCH EVALUATION COMPLETE")
        print(f"{'='*70}")
        print(f"Summary saved to: {summary_csv}")
        print(f"\nResults table:")
        print(df.to_string(index=False))

        # Also save as JSON
        summary_json = output_dir / f"batch_eval_summary_{timestamp}.json"
        with summary_json.open("w") as f:
            json.dump(all_results, f, indent=2)
    else:
        print("\nNo successful evaluations completed.")

if __name__ == "__main__":
    main()
