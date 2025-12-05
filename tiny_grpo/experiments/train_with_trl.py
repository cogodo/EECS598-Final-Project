"""
Industry Standard: GRPO Training using HuggingFace TRL

This is the "production-ready" way to do GRPO training.
TRL's GRPOTrainer handles all the complexity for you.

Source: https://huggingface.co/docs/trl/en/grpo_trainer
Tutorial: https://huggingface.co/learn/cookbook/en/fine_tuning_llm_grpo_trl
"""

# NOTE: This requires TRL to be installed
# pip install trl>=0.9.0

from pathlib import Path
import sys

# Check if TRL is available
try:
    from trl import GRPOConfig, GRPOTrainer
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig
    from datasets import load_dataset
except ImportError as e:
    print("❌ Missing dependencies!")
    print(f"Error: {e}")
    print("\nInstall with: pip install trl peft datasets")
    sys.exit(1)

# Add src to path for reward model
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
from reward_model import AceRewardModel
from math_verifier import MathVerifier

def prepare_reward_function():
    """Create reward function that combines RM + verifier"""
    reward_model = AceRewardModel()
    math_verifier = MathVerifier(method="flexible", correct_reward=1.0, format_reward=0.0)

    def reward_fn(prompts, completions, oracle_answers):
        """
        TRL expects a reward function that returns scores for each completion

        Args:
            prompts: List of question strings
            completions: List of model completions
            oracle_answers: List of ground truth answers

        Returns:
            List of reward scores
        """
        rewards = []
        for prompt, completion, oracle in zip(prompts, completions, oracle_answers):
            # Get verifier score
            verl_result = math_verifier.verify(prompt, completion, oracle)
            verl_score = verl_result["reward"]

            # Get reward model score
            try:
                rm_score = reward_model.compute_batch_reward(prompt, [completion])[0]
            except:
                rm_score = 0.0

            # Combine (simple average for now)
            reward = 0.5 * verl_score + 0.5 * ((rm_score + 7) / 14)  # Normalize RM to [0,1]
            rewards.append(reward)

        return rewards

    return reward_fn

def main():
    # Configuration
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    output_dir = Path("./output_trl")
    output_dir.mkdir(exist_ok=True)

    # LoRA Config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # GRPO Training Config
    config = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        num_generation_per_prompt=8,  # Like group_size
        max_completion_length=512,
        max_prompt_length=256,
        temperature=1.0,
        logging_steps=1,
        save_steps=5,
        bf16=True,
    )

    # Load model and tokenizer
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="bfloat16",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print("Loading dataset...")
    # You can load from your JSONL or use HF datasets
    # For now, this is a placeholder
    # dataset = load_dataset("json", data_files="data/train_gsm8k.jsonl")

    print("""
    ⚠️  This is a template showing how to use TRL's GRPOTrainer.

    To complete this:
    1. Install TRL: pip install trl>=0.9.0
    2. Load your GSM8K dataset
    3. Implement the reward function integration
    4. Configure the trainer

    Benefits of using TRL:
    - ✓ Production-tested code
    - ✓ Built-in optimizations
    - ✓ Maintained by HuggingFace
    - ✓ Used in real models (DeepSeek R1, etc.)

    Your custom implementation is great for learning!
    But for production, consider TRL's GRPOTrainer.

    Learn more:
    - https://huggingface.co/docs/trl/en/grpo_trainer
    - https://huggingface.co/learn/cookbook/en/fine_tuning_llm_grpo_trl
    """)

    # Create trainer
    # trainer = GRPOTrainer(
    #     model=model,
    #     args=config,
    #     tokenizer=tokenizer,
    #     train_dataset=dataset,
    #     reward_function=prepare_reward_function(),
    #     peft_config=lora_config,
    # )

    # Train
    # trainer.train()

if __name__ == "__main__":
    main()
