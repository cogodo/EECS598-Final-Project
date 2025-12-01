# GRPO Training with TinyLlama

This directory contains the implementation of Group Relative Policy Optimization (GRPO) training for TinyLlama-1.1B-Chat with integrated reward modeling and math verification.

## Overview

The training script has been refactored to:
- Use TinyLlama-1.1B-Chat-v1.0 instead of Llama-3.2-1B
- Integrate AceRewardModel for scoring responses
- Integrate MathVerifier for correctness verification
- Combine both signals using a hybrid reward function
- Use dummy data for testing

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
pip install verl  # For math verification
```

2. Ensure you have CUDA available for GPU training

## File Structure

- `train.py` - Main GRPO training script with TinyLlama integration
- `loss.py` - GRPO loss implementation
- `replay_buffer.py` - Experience replay buffer
- `data/dummy_math_tasks.jsonl` - Dummy test data (16 simple math problems)
- `data/math_tasks.jsonl` - Original math tasks (not used by default)

## Key Changes from Original

### Model
- Changed from `meta-llama/Llama-3.2-1B-Instruct` to `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Updated system prompt for TinyLlama chat format
- Removed `attn_implementation="flash_attention_2"` (not required for TinyLlama)

### Reward System
- **Hybrid Reward Function**: Combines two signals:
  1. **Math Verifier** (verl): Binary correctness check (0 or 1)
  2. **Reward Model** (AceRewardModel): Quality/reasoning assessment
  
- **Formula**: 
  - If correct: `(1 - β) + 2β * normalized_rm_score`
  - If incorrect: `-α + 2α * normalized_rm_score`
  - Default: α=0.5, β=0.5

### Training Parameters
- `group_size`: 8 (reduced from 12)
- `rollouts_per_step`: 8 (reduced from 32)
- `train_batch_size`: 8 (reduced from 16)
- `max_length`: 512 (reduced from 1024)
- `temperature`: 0.8 (reduced from 1.0)
- `checkpoint_interval`: 5 (reduced from 20)

### Data
- Uses `data/dummy_math_tasks.jsonl` by default
- Contains 16 simple math problems for testing
- Format: `{"question": "...", "answer": "..."}`

## Usage

Run training with default parameters:
```bash
cd tiny-grpo
python train.py
```

The script will:
1. Load TinyLlama model and tokenizer
2. Initialize reward model (AceRewardModel) and verifier (MathVerifier)
3. Run warmup to calibrate reward model bounds
4. Train using GRPO with hybrid rewards
5. Save checkpoints to `./output/`

## Monitoring

- Set `wandb_project = "your_project_name"` in `main()` to enable W&B logging
- Checkpoints saved every 5 steps by default
- Prints reward statistics during training

## Configuration

Edit `main()` function in `train.py` to adjust:
- `alpha`, `beta`: Hybrid reward weights
- `group_size`: Number of rollouts per question
- `lr`: Learning rate (default 5e-6)
- `kl_weight`: KL divergence penalty weight
- `clip_eps`: PPO clipping epsilon

## Notes

- The reward model (AceRewardModel) requires significant GPU memory
- Consider CPU offloading for the reward model if needed
- Dummy data is intentionally simple for testing the pipeline
- Replace with real math dataset for actual training
