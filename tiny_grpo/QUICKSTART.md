# Quick Start Guide - GRPO Training with TinyLlama

## What Was Done

✅ Refactored `train.py` to use **TinyLlama-1.1B-Chat-v1.0**
✅ Integrated **AceRewardModel** from `src/reward_model.py`
✅ Integrated **MathVerifier** from `src/math_verifier.py`
✅ Implemented **Hybrid Reward System** using `combine_hybrid_score`
✅ Created **dummy test data** (16 math problems)
✅ Adjusted hyperparameters for TinyLlama
✅ Added warmup phase for reward model calibration

## Quick Start

### 1. Install Dependencies
```bash
cd tiny-grpo
pip install torch transformers accelerate wandb verl
```

### 2. Test Setup (Optional but Recommended)
```bash
python test_setup.py
```
This will verify:
- All imports work correctly
- Data file exists
- Math verifier functions
- Hybrid score computation works
- (Optional) Model loading test
- (Optional) Reward model test

### 3. Run Training
```bash
# Run with default sigmoid-based reward scorer
python train.py

# Or run with tanh-based reward scorer
python train.py --reward-scorer tanh
```

## What to Expect

### Training Flow:
1. **Loading** (~2-5 min)
   - Downloads TinyLlama if not cached
   - Loads reward model (if GPU available)
   - Initializes math verifier

2. **Warmup** (~1-2 min)
   - Generates 10 samples
   - Calibrates reward model bounds (min_rm, max_rm)

3. **Training Loop**
   - Each step:
     - Generates 8 rollouts per question
     - Computes hybrid rewards (verifier + RM)
     - Updates policy using GRPO
     - Prints reward statistics
   - Checkpoints every 5 steps in `./output/`

### Console Output Example:
```
Loading TinyLlama model...
Loading reward model and math verifier...
found 16 matching prompts
Running warmup to determine reward model bounds...
RM bounds: min=-2.3456, max=1.2345
rollout q='What is 5 + 3?', a='8', returns=6.40, replay_buffer_size=8...
returns of step 0: 51.2000
0: kl=0.0234, grad_norm=1.2345
```

## Configuration

### Command Line Arguments

```bash
python train.py --reward-scorer {sigmoid|tanh}
```

**Reward Scorer Options:**
- `sigmoid` (default): Uses sigmoid-based difficulty weighting (original `get_final_reward`)
- `tanh`: Uses tanh-based difficulty weighting (experimental `get_our_final_reward`)

The reward scorer controls how the final reward is computed based on difficulty estimates:
- **Sigmoid scorer**: `w = w_min + (w_max - w_min) / (1 + exp(-k*(σ_u - σ_bar)))`
- **Tanh scorer**: `w = w_min + (w_max - w_min) * tanh(r_hat³)`

### Code Configuration

Edit `main()` in `train.py` to customize:

```python
# Model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Training
group_size = 8              # Rollouts per question
rollouts_per_step = 8       # Questions per batch
train_batch_size = 8        # Training batch size
lr = 5e-6                   # Learning rate
max_length = 512            # Max sequence length
temperature = 0.8           # Sampling temperature

# Hybrid Reward
alpha = 0.5                 # Weight for incorrect answers
beta = 0.5                  # Weight for correct answers
kl_weight = 0.01           # KL penalty weight

# Monitoring
checkpoint_interval = 5     # Save every N steps
wandb_project = None       # Set to enable W&B logging
```

## Memory Requirements

| Component | Memory | Device |
|-----------|--------|--------|
| TinyLlama | ~2GB | GPU |
| AceRewardModel | ~14GB | GPU/CPU |
| Training | ~2GB | GPU |
| **Total** | **~18GB** | **GPU** |

**Tip:** If you have limited GPU memory, modify `reward_model.py` to use `device_map="cpu"` for the reward model.

## Data Format

Dummy data (`data/dummy_math_tasks.jsonl`):
```json
{"question": "What is 5 + 3?", "answer": "8"}
{"question": "What is 12 - 7?", "answer": "5"}
...
```

To use your own data:
1. Create JSONL file with same format
2. Update data path in `main()`:
   ```python
   prompts = read_prompts("data/your_data.jsonl", ...)
   ```

## Troubleshooting

### Issue: Import errors
**Solution:** Install dependencies: `pip install torch transformers verl`

### Issue: CUDA out of memory
**Solution:** Reduce batch sizes or offload reward model to CPU

### Issue: Reward model fails
**Solution:** The code handles this gracefully with try/except, uses default score of 0.0

### Issue: verl not found
**Solution:** Install with `pip install verl`

## Files Created

- `train.py` - Refactored training script ✓
- `data/dummy_math_tasks.jsonl` - Test data ✓
- `test_setup.py` - Verification script ✓
- `setup.sh` - Automated setup script ✓
- `README_REFACTORED.md` - Detailed documentation ✓
- `REFACTORING_SUMMARY.md` - Technical changes ✓
- `QUICKSTART.md` - This file ✓

## Next Steps

1. ✅ Run `test_setup.py` to verify everything works
2. ✅ Run `train.py` with dummy data to test the pipeline
3. 📝 Prepare your real math dataset
4. 🎛️ Tune hyperparameters (α, β, lr, group_size)
5. 📊 Enable W&B logging for monitoring
6. 🚀 Train on full dataset

## Support

For issues or questions:
- Check `README_REFACTORED.md` for detailed docs
- Review `REFACTORING_SUMMARY.md` for technical details
- Examine the code comments in `train.py`

---
**Ready to train!** 🚀
