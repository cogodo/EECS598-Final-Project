# GRPO Training Refactoring Summary

## Overview
Refactored `tiny-grpo/train.py` to integrate TinyLlama-1.1B-Chat with the reward model and math verifier from the `src/` directory.

## Changes Made

### 1. Model Changes
**Before:** `meta-llama/Llama-3.2-1B-Instruct`
**After:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

- Changed from `LlamaForCausalLM` to `AutoModelForCausalLM` for better compatibility
- Removed `attn_implementation="flash_attention_2"` (not needed for TinyLlama)
- Updated system prompt to be more suitable for TinyLlama's capabilities

### 2. Reward System Integration

#### Added Imports:
```python
from reward_model import AceRewardModel
from math_verifier import MathVerifier
from utils import combine_hybrid_score
```

#### Modified `rollout()` function signature:
Added parameters:
- `reward_model: AceRewardModel`
- `math_verifier: MathVerifier`
- `min_rm, max_rm: float` - RM score bounds for normalization
- `alpha, beta, eps: float` - Hybrid reward parameters

#### Hybrid Reward Computation:
Replaced simple answer matching with hybrid approach:
```python
# Get verifier score (0 or 1)
verifier_result = math_verifier.verify(task, completion, oracle_answer)
verl_score = verifier_result["reward"]

# Get reward model score
rm_outputs = reward_model.compute_reward(task, completion)
rm_score = rm_outputs.logits[0][0].item()

# Combine using hybrid function
hybrid_reward = combine_hybrid_score(
    verl_score, rm_score, min_rm, max_rm, eps, alpha, beta
)
```

### 3. Training Parameters

Reduced parameters for faster testing:
- `group_size`: 12 → 8
- `rollouts_per_step`: 32 → 8
- `train_batch_size`: 16 → 8
- `max_length`: 1024 → 512
- `temperature`: 1.0 → 0.8
- `checkpoint_interval`: 20 → 5

### 4. Main Function Updates

Added warmup phase to calibrate reward model bounds:
```python
# Warmup pass to get RM bounds
min_rm = float('inf')
max_rm = float('-inf')

# Generate samples and compute RM scores to find min/max
for prompt in prompts[:10]:
    # ... generate and score ...
    rm_score = reward_model.compute_reward(q, completion)
    min_rm = min(min_rm, rm_score)
    max_rm = max(max_rm, rm_score)
```

Initialize models:
```python
reward_model = AceRewardModel()
math_verifier = MathVerifier(method="flexible", correct_reward=1.0, format_reward=0.0)
```

Pass to rollout:
```python
sequence_ids, returns, action_mask, completions = rollout(
    model, tokenizer, q, a,
    num_rollouts=group_size,
    reward_model=reward_model,
    math_verifier=math_verifier,
    min_rm=min_rm, max_rm=max_rm,
    alpha=alpha, beta=beta, eps=eps,
    # ... other params
)
```

### 5. Data Updates

Created `data/dummy_math_tasks.jsonl` with 16 simple math problems:
- Basic arithmetic (addition, subtraction, multiplication, division)
- Word problems
- Format: `{"question": "...", "answer": "..."}`

Changed data loading:
```python
prompts = read_prompts(
    "data/dummy_math_tasks.jsonl",  # Changed from "data/math_tasks.jsonl"
    predicate=lambda x: len(x["question"]) < 256,  # Relaxed filters
    max_rows=64,  # Reduced from 64*1024
)
```

## Files Created

1. **`tiny-grpo/data/dummy_math_tasks.jsonl`**
   - 16 simple math problems for testing

2. **`tiny-grpo/README_REFACTORED.md`**
   - Comprehensive documentation
   - Setup instructions
   - Configuration guide
   - Key changes explained

3. **`tiny-grpo/test_setup.py`**
   - Verification script to test all components
   - Tests imports, data, verifier, hybrid scoring, model loading
   - Interactive prompts for expensive operations

## Architecture

```
User Question → TinyLlama → Multiple Responses → Scoring
                                                    ↓
                          Math Verifier (verl) ←→ Binary Score (0/1)
                                ↓
                          Reward Model (Ace) ←→ Quality Score
                                ↓
                          Hybrid Function ←→ Combined Reward
                                ↓
                          GRPO Loss ←→ Policy Update
```

## Hybrid Reward Formula

```python
if verl_score == 1:  # Correct answer
    reward = (1 - β) + 2β * ((rm_score - min_rm) / (max_rm - min_rm + eps))
else:  # Incorrect answer
    reward = -α + 2α * ((rm_score - min_rm) / (max_rm - min_rm + eps))
```

Where:
- α = 0.5 (weight for incorrect answers)
- β = 0.5 (weight for correct answers)
- eps = 0.01 (numerical stability)

## Memory Considerations

- **TinyLlama**: ~2GB GPU memory (bfloat16)
- **AceRewardModel**: ~14GB GPU memory (7B model)
- **Total**: ~16GB GPU memory recommended
- Consider CPU offloading for reward model if needed

## Testing

Run the test script to verify setup:
```bash
cd tiny-grpo
python test_setup.py
```

Then run training:
```bash
python train.py
```

## Next Steps

1. Install dependencies: `pip install -r requirements.txt verl`
2. Run test script to verify setup
3. Test with dummy data
4. Replace with real math dataset
5. Tune hyperparameters (α, β, group_size, lr)
6. Monitor with W&B for longer training runs
