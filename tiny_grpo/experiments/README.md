# GRPO Model Evaluation Guide

This directory contains scripts for evaluating your GRPO-trained models with beautiful visualizations!

## 📁 File Structure

```
experiments/
├── eval_model.py          # Single checkpoint evaluation
├── batch_eval.py          # Batch evaluation of multiple checkpoints
├── visualize_results.py   # Create charts and plots
└── README.md             # This file
```

## 🚀 Quick Start

### 1. Download Test Data

First, download the GSM8K test set:

```bash
python -c "from datasets import load_dataset; ds = load_dataset('openai/gsm8k', 'main'); ds['test'].to_json('data/test_gsm8k.jsonl')"
```

### 2. Evaluate a Single Checkpoint

```bash
python tiny_grpo/experiments/eval_model.py \
  --checkpoint output/step_4 \
  --test_data data/test_gsm8k.jsonl \
  --num_samples 8
```

**For LoRA checkpoints:**
```bash
python tiny_grpo/experiments/eval_model.py \
  --checkpoint output/step_4 \
  --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --test_data data/test_gsm8k.jsonl \
  --num_samples 8
```

### 3. Batch Evaluate All Checkpoints

```bash
python tiny_grpo/experiments/batch_eval.py \
  --checkpoint_dir output \
  --test_data data/test_gsm8k.jsonl \
  --num_samples 8 \
  --output_dir results
```

This will:
- Find all `step_*` checkpoints in `output/`
- Evaluate each one
- Save individual results + summary CSV/JSON

### 4. Visualize Results

```bash
python tiny_grpo/experiments/visualize_results.py \
  --results results/batch_eval_summary_TIMESTAMP.csv \
  --output_dir plots
```

This generates:
- `pass_at_k.png` - Pass@k metrics over training
- `accuracy.png` - Model accuracy over training
- `avg_correct_per_question.png` - Average correct answers
- `reward_model_scores.png` - RM scores for correct vs incorrect
- `combined_metrics.png` - All metrics in one figure
- `summary_table.txt` - Formatted text table

## 📊 Metrics Explained

- **Accuracy**: % of questions with at least 1 correct answer out of k samples
- **Pass@k**: Probability that at least 1 out of k samples is correct
- **Avg Correct/Question**: Average number of correct answers per question (out of k)
- **RM Scores**: Reward model scores for correct vs incorrect responses

## 🔧 Advanced Options

### Quick Testing (100 questions, no RM)

```bash
python tiny_grpo/experiments/eval_model.py \
  --checkpoint output/step_4 \
  --test_data data/train_gsm8k.jsonl \
  --max_questions 100 \
  --no_rm
```

### Save Detailed Results

```bash
python tiny_grpo/experiments/eval_model.py \
  --checkpoint output/step_4 \
  --test_data data/test_gsm8k.jsonl \
  --num_samples 8 \
  --output results/step_4_detailed.json
```

### Custom Checkpoint Pattern

```bash
python tiny_grpo/experiments/batch_eval.py \
  --checkpoint_dir output \
  --pattern "step_[0-9]*" \
  --test_data data/test_gsm8k.jsonl
```

## 💡 Tips

1. **LoRA vs Full Model**:
   - LoRA checkpoints: Small (~10-50MB), contain only adapter weights
   - Full checkpoints: Large (~2GB), contain entire model
   - The scripts auto-detect checkpoint type by looking for `adapter_config.json`

2. **Where to put checkpoints**:
   - Training saves to `output/step_X/` by default
   - For LoRA: Each checkpoint will have `adapter_model.safetensors` + `adapter_config.json`
   - For full models: Each checkpoint will have `model.safetensors` + `config.json`

3. **Memory optimization**:
   - Use `--no_rm` to skip reward model (saves ~7GB VRAM)
   - Reduce `--num_samples` for faster evaluation
   - Use `--max_questions` for quick tests

4. **Parallel evaluation**:
   - The batch script evaluates sequentially to avoid OOM
   - It clears GPU cache between checkpoints

## 📈 Example Workflow

```bash
# 1. Train your model with LoRA
python tiny_grpo/train_batched_grpo.py  # Now has LoRA enabled by default!

# 2. Get test data
python -c "from datasets import load_dataset; ds = load_dataset('openai/gsm8k', 'main'); ds['test'].to_json('data/test_gsm8k.jsonl')"

# 3. Evaluate all checkpoints
python tiny_grpo/experiments/batch_eval.py \
  --checkpoint_dir output \
  --test_data data/test_gsm8k.jsonl \
  --num_samples 8 \
  --output_dir results

# 4. Visualize
python tiny_grpo/experiments/visualize_results.py \
  --results results/batch_eval_summary_*.csv \
  --output_dir plots

# 5. Check your results!
open plots/combined_metrics.png
```

## 🐛 Troubleshooting

**Problem**: "Import reward_model could not be resolved"
- **Solution**: This is just a Pylance warning. The code works fine at runtime because we add `src/` to `sys.path`.

**Problem**: "No checkpoints found"
- **Solution**: Check your `--checkpoint_dir` path. Make sure you have `step_*` directories there.

**Problem**: "CUDA out of memory"
- **Solution**: Use `--no_rm`, reduce `--num_samples`, or use `--max_questions 100` for testing.

**Problem**: "Test data not found"
- **Solution**: Run the download command above, or use `--test_data data/train_gsm8k.jsonl` with `--max_questions`.

## 🎨 Visualization Dependencies

Make sure you have these installed:

```bash
pip install matplotlib seaborn pandas
```

## 📝 Notes

- All scripts support both LoRA and full model checkpoints
- Results are timestamped to avoid overwriting
- CSV files can be opened in Excel/Google Sheets for further analysis
- Plots are saved at 300 DPI for publication quality
