"""Visualize evaluation results with charts"""
import json
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

def load_batch_results(summary_file: Path) -> pd.DataFrame:
    """Load batch evaluation summary"""
    if summary_file.suffix == '.csv':
        return pd.read_csv(summary_file)
    elif summary_file.suffix == '.json':
        with summary_file.open('r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported file format: {summary_file.suffix}")

def extract_step_number(checkpoint_name: str) -> int:
    """Extract step number from checkpoint name (e.g., 'step_4' -> 4)"""
    try:
        return int(checkpoint_name.split('_')[-1])
    except:
        return 0

def plot_pass_at_k(df: pd.DataFrame, output_path: Path):
    """Plot Pass@k metrics over training steps"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Extract step numbers
    df['step'] = df['checkpoint'].apply(extract_step_number)
    df = df.sort_values('step')

    # Plot pass@k curves
    metrics_to_plot = []
    if 'pass@1' in df.columns:
        metrics_to_plot.append(('pass@1', 'Pass@1', 'o-', '#2E86AB'))
    if 'pass@5' in df.columns:
        metrics_to_plot.append(('pass@5', 'Pass@5', 's-', '#A23B72'))
    if 'pass@8' in df.columns:
        metrics_to_plot.append(('pass@8', 'Pass@8', '^-', '#F18F01'))

    for metric, label, style, color in metrics_to_plot:
        ax.plot(df['step'], df[metric] * 100, style,
                label=label, linewidth=2.5, markersize=8, color=color)

    ax.set_xlabel('Training Step', fontsize=13, fontweight='bold')
    ax.set_ylabel('Pass@k (%)', fontsize=13, fontweight='bold')
    ax.set_title('Pass@k Performance vs Training Steps', fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_accuracy(df: pd.DataFrame, output_path: Path):
    """Plot accuracy over training steps"""
    fig, ax = plt.subplots(figsize=(12, 6))

    df['step'] = df['checkpoint'].apply(extract_step_number)
    df = df.sort_values('step')

    ax.plot(df['step'], df['accuracy'] * 100, 'o-',
            linewidth=2.5, markersize=10, color='#C73E1D', label='Accuracy')

    ax.set_xlabel('Training Step', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Model Accuracy (≥1 Correct) vs Training Steps',
                 fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_reward_model_scores(df: pd.DataFrame, output_path: Path):
    """Plot reward model scores for correct vs incorrect answers"""
    if 'avg_rm_score_correct' not in df.columns or 'avg_rm_score_incorrect' not in df.columns:
        print("Skipping reward model plot (scores not available)")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    df['step'] = df['checkpoint'].apply(extract_step_number)
    df = df.sort_values('step')

    ax.plot(df['step'], df['avg_rm_score_correct'], 'o-',
            linewidth=2.5, markersize=8, color='#06A77D', label='Correct Answers')
    ax.plot(df['step'], df['avg_rm_score_incorrect'], 's-',
            linewidth=2.5, markersize=8, color='#D62828', label='Incorrect Answers')

    ax.set_xlabel('Training Step', fontsize=13, fontweight='bold')
    ax.set_ylabel('Avg Reward Model Score', fontsize=13, fontweight='bold')
    ax.set_title('Reward Model Scores: Correct vs Incorrect Answers',
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_avg_correct_per_question(df: pd.DataFrame, output_path: Path):
    """Plot average number of correct answers per question"""
    fig, ax = plt.subplots(figsize=(12, 6))

    df['step'] = df['checkpoint'].apply(extract_step_number)
    df = df.sort_values('step')

    max_samples = df['num_samples_per_question'].iloc[0] if 'num_samples_per_question' in df.columns else 8

    ax.plot(df['step'], df['avg_correct_per_question'], 'o-',
            linewidth=2.5, markersize=10, color='#7209B7', label='Avg Correct/Question')

    ax.set_xlabel('Training Step', fontsize=13, fontweight='bold')
    ax.set_ylabel('Avg Correct Answers per Question', fontsize=13, fontweight='bold')
    ax.set_title('Average Correct Answers per Question vs Training Steps',
                 fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max_samples)
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_combined_metrics(df: pd.DataFrame, output_path: Path):
    """Plot all key metrics in a single figure with subplots"""
    df['step'] = df['checkpoint'].apply(extract_step_number)
    df = df.sort_values('step')

    has_rm = 'avg_rm_score_correct' in df.columns and 'avg_rm_score_incorrect' in df.columns
    n_plots = 4 if has_rm else 3

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    # Plot 1: Pass@k
    ax1 = fig.add_subplot(gs[0, 0])
    if 'pass@1' in df.columns:
        ax1.plot(df['step'], df['pass@1'] * 100, 'o-', label='Pass@1', linewidth=2, markersize=6)
    if 'pass@5' in df.columns:
        ax1.plot(df['step'], df['pass@5'] * 100, 's-', label='Pass@5', linewidth=2, markersize=6)
    if 'pass@8' in df.columns:
        ax1.plot(df['step'], df['pass@8'] * 100, '^-', label='Pass@8', linewidth=2, markersize=6)
    ax1.set_xlabel('Training Step', fontweight='bold')
    ax1.set_ylabel('Pass@k (%)', fontweight='bold')
    ax1.set_title('Pass@k Performance', fontweight='bold', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)

    # Plot 2: Accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df['step'], df['accuracy'] * 100, 'o-', linewidth=2, markersize=8, color='#C73E1D')
    ax2.set_xlabel('Training Step', fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontweight='bold')
    ax2.set_title('Model Accuracy', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    # Plot 3: Avg Correct per Question
    ax3 = fig.add_subplot(gs[1, 0])
    max_samples = df['num_samples_per_question'].iloc[0] if 'num_samples_per_question' in df.columns else 8
    ax3.plot(df['step'], df['avg_correct_per_question'], 'o-',
             linewidth=2, markersize=8, color='#7209B7')
    ax3.set_xlabel('Training Step', fontweight='bold')
    ax3.set_ylabel('Avg Correct/Question', fontweight='bold')
    ax3.set_title('Avg Correct Answers per Question', fontweight='bold', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, max_samples)

    # Plot 4: Reward Model Scores (if available)
    if has_rm:
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(df['step'], df['avg_rm_score_correct'], 'o-',
                 label='Correct', linewidth=2, markersize=6, color='#06A77D')
        ax4.plot(df['step'], df['avg_rm_score_incorrect'], 's-',
                 label='Incorrect', linewidth=2, markersize=6, color='#D62828')
        ax4.set_xlabel('Training Step', fontweight='bold')
        ax4.set_ylabel('Avg RM Score', fontweight='bold')
        ax4.set_title('Reward Model Scores', fontweight='bold', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    fig.suptitle('GRPO Training Evaluation Metrics', fontsize=16, fontweight='bold', y=0.995)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def create_summary_table(df: pd.DataFrame, output_path: Path):
    """Create a formatted summary table"""
    df['step'] = df['checkpoint'].apply(extract_step_number)
    df = df.sort_values('step')

    # Select columns
    columns = ['checkpoint', 'accuracy', 'pass@1']
    if 'pass@5' in df.columns:
        columns.append('pass@5')
    if 'pass@8' in df.columns:
        columns.append('pass@8')
    columns.append('avg_correct_per_question')

    table_df = df[columns].copy()

    # Format percentages
    for col in ['accuracy', 'pass@1', 'pass@5', 'pass@8']:
        if col in table_df.columns:
            table_df[col] = table_df[col].apply(lambda x: f"{x*100:.2f}%")

    table_df['avg_correct_per_question'] = table_df['avg_correct_per_question'].apply(lambda x: f"{x:.2f}")

    # Save as text
    with output_path.open('w') as f:
        f.write("="*80 + "\n")
        f.write("EVALUATION SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(table_df.to_string(index=False))
        f.write("\n\n" + "="*80 + "\n")

    print(f"Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize evaluation results")
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to batch evaluation summary (CSV or JSON)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots",
        help="Directory to save plots"
    )

    args = parser.parse_args()

    # Load results
    results_file = Path(args.results)
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        return

    print(f"Loading results from {results_file}...")
    df = load_batch_results(results_file)
    print(f"Loaded {len(df)} checkpoint results\n")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("Generating plots...\n")

    plot_pass_at_k(df, output_dir / "pass_at_k.png")
    plot_accuracy(df, output_dir / "accuracy.png")
    plot_avg_correct_per_question(df, output_dir / "avg_correct_per_question.png")
    plot_reward_model_scores(df, output_dir / "reward_model_scores.png")
    plot_combined_metrics(df, output_dir / "combined_metrics.png")
    create_summary_table(df, output_dir / "summary_table.txt")

    print(f"\nAll plots saved to {output_dir}/")
    print("\nGenerated files:")
    for file in sorted(output_dir.glob("*")):
        print(f"  - {file.name}")

if __name__ == "__main__":
    main()
