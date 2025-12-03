from collections.abc import Callable
import json
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Iterator, Optional
import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.profiler import profile, ProfilerActivity, record_function
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)
from loss import approx_kl_divergence, GRPOLoss
from replay_buffer import ReplayBuffer, Experience, join_experience_batch

# Add parent directory to path to import from src
sys.path.append(str(Path(__file__).parent.parent / "src"))
from reward_model import AceRewardModel
from math_verifier import MathVerifier
from utils import combine_hybrid_score


def load_model(
    model_name_or_path: str,
    trust_remote_code: bool = False,
    bf16: bool = True,
    device_map=None,
) -> tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        device_map=device_map,
    )
    return model, tokenizer


# TinyLlama system prompt for math reasoning
system_prompt = """You are a helpful math assistant. Solve the given problem step by step and provide your final answer wrapped in <answer> tags, like this: <answer>your answer here</answer>"""


@torch.no_grad()
def rollout(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    task: str,
    oracle_answer: str,
    num_rollouts: int,
    reward_model: AceRewardModel,
    math_verifier: MathVerifier,
    min_rm: float = 0.0,
    max_rm: float = 1.0,
    alpha: float = 0.5,
    beta: float = 0.5,
    eps: float = 0.01,
    max_length: int = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:

    model.eval()

    # 1. format prompt
    chat_messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": task,
        },
    ]
    chat_prompt = tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer(
        [chat_prompt],
        return_tensors="pt",
        padding=True,
        padding_side="left",
        return_attention_mask=True,
    ).to("cuda")

    # duplicate prompt num_rollouts times
    model_inputs["attention_mask"] = model_inputs["attention_mask"].repeat(
        num_rollouts, 1
    )

    input_ids = model_inputs["input_ids"].repeat(num_rollouts, 1)
    model_inputs["input_ids"] = input_ids

    # 2. sample completions
    pad_token_id = tokenizer.eos_token_id
    generation_config = GenerationConfig(
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        max_length=max_length,
        pad_token_id=pad_token_id,
    )
    sequence_ids = model.generate(**model_inputs, generation_config=generation_config)
    completions = tokenizer.batch_decode(
        sequence_ids[:, input_ids.shape[1] :], skip_special_tokens=True
    )

    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    action_mask[:, input_ids.shape[1] :] = True
    action_mask[sequence_ids == pad_token_id] = False
    action_mask = action_mask[:, 1:]

    # 3. determine rewards using hybrid approach
    returns = torch.zeros(num_rollouts, 1, dtype=torch.float)
    for i, completion in enumerate(completions):
        # Get verifier score (binary: correct or not)
        verifier_result = math_verifier.verify(task, completion, oracle_answer)
        verl_score = verifier_result["reward"]
        
        # Get reward model score
        try:
            rm_outputs = reward_model.compute_reward(task, completion)
            rm_score = rm_outputs.logits[0][0].item()
        except Exception as e:
            print(f"Warning: Reward model failed with error: {e}, using default score")
            rm_score = 0.0
        
        # Combine scores using hybrid function
        hybrid_reward = combine_hybrid_score(
            verl_score, rm_score, min_rm, max_rm, eps, alpha, beta
        )
        
        returns[i] = hybrid_reward

    return sequence_ids, returns.to(sequence_ids.device), action_mask, completions


def init_rng(seed: int) -> torch.Generator:
    random.seed(seed)
    return torch.manual_seed(seed)


def group_advantages(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (returns - returns.mean()) / (returns.std() + eps)


def sequence_log_probs_from_logits(
    logits: torch.tensor, output_ids: torch.tensor
) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=-1)
    return log_prob.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)


def sequences_log_probs(
    model: AutoModelForCausalLM,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(mask=(attention_mask == 0), value=1)
    output = model.forward(
        input_ids=sequence_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    logits = output["logits"]
    log_probs = sequence_log_probs_from_logits(
        logits=logits[:, :-1].to(torch.float32),
        output_ids=sequence_ids[:, 1:],
    )
    return log_probs


def read_jsonl(file_name: str | Path) -> Iterator:
    file_path = Path(file_name)
    with file_path.open(mode="r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def read_prompts(
    file_name: str,
    predicate: Optional[Callable[[Any], bool]] = None,
    max_rows: Optional[int] = None,
) -> list:
    rows = []
    for x in read_jsonl(file_name):
        if predicate is None or predicate(x):
            rows.append(x)
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows


def main():
    seed = 42
    wandb_project = None  # "tiny_grpo"
    device_index = 0
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    checkpoint_path = Path("./output")
    checkpoint_interval = 5
    train_batch_size = 8
    lr = 5e-6
    kl_weight = 0.01
    clip_eps = 0.2

    group_size = 8
    rollouts_per_step = 8
    epochs_per_step = 1
    max_norm = 1.0  # gradient clipping

    # Hybrid reward parameters
    alpha = 0.5
    beta = 0.5
    eps = 0.01

    # rollout params
    max_length = 512
    top_p = 1.0
    temperature = 0.8

    # Profiling configuration
    enable_profiling = True
    profile_output_dir = Path("./profiling_results")
    profile_output_dir.mkdir(exist_ok=True)

    device = torch.device("cuda", device_index)
    cpu_device = torch.device("cpu")
    init_rng(seed)

    # Profiling timers
    profiling_stats = {
        "rollout_time": [],
        "experience_creation_time": [],
        "training_time": [],
        "total_step_time": [],
    }

    print("Loading TinyLlama model...")
    reference_model, _ = load_model(model_name, device_map=device)
    model, tokenizer = load_model(model_name, device_map=device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    reference_model.eval()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    pad_token_id = tokenizer.eos_token_id

    # Initialize reward model and verifier
    print("Loading reward model and math verifier...")
    reward_model = AceRewardModel()
    math_verifier = MathVerifier(method="flexible", correct_reward=1.0, format_reward=0.0)

    # Load prompts - use dummy data for testing
    prompts = read_prompts(
        "data/dummy_math_tasks.jsonl",
        predicate=lambda x: len(x["question"]) < 256,
        max_rows=64,
    )
    print(f"found {len(prompts)} matching prompts")
    prompt_loader = DataLoader(
        prompts,
        batch_size=rollouts_per_step,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
    )

    replay_buffer = ReplayBuffer()
    objective = GRPOLoss(clip_eps=clip_eps, kl_weight=kl_weight)

    if wandb_project is None:
        wandb.init(mode="disabled")
    else:
        wandb.init(project=wandb_project)

    # Warmup pass to get RM bounds
    print("Running warmup to determine reward model bounds...")
    min_rm = float('inf')
    max_rm = float('-inf')
    
    #TEMP todo fix later
    min_rm = -7
    max_rm = 7

    # with torch.no_grad():
    #     for i, prompt in enumerate(prompts[:min(10, len(prompts))]):
    #         q = prompt["question"]
    #         a = prompt["answer"]
    #         # Generate a single completion for RM calibration
    #         chat_messages = [
    #             {"role": "system", "content": system_prompt},
    #             {"role": "user", "content": q},
    #         ]
    #         chat_prompt = tokenizer.apply_chat_template(
    #             chat_messages, tokenize=False, add_generation_prompt=True
    #         )
    #         model_inputs = tokenizer([chat_prompt], return_tensors="pt", padding=True).to(device)
    #         output = model.generate(**model_inputs, max_length=max_length, temperature=temperature, do_sample=True)
    #         completion = tokenizer.decode(output[0, model_inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            
    #         try:
    #             rm_outputs = reward_model.compute_reward(q, completion)
    #             rm_score = rm_outputs.logits[0][0].item()
    #             min_rm = min(min_rm, rm_score)
    #             max_rm = max(max_rm, rm_score)
    #         except Exception as e:
    #             print(f"Warning during warmup: {e}")
    
    print(f"RM bounds: min={min_rm:.4f}, max={max_rm:.4f}")

    for k, prompt_batch in enumerate(prompt_loader):
        step_start_time = time.time()
        rollout_returns = []

        replay_buffer.clear()

        questions = prompt_batch["question"]
        answers = prompt_batch["answer"]

        with torch.no_grad():
            for q, a in zip(questions, answers):
                rollout_start = time.time()

                with record_function("rollout"):
                    sequence_ids, returns, action_mask, completions = rollout(
                        model,
                        tokenizer,
                        q,
                        a,
                        num_rollouts=group_size,
                        reward_model=reward_model,
                        math_verifier=math_verifier,
                        min_rm=min_rm,
                        max_rm=max_rm,
                        alpha=alpha,
                        beta=beta,
                        eps=eps,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p,
                    )

                rollout_time = time.time() - rollout_start
                profiling_stats["rollout_time"].append(rollout_time)

                print(
                    f"rollout q='{q}', a='{a}', returns={returns.sum().item():.2f}, rollout_time={rollout_time:.3f}s, replay_buffer_size={len(replay_buffer)}, sequence_ids={sequence_ids.shape}"
                )
                rollout_returns.append(returns.cpu())

                exp_start = time.time()

                with record_function("experience_creation"):
                    advantages = group_advantages(returns)
                    attention_mask = sequence_ids != pad_token_id

                    log_probs = sequences_log_probs(
                        model=model,
                        sequence_ids=sequence_ids,
                        attention_mask=attention_mask,
                    )
                    log_probs_ref = sequences_log_probs(
                        model=reference_model,
                        sequence_ids=sequence_ids,
                        attention_mask=attention_mask,
                    )
                    kl = approx_kl_divergence(
                        log_probs=log_probs,
                        log_probs_ref=log_probs_ref,
                        action_mask=action_mask,
                    )

                    experience = Experience(
                        sequences=sequence_ids,
                        action_log_probs=log_probs,
                        log_probs_ref=log_probs_ref,
                        returns=returns,
                        advantages=advantages,
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        kl=kl,
                    )
                    replay_buffer.append(experience.to(cpu_device))

                exp_time = time.time() - exp_start
                profiling_stats["experience_creation_time"].append(exp_time)

        torch.cuda.empty_cache()
        episode_return_sum = torch.stack(rollout_returns).sum()
        print(f"returns of step {k}: {episode_return_sum:.4f}")
        wandb.log({"returns": episode_return_sum})

        experience_sampler = DataLoader(
            replay_buffer,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=join_experience_batch,
        )

        training_start = time.time()

        # Enable PyTorch profiler for detailed analysis (optional, can be resource intensive)
        profiler_context = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) if enable_profiling and k == 0 else None

        if profiler_context:
            profiler_context.__enter__()

        for step_epoch in range(epochs_per_step):
            model.train()

            for exp in experience_sampler:
                exp: Experience

                exp = exp.to(device)

                optimizer.zero_grad()

                with record_function("forward_pass"):
                    log_probs = sequences_log_probs(
                        model, sequence_ids=exp.sequences, attention_mask=exp.attention_mask
                    )

                with record_function("loss_computation"):
                    loss, kl = objective(log_probs=log_probs, experience=exp)

                if not loss.isfinite():
                    print(f"Loss not finite, skipping backward, loss={loss}")
                    print(f"experience.advantages={experience.advantages}")
                    continue

                with record_function("backward_pass"):
                    loss.backward()
                    grad_norm = clip_grad_norm_(model.parameters(), max_norm=max_norm)

                print(f"{step_epoch}: kl={kl: .4f}, grad_norm={grad_norm: .4f}")
                wandb.log({"kl": kl, "grad_norm": grad_norm})

                with record_function("optimizer_step"):
                    optimizer.step()

        if profiler_context:
            profiler_context.__exit__(None, None, None)
            # Save profiler trace
            trace_path = profile_output_dir / f"trace_step_{k}.json"
            profiler_context.export_chrome_trace(str(trace_path))
            print(f"Profiler trace saved to {trace_path}")

            # Print summary
            print("\n=== Profiling Summary ===")
            print(profiler_context.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            print("========================\n")

        training_time = time.time() - training_start
        profiling_stats["training_time"].append(training_time)

        total_step_time = time.time() - step_start_time
        profiling_stats["total_step_time"].append(total_step_time)

        # Log timing stats
        print(f"\n=== Step {k} Timing ===")
        print(f"Total step time: {total_step_time:.3f}s")
        print(f"Avg rollout time: {sum(profiling_stats['rollout_time'][-len(questions):]) / len(questions):.3f}s")
        print(f"Avg experience creation time: {sum(profiling_stats['experience_creation_time'][-len(questions):]) / len(questions):.3f}s")
        print(f"Training time: {training_time:.3f}s")
        print("=====================\n")

        wandb.log({
            "step_time": total_step_time,
            "avg_rollout_time": sum(profiling_stats['rollout_time'][-len(questions):]) / len(questions),
            "training_time": training_time,
        })

        if (
            checkpoint_path is not None
            and checkpoint_interval is not None
            and (k + 1) % checkpoint_interval == 0
        ):
            model.save_pretrained(checkpoint_path / f"step_{k}")

    if checkpoint_path is not None:
        model.save_pretrained(checkpoint_path / f"step_{k}")

    # Save final profiling statistics
    print("\n=== Final Profiling Statistics ===")
    print(f"Total rollouts: {len(profiling_stats['rollout_time'])}")
    print(f"Avg rollout time: {sum(profiling_stats['rollout_time']) / len(profiling_stats['rollout_time']):.3f}s")
    print(f"Avg experience creation time: {sum(profiling_stats['experience_creation_time']) / len(profiling_stats['experience_creation_time']):.3f}s")
    print(f"Avg training time per step: {sum(profiling_stats['training_time']) / len(profiling_stats['training_time']):.3f}s")
    print(f"Avg total step time: {sum(profiling_stats['total_step_time']) / len(profiling_stats['total_step_time']):.3f}s")
    print("==================================\n")


if __name__ == "__main__":
    main()
