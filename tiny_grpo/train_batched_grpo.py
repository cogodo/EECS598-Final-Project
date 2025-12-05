import sys
import time
import json
import random
import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, List, Tuple
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.profiler import profile, ProfilerActivity, record_function
from transformers import AutoTokenizer, PreTrainedTokenizer, AutoModelForCausalLM, GenerationConfig

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from reward_model import AceRewardModel
from math_verifier import MathVerifier
from utils import combine_hybrid_score
from loss import approx_kl_divergence, GRPOLoss
from replay_buffer import ReplayBuffer, Experience, join_experience_batch

SYSTEM_PROMPT = """You are a helpful math assistant. Please solve the problem step by step, showing your reasoning clearly. 
Once you have solved the problem, provide your final numerical answer wrapped in <answer> tags, like this: <answer>number</answer>"""

def load_model(
    model_name_or_path: str,
    trust_remote_code: bool = False,
    bf16: bool = True,
    device_map=None,
) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        device_map=device_map,
    )
    model.config.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

def init_rng(seed: int):
    random.seed(seed)
    return torch.manual_seed(seed)

def read_jsonl(file_name: str | Path) -> Iterator:
    with Path(file_name).open(mode="r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def read_prompts(file_name: str, predicate: Optional[Callable] = None, max_rows: Optional[int] = None) -> List:
    rows = []
    for x in read_jsonl(file_name):
        if predicate is None or predicate(x):
            rows.append(x)
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows

@torch.no_grad()
def rollout(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    task: str,
    oracle_answer: str,
    num_rollouts: int,
    reward_model: AceRewardModel,
    math_verifier: MathVerifier,
    min_rm: float,
    max_rm: float,
    alpha: float,
    beta: float,
    eps: float,
    max_length: int = 512,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    
    model.eval()
    
    # 1. Prepare Inputs
    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": task}],
        tokenize=False, 
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([chat_prompt], return_tensors="pt", padding=True).to("cuda")
    input_ids = model_inputs["input_ids"].repeat(num_rollouts, 1)
    attention_mask = model_inputs["attention_mask"].repeat(num_rollouts, 1)

    # 2. Generate
    gen_config = GenerationConfig(
        do_sample=True, top_p=top_p, temperature=temperature, 
        max_length=max_length, pad_token_id=tokenizer.eos_token_id
    )
    
    t_gen_start = time.time()
    sequence_ids = model.generate(
        input_ids=input_ids, attention_mask=attention_mask, generation_config=gen_config
    )
    gen_time = time.time() - t_gen_start

    # 3. Decode
    completions = tokenizer.batch_decode(
        sequence_ids[:, input_ids.shape[1]:], skip_special_tokens=True
    )

    # --- Print Generated Completions ---
    # uncomment to view responses
    # print(f"\n--- Generated {len(completions)} Responses for: {task[:50]}... ---")
    # for i, c in enumerate(completions):
    #     print(f"[Response {i}]: {c}\n")
    # print("---------------------------------------------------")

    # 4. Create Mask (masking out padding)
    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    action_mask[:, input_ids.shape[1]:] = True
    action_mask[sequence_ids == tokenizer.eos_token_id] = False
    action_mask = action_mask[:, 1:]

    # 5. Compute Batch Rewards (AceRM)
    t_rm_start = time.time()
    try:
        rm_scores_list = reward_model.compute_batch_reward(task, completions)
        if len(rm_scores_list) != len(completions):
            print(f"Error: RM score mismatch. Expected {len(completions)}, got {len(rm_scores_list)}.")
            rm_scores_list = [0.0] * len(completions)
    except Exception as e:
        print(f"RM Failed: {e}")
        rm_scores_list = [0.0] * len(completions)
    rm_time = time.time() - t_rm_start

    # 6. Verify and Combine Scores
    returns = torch.zeros(num_rollouts, 1, dtype=torch.float)
    t_verify = 0
    
    for i, completion in enumerate(completions):
        t_v_start = time.time()
        verl_score = math_verifier.verify(task, completion, oracle_answer)["reward"]
        t_verify += (time.time() - t_v_start)
        
        hybrid_reward = combine_hybrid_score(
            verl_score, rm_scores_list[i], min_rm, max_rm, eps, alpha, beta
        )
        returns[i] = hybrid_reward

    print(f"[Timing] Gen: {gen_time:.2f}s | Batch RM: {rm_time:.3f}s | Verifier: {t_verify:.3f}s")
    
    return sequence_ids, returns.to(sequence_ids.device), action_mask, completions

def group_advantages(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (returns - returns.mean()) / (returns.std() + eps)

def sequences_log_probs(model, sequence_ids, attention_mask):
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(mask=(attention_mask == 0), value=1)
    
    output = model(
        input_ids=sequence_ids, attention_mask=attention_mask, 
        position_ids=position_ids, use_cache=False
    )
    
    # Select log probs for the tokens that were generated
    log_probs = F.log_softmax(output["logits"][:, :-1].to(torch.float32), dim=-1)
    return log_probs.gather(dim=-1, index=sequence_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

def main():
    # --- Configuration ---
    config = {
        "seed": 42,
        "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "checkpoint_path": Path("./output"),
        "train_batch_size": 8,
        "lr": 5e-6,
        "group_size": 8,
        "rollouts_per_step": 8,
        "max_norm": 1.0,
        "alpha": 0.5, "beta": 0.5, "eps": 0.01,
        "min_rm": -7.0, "max_rm": 7.0, # Pre-calibrated bounds
        "enable_profiling": True,
        "max_length": 512
    }
    
    init_rng(config["seed"])
    device = torch.device("cuda", 0)
    wandb.init(mode="disabled") # Set to "online" for tracking

    # --- Load Models ---
    print("Loading Models...")
    ref_model, _ = load_model(config["model_name"], device_map=device)
    model, tokenizer = load_model(config["model_name"], device_map=device)
    ref_model.eval()
    
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    reward_model = AceRewardModel()
    math_verifier = MathVerifier(method="flexible", correct_reward=1.0, format_reward=0.0)

    # --- Data Loading ---
    prompts = read_prompts("data/train.jsonl", predicate=lambda x: len(x["question"]) < 512, max_rows=64)
    print(f"Loaded {len(prompts)} prompts")
    prompt_loader = DataLoader(prompts, batch_size=config["rollouts_per_step"], shuffle=True, drop_last=True)
    
    replay_buffer = ReplayBuffer()
    objective = GRPOLoss(clip_eps=0.2, kl_weight=0.01)

    # --- Warmup to determine reward bounds --- #
    print("Running warmup to determine reward model bounds...")
    min_rm = float('inf')
    max_rm = float('-inf')
    
    #TEMP todo fix later
    min_rm = -7
    max_rm = 7

    max_length = 512

    with torch.no_grad():
        for i, prompt in enumerate(prompts[:min(10, len(prompts))]):
            q = prompt["question"]
            a = prompt["answer"]
            # Generate a single completion for RM calibration
            chat_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ]
            chat_prompt = tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs = tokenizer([chat_prompt], return_tensors="pt", padding=True).to(device)
            output = model.generate(**model_inputs, max_length=max_length, temperature=temperature, do_sample=True)
            completion = tokenizer.decode(output[0, model_inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            
            try:
                rm_outputs = reward_model.compute_reward(q, completion)
                rm_score = rm_outputs.logits[0][0].item()
                min_rm = min(min_rm, rm_score)
                max_rm = max(max_rm, rm_score)
            except Exception as e:
                print(f"Warning during warmup: {e}")
    
    print(f"RM bounds: min={min_rm:.4f}, max={max_rm:.4f}")

    



    # --- Training Loop ---
    for k, batch in enumerate(prompt_loader):
        print(f"\n=== Step {k} ===")
        replay_buffer.clear()
        
        # 1. Rollout Phase
        for q, a in zip(batch["question"], batch["answer"]):

            # GSM8K parsing. get value after "####" as oracle answer
            if "####" in a:
                oracle_answer = a.split("####")[-1].strip()
            else:
                oracle_answer = a # Fallback for dummy data
            

            sequence_ids, returns, action_mask, _ = rollout(
                model, tokenizer, q, oracle_answer, config["group_size"], reward_model, math_verifier,
                config["min_rm"], config["max_rm"], config["alpha"], config["beta"], config["eps"]
            )
            
            # 2. Experience Creation
            with torch.no_grad():
                att_mask = sequence_ids != tokenizer.eos_token_id
                log_probs = sequences_log_probs(model, sequence_ids, att_mask)
                log_probs_ref = sequences_log_probs(ref_model, sequence_ids, att_mask)
                
                exp = Experience(
                    sequences=sequence_ids,
                    action_log_probs=log_probs,
                    log_probs_ref=log_probs_ref,
                    returns=returns,
                    advantages=group_advantages(returns),
                    attention_mask=att_mask,
                    action_mask=action_mask,
                    kl=approx_kl_divergence(log_probs, log_probs_ref, action_mask)
                )
                replay_buffer.append(exp.to("cpu"))

        # 3. Optimization Phase
        train_loader = DataLoader(replay_buffer, batch_size=config["train_batch_size"], shuffle=True, collate_fn=join_experience_batch)
        
        model.train()
        for _ in range(1): # epochs per step
            for exp in train_loader:
                exp = exp.to(device)
                optimizer.zero_grad()
                
                curr_log_probs = sequences_log_probs(model, exp.sequences, exp.attention_mask)
                loss, kl = objective(curr_log_probs, exp)
                
                if loss.isfinite():
                    loss.backward()
                    clip_grad_norm_(model.parameters(), config["max_norm"])
                    optimizer.step()
                    wandb.log({"loss": loss.item(), "kl": kl.item()})
                    print(f"Loss: {loss.item():.4f}, KL: {kl.item():.4f}")
                else:
                    print("Skipping non-finite loss")

        # 4. Checkpointing
        if (k + 1) % 5 == 0:
            model.save_pretrained(config["checkpoint_path"] / f"step_{k}")

if __name__ == "__main__":
    main()
