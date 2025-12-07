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
from utils import combine_hybrid_score, get_final_reward
from loss import approx_kl_divergence, GRPOLoss
from replay_buffer import ReplayBuffer, Experience, join_experience_batch

SYSTEM_PROMPT = """You are a helpful math assistant. Please solve the problem step by step, showing your reasoning clearly. 
Once you have solved the problem, provide your final numerical answer wrapped in <answer> tags, like this: <answer>number</answer>"""

SIGMA_BAR_LIST = [] # running values of sigma us - the stdev of rm scores

epochs = 20

def load_model(
    model_name_or_path: str,
    trust_remote_code: bool = False,
    bf16: bool = True,
    device_map=None,
) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # ← ADD THIS LINE

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

def read_prompts(file_name: str, predicate: Optional[Callable] = None, max_rows: Optional[int] = 200) -> List:
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
    # gen_config = GenerationConfig(
    #     do_sample=True, top_p=top_p, temperature=temperature, 
    #     max_length=max_length, pad_token_id=tokenizer.eos_token_id
    # )
    gen_config = GenerationConfig(
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        max_new_tokens=max_length,
        pad_token_id=tokenizer.eos_token_id,
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
    sigma_u = 0.0 # std dev of reward model scores across candidates
    t_rm_start = time.time()
    try:
        rm_scores_list = reward_model.compute_batch_reward(task, completions)
        if len(rm_scores_list) != len(completions):
            print(f"Error: RM score mismatch. Expected {len(completions)}, got {len(rm_scores_list)}.")
            rm_scores_list = [0.0] * len(completions)
    except Exception as e:
        print(f"RM Failed: {e}")
        rm_scores_list = [0.0] * len(completions)

    # standard deviation of reward model
    sigma_u = torch.std(torch.tensor(rm_scores_list))
    SIGMA_BAR_LIST.append(sigma_u)
    rm_time = time.time() - t_rm_start

    # 6. Verify and Combine Scores
    returns = torch.zeros(num_rollouts, 1, dtype=torch.float)
    verifier_returns = torch.zeros(num_rollouts, 1, dtype=torch.float)


    t_verify = 0
    

    for i, completion in enumerate(completions):
        t_v_start = time.time()
        verl_score = math_verifier.verify(task, completion, oracle_answer)["reward"]
        t_verify += (time.time() - t_v_start)
        
        # get r_hat according to verifier-rm blending
        r_hat = combine_hybrid_score(
            verl_score, rm_scores_list[i], min_rm, max_rm, eps, alpha, beta
        )

        # get final reward with variance aware reweighting
        hybrid_reward = get_final_reward(r_hat, sigma_bar=torch.stack(SIGMA_BAR_LIST).mean(), sigma_u=sigma_u)

        returns[i] = hybrid_reward

        verifier_returns[i] = verl_score

    # print(f"[Timing] Gen: {gen_time:.2f}s | Batch RM: {rm_time:.3f}s | Verifier: {t_verify:.3f}s")
    
    return sequence_ids, returns.to(sequence_ids.device), action_mask, completions, verifier_returns

@torch.no_grad()
def rollout_batch(
    model,
    tokenizer,
    tasks: List[str],
    oracle_answers: List[str],
    num_rollouts: int,
    reward_model: AceRewardModel,
    math_verifier: MathVerifier,
    min_rm: float,
    max_rm: float,
    alpha: float,
    beta: float,
    eps: float,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device="cuda",
):
    """Batched rollout for an entire mini-batch of prompts."""

    B = len(tasks)                         # batch_size (# of questions)
    K = num_rollouts                       # group_size
    total = B * K                          # total completions

    # 1. Build chat prompts for all tasks
    chat_prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for task in tasks
    ]

    # 2. Tokenize once for the entire batch
    enc = tokenizer(chat_prompts, return_tensors="pt", padding=True).to(device)

    input_ids = enc["input_ids"]                 # [B, L]
    attn_mask = enc["attention_mask"]

    # 3. Repeat each question K times → [B*K, L]
    input_ids = input_ids.repeat_interleave(K, dim=0)
    attn_mask = attn_mask.repeat_interleave(K, dim=0)

    # 4. Generate all completions in ONE CALL
    gen_config = GenerationConfig(
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        generation_config=gen_config,
    )

    # 5. Decode all completions
    completions = tokenizer.batch_decode(
        generated[:, input_ids.shape[1]:],
        skip_special_tokens=True,
    )

    # 6. Build masks
    action_mask = torch.zeros_like(generated, dtype=torch.bool)
    action_mask[:, enc["input_ids"].shape[1]:] = True
    action_mask[generated == tokenizer.eos_token_id] = False
    action_mask = action_mask[:, 1:]

    # 7. Compute rewards (still per question)
    all_returns = []
    all_verifier_returns = []
    idx = 0

    for task, oracle in zip(tasks, oracle_answers):
        each_completions = completions[idx: idx + K]

        # reward model
        rm_scores = reward_model.compute_batch_reward(task, each_completions)
        rm_scores_t = torch.tensor(rm_scores, dtype=torch.float32, device=device)

        sigma_u = rm_scores_t.std()
        SIGMA_BAR_LIST.append(sigma_u)
        sigma_bar = torch.stack(SIGMA_BAR_LIST).mean()

        returns = torch.zeros(K, 1, dtype=torch.float32, device=device)
        verifier_returns = torch.zeros(K, 1, dtype=torch.float32, device=device)


        for i, comp in enumerate(each_completions):
            verl_score = math_verifier.verify(task, comp, oracle)["reward"]

            r_hat = combine_hybrid_score(
                verl_score, rm_scores[i], min_rm, max_rm, eps, alpha, beta
            )

            hybrid = get_final_reward(r_hat, sigma_bar=sigma_bar, sigma_u=sigma_u)
            returns[i] = hybrid

            verifier_returns[i] = verl_score


        all_returns.append(returns)
        all_verifier_returns.append(verifier_returns)

        idx += K

    # stack into [B*K, 1]
    all_returns = torch.cat(all_returns, dim=0)
    all_verifier_returns = torch.cat(all_verifier_returns, dim=0)


    return generated, all_returns, action_mask, completions, all_verifier_returns



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
        "lr": 1e-5,
        "group_size": 8,
        "rollouts_per_step": 8,
        "max_norm": 1.0,
        "alpha": 0.5, "beta": 0.5, "eps": 0.01,
        "min_rm": -7.0, "max_rm": 7.0, # Pre-calibrated bounds
        "enable_profiling": True,
        "max_length": 200
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
    # adjust max_rows for training size
    prompts = read_prompts("data/train.jsonl", predicate=lambda x: len(x["question"]) < 512, max_rows=500)
    print(f"Loaded {len(prompts)} prompts")
    prompt_loader = DataLoader(prompts, batch_size=config["rollouts_per_step"], shuffle=True, drop_last=True)
    
    test_prompts = read_prompts("data/test.jsonl", predicate=lambda x: len(x["question"]) < 512, max_rows=10)
    print(f"Loaded {len(test_prompts)} prompts")
    test_prompt_loader = DataLoader(test_prompts, batch_size=config["rollouts_per_step"], shuffle=True, drop_last=False)

    replay_buffer = ReplayBuffer()
    objective = GRPOLoss(clip_eps=0.2, kl_weight=0.01)

    # --- Warmup to determine reward model bounds --- #
    print("Running warmup to determine reward model bounds...")
    min_rm = float('inf')
    max_rm = float('-inf')

    temperature = 1.0

    
    with torch.no_grad():
        for i, prompt in enumerate(prompts[:min(20, len(prompts))]):
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
            output = model.generate(**model_inputs, max_length=config["max_length"], temperature=temperature, do_sample=True)
            completion = tokenizer.decode(output[0, model_inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            
            try:
                rm_outputs = reward_model.compute_reward(q, completion)
                from_answer = reward_model.compute_reward(q, a)

                rm_score = rm_outputs[0]
                answer_score = from_answer[0]

                min_rm = min(min_rm, rm_score)
                min_rm = min(min_rm, answer_score)

                max_rm = max(max_rm, rm_score)
                max_rm = max(max_rm, answer_score)

            except Exception as e:
                print(f"Warning during warmup: {e}")
    
    print(f"RM bounds: min={min_rm:.4f}, max={max_rm:.4f}")
    
    config["min_rm"] = min_rm
    config["max_rm"] = max_rm


    print("\n ----- BEGIN TRAINING ------ \n")


    train_rewards = torch.zeros(epochs)
    train_rewards_std = torch.zeros(epochs)

    train_verifier = torch.zeros(epochs)

    test_rewards = torch.zeros(epochs)
    test_rewards_std = torch.zeros(epochs)

    test_verifier = torch.zeros(epochs)

    curr_step_losses_epoch = torch.zeros(epochs)
    curr_step_KL_epoch = torch.zeros(epochs)

    for e in range(epochs):

        reward_prompt = torch.zeros(len(prompts))
        verifer_reward_prompt = torch.zeros(len(prompts))
        # --- Training Loop ---
        for k, batch in enumerate(prompt_loader):
            # print(f"\n=== Step {k} ===")
            replay_buffer.clear()
            

            # 1. Batched Rollout Phase
            tasks = list(batch["question"])
            oracle_answers = [
                ans.split("####")[-1].strip() if "####" in ans else ans
                for ans in batch["answer"]
            ]

            sequence_ids, returns, action_mask, completions, verifer_returns = rollout_batch(
                model,
                tokenizer,
                tasks,
                oracle_answers,
                config["group_size"],
                reward_model,
                math_verifier,
                config["min_rm"],
                config["max_rm"],
                config["alpha"],
                config["beta"],
                config["eps"],
                max_new_tokens=config["max_length"],
                device=device,
            )

            # 2. Experience Creation (batched)
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
                    kl=approx_kl_divergence(log_probs, log_probs_ref, action_mask),
                )

            replay_buffer.clear()
            replay_buffer.append(exp.to("cpu"))


            # 3. Optimization Phase
            train_loader = DataLoader(replay_buffer, batch_size=config["train_batch_size"], shuffle=True, collate_fn=join_experience_batch)
            
            model.train()
            curr_step_losses = []
            curr_step_KLs = []

            optim_per_step = 1

            # optimization steps per prompt
            for _ in range(optim_per_step): 
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
                        # unnecesart print
                        # print(f"Loss: {loss.item():.4f}, KL: {kl.item():.4f}")
                        curr_step_losses.append(loss.item())
                        curr_step_KLs.append(kl.item())
                    
                        if _ == 0:
                            curr_step_losses_epoch[e] = loss.item()
                            curr_step_KL_epoch[e] = kl.item()

                    else:
                        print("Skipping non-finite loss")

            reward_prompt[k] = returns.mean()
            verifer_reward_prompt[k] = verifer_returns.max()



        test_reward_prompt = torch.zeros(len(test_prompts))
        test_verifier_reward_prompt = torch.zeros(len(test_prompts))
        # --- testing_loop Loop ---
        for k, batch in enumerate(test_prompt_loader):

            # 1. Batched Rollout Phase
            tasks = list(batch["question"])
            oracle_answers = [
                ans.split("####")[-1].strip() if "####" in ans else ans
                for ans in batch["answer"]
            ]

            sequence_ids, returns, action_mask, completions, verifer_returns = rollout_batch(
                model,
                tokenizer,
                tasks,
                oracle_answers,
                config["group_size"],
                reward_model,
                math_verifier,
                config["min_rm"],
                config["max_rm"],
                config["alpha"],
                config["beta"],
                config["eps"],
                max_new_tokens=config["max_length"],
                device=device,
            )


            test_reward_prompt[k] = returns.mean()
            test_verifier_reward_prompt[k] = verifer_returns.max()

        train_rewards[e] = reward_prompt.mean()
        train_rewards_std[e] = reward_prompt.std()
        train_verifier[e] = verifer_reward_prompt.mean()

        test_rewards[e] = test_reward_prompt.mean()
        test_rewards_std[e] = test_reward_prompt.std()
        test_verifier[e] = test_verifier_reward_prompt.mean()


        print(f'Epoch: {e}, Average Train Reward: {train_rewards[e]}, Average Train STD: {train_rewards_std[e]}, train_verifier: {train_verifier[e]}, Average Test Reward: {test_rewards[e]}, Avergre Test Reward STD: {test_rewards_std[e]}, test_verifier: {test_verifier[e]}, GRPO Loss: {curr_step_losses_epoch[e]}, KL Divergence: {curr_step_KL_epoch[e]}')

        # 4. Checkpointing
        if (e + 1) % 20 == 0:
            model.save_pretrained(config["checkpoint_path"] / f"step_{e}")


    History = {
        "train_rewards": train_rewards,
        "train_rewards_std": train_rewards_std,
        "train_verifier": train_verifier,
        "test_rewards": test_rewards,
        "test_rewards_std": test_rewards_std,
        "train_verifier": train_verifier,
        "train_verifier": train_verifier,
        "test_verifier": test_verifier,
        "curr_step_losses": curr_step_losses_epoch,
        "KL_Divergence": curr_step_KL_epoch
    }


    torch.save(History, "Reward_history.pt")



if __name__ == "__main__":
    main()
