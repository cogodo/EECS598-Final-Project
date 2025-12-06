        reward_prompt = torch.zeros(len(prompts))
        # --- Training Loop ---
        for k, batch in enumerate(prompt_loader):
            print(f"\n=== Step {k} ===")
            replay_buffer.clear()
            

            # 1. Batched Rollout Phase
            tasks = list(batch["question"])
            oracle_answers = [
                ans.split("####")[-1].strip() if "####" in ans else ans
                for ans in batch["answer"]
            ]

            sequence_ids, returns, action_mask, completions = rollout_batch(
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
