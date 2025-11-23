import torch
from torch import nn
import yaml
import os

from model import TinyLlama
from reward_model import AceRewardModel
from math_verifier import MathVerifier
from utils import combine_hybrid_score
# import all other shit


class Trainer:
    def __init__(self, cfg_pth='configs/config.yaml'):
        # initialize all the parts: RM, verl, TinyLlama, hyperparams, etc.
        #how can we do this to not kill memory?
        with open(cfg_pth) as f:
            self.config = yaml.safe_load(f)
        self.ckpt_path = self.config.get('ckpt_path', 'checkpoints/ckpt.pt')

        self.base_model = TinyLlama()
        self.reward_model = AceRewardModel()
        self.verl_model = MathVerifier()
    
    def compute_reward(self, question, response, ground_truth, min_rm, max_rm):
        # put responses through RM and verl
        rm_score = self.reward_model(question, response)
        verl_score = self.verl_model(question, response, ground_truth)
        # put the outputs of these through the math function with alpha, beta
        combined = combine_hybrid_score(verl_score, rm_score, min_rm, max_rm, 
                                        self.config['eps'], self.config['alpha'], self.config['beta'])
        return combined

    def compute_advantages(self, rewards):
        mean_reward = sum(rewards) / len(rewards)
        return [r - mean_reward for r in rewards]
    
    def train(self):
        # train the model
        #load in the ckpt if it exists
        start_epoch = 0
        K = self.config['K']
        optimizer = torch.optim.Adam(self.base_model.parameters())

        if os.path.exists(self.ckpt_path):
            ckpt = torch.load(self.ckpt_path)
            start_epoch = ckpt['epoch'] + 1
            self.base_model.model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        
        self.base_model.load_model()
        train_data, val_data = self.base_model.prep_data()
        num_epochs = self.config['num_epochs']

        ckpt_every = num_epochs // 4

        # === Warmup pass to get RM bounds ===
        min_rm = float('inf')
        max_rm = float('-inf')

        self.base_model.eval()
        with torch.no_grad():
            for question, answer in train_data:
                response = self.base_model(question)
                rm_score = self.reward_model(response)
                if rm_score > max_rm:
                    max_rm = rm_score
                if rm_score < min_rm:
                    min_rm = rm_score
                    
        for epoch in range(start_epoch, num_epochs):
            self.base_model.train()
            # go through all data
            for question, ground_truth in train_data:
                # get responses
                with torch.no_grad():
                    responses = [self.base_model.generate(question) for _ in range(K)]
                    # Also compute rewards here — they don't need gradients
                    rewards = [self.compute_reward(question, resp, ground_truth, min_rm, max_rm) for resp in responses]
                    # compute advantages TODO update to variance-aware weighting
                    advantages = self.compute_advantages(rewards)
                
                optimizer.zero_grad()
                # calc loss + backprop
                for response, advantage in zip(responses, advantages):
                    log_prob = self.base_model.get_log_prob(question, response)
                    partial_loss = -log_prob * advantage / K
                    partial_loss.backward()

                optimizer.step()
                

            # checkpointing logic
            if epoch % ckpt_every == 0:
                self.base_model.eval()
                with torch.no_grad():
                    total_reward = 0
                    for question, answer in val_data:
                        response = self.base_model.generate(question)
                        reward = self.compute_reward(question, response, answer, min_rm, max_rm)
                        total_reward += reward
                    avg_reward = total_reward / len(val_data)

                    print(f"Epoch {epoch}, avg reward: {avg_reward}")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.base_model.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, self.ckpt_path)
                    print(f"Saved checkpoint at epoch {epoch}")