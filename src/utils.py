import torch

def combine_hybrid_score(verl_score, rm_score, min_rm, max_rm, eps, alpha, beta):
    if verl_score == 1:
        return (1 - beta) + 2 * beta * ((rm_score - min_rm) / (max_rm - min_rm + eps))
    else:
        return -alpha + 2 * alpha * ((rm_score - min_rm) / (max_rm - min_rm + eps))
    
def get_final_reward(r_hat, sigma_bar, sigma_u):

    # these are hyperparameters in the paper
    w_min = 0.5
    w_max = 2.0
    k = 5

    w_difficulty = w_min+(w_max-w_min)*1/(1+torch.exp(-k*(sigma_u-sigma_bar)))

    r_final = w_difficulty*r_hat

    return r_final


def get_our_final_reward(r_hat, sigma_bar, sigma_u):

    # these are hyperparameters in the paper
    w_min = 0.5
    w_max = 2.0
    k = 5

    w_difficulty = w_min+(w_max-w_min)*torch.tanh(r_hat**3)

    r_final = w_difficulty*r_hat

    return r_final
