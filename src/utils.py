import torch
import torch.nn.functional as F

def combine_hybrid_score(verl_score, rm_score, min_rm, max_rm, eps, alpha, beta):
    if verl_score == 1:
        return (1 - beta) + 2 * beta * ((rm_score - min_rm) / (max_rm - min_rm + eps))
    else:
        return -alpha + 2 * alpha * ((rm_score - min_rm) / (max_rm - min_rm + eps))

def tanh_combine_reward(r_rm, r_ver,
                   a=8.0,                  # max magnitude
                   k_pos=0.04, p_pos=2.0,   # controls right-side steepness & curvature
                   k_neg=0.4, p_neg=2.0,   # controls left-side steepness & curvature
                   ):
    """
    Asymmetric tanh(x^p) with smooth reflection blending.
    Gives an S-shaped curve with adjustable steepness.
    """

    r_rm = torch.as_tensor(r_rm, dtype=torch.float32)
    r_ver = torch.as_tensor(r_ver, dtype=torch.float32)

    # Positive and negative regions independently transformed
    pos_val = F.relu(r_rm)     # max(0, r_rm)
    neg_val = F.relu(-r_rm)    # max(0, -r_rm)

    # tanh(x^p) shaping
    g_pos =  a * torch.tanh(k_pos * (pos_val ** p_pos))
    g_neg = -a * torch.tanh(k_neg * (neg_val ** p_neg))

    # Base asymmetric curve
    g = g_pos + g_neg

    # Reflected curve (swap pos/neg processing, negate)
    g_reflect_pos =  a * torch.tanh(k_pos * (F.relu(-r_rm) ** p_pos))
    g_reflect_neg = -a * torch.tanh(k_neg * (F.relu(r_rm)  ** p_neg))
    g_reflect = -(g_reflect_pos + g_reflect_neg)

    # Blend
    r_final = (1 - r_ver) * g + r_ver * g_reflect

    return r_final
    
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

