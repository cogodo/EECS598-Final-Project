
def combine_hybrid_score(verl_score, rm_score, min_rm, max_rm, eps, alpha, beta):
    if verl_score == 1:
        return (1 - beta) + 2 * beta * ((rm_score - min_rm) / (max_rm - min_rm + eps))
    else:
        return -alpha + 2 * alpha * ((rm_score - min_rm) / (max_rm - min_rm + eps))
