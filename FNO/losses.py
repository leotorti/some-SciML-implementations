import torch


def time_weighted_loss(output,target,time,time_weight=1.0):
    if time_weight<0:
        raise ValueError('time_weight must be nonnegative.')
    weight=(1.0+time_weight*time).unsqueeze(1)
    return(weight*(output-target)**2).mean()


def relative_l2_per_sample(output,target):
    denominator=torch.linalg.vector_norm(target,dim=-1).clamp_min(1e-12)
    return torch.linalg.vector_norm(output-target,dim=-1)/denominator
