import numpy as np
import random
import gym
import torch

def set_global_seeds(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def set_global_log_levels(level):
    gym.logger.set_level(level)


def orthogonal_init(module, gain = 1.0):
    torch.nn.init.orthogonal_(module.weight.data, gain)
    torch.nn.init.constant_(module.bias.data, 0)

    return module