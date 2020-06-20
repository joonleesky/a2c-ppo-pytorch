from .misc_util import orthogonal_init

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


class CategoricalPolicy(nn.Module):
    def __init__(self, 
                 embedder,
                 action_size):
        """
        embedder: (torch.Tensor) model to extract the embedding for observation
        action_size: number of the categorical actions
        """ 
        super(CategoricalPolicy, self).__init__()
        self.embedder = embedder

        # small scale weight-initialization in policy enhances the stability
        self.fc_policy = orthogonal_init(nn.Linear(self.embedder.output_dim, action_size), gain = 0.01)
        self.fc_value = orthogonal_init(nn.Linear(self.embedder.output_dim, 1), gain=1.0)

    def pi(self, x):
        """
        p:(torch.distributions.Categorical)
        """
        embedding = self.embedder(x)
        embedding_flat = embedding.view(embedding.size(0), -1)
        logits = self.fc_policy(embedding_flat)
        log_probs = F.log_softmax(logits, dim = 1)
        p = Categorical(torch.exp(log_probs))
        return p
    
    def action(self, x):
        """
        action: (torch.Tensor) shape = (batch_size, )
        """
        p = self.pi(x)
        return p.sample()
    
    def log_prob(self, x, action):
        """
        log_prob:(torch.Tensor) shape = (batch_size, )
        """
        p = self.pi(x)
        return p.log_prob(action)
    
    def v(self, x):
        """
        v:(torch.Tensor) shape = (batch_size, )
        """
        embedding = self.embedder(x)
        embedding_flat = embedding.view(embedding.size(0), -1)
        v = self.fc_value(embedding_flat).reshape(-1)
        return v
    

class DiagGaussianPolicy(nn.Module):
    def __init__(self, 
                 embedder,
                 action_size,
                 action_range = (-1, 1)):
        """
        embedder: (torch.Tensor) model to extract the embedding for observation
        action_size: number of the categorical actions
        """ 
        super(DiagGaussianPolicy, self).__init__()
        self.embedder = embedder
        self.fc_policy = orthogonal_init(nn.Linear(self.embedder.output_dim, action_size), gain = 1.0)
        self.fc_value  = orthogonal_init(nn.Linear(self.embedder.output_dim, 1), gain = 1.0)
        self.log_std = nn.Parameter(torch.zeros(action_size))
        self.action_range = action_range
        
    def pi(self, x):
        """
        p:(torch.distributions.Normal)
        """
        embedding = self.embedder(x)
        embedding_flat = embedding.view(embedding.size(0), -1)
        mu = self.fc_policy(embedding_flat)
        p = Normal(mu, torch.exp(self.log_std))
        return p
    
    def action(self, x):
        """
        action: (torch.Tensor) shape = (batch_size, action_dim)
        """
        p = self.pi(x)
        action = p.rsample()
        return torch.clamp(action, self.action_range[0], self.action_range[1])
    
    def log_prob(self, x, action):
        """
        log_prob:(torch.Tensor) shape = (batch_size, )
        """
        p = self.pi(x)
        return p.log_prob(action)
        
    def v(self, x):
        embedding = self.embedder(x)
        embedding_flat = embedding.view(embedding.size(0), -1)
        v = self.fc_value(embedding_flat).reshape(-1)
        return v


