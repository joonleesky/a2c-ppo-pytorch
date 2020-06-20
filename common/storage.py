import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np

class Storage():

    def __init__(self, obs_shape, num_steps, num_envs, device):
        self.obs_shape = obs_shape
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device

        self.obs_batch = torch.zeros(self.num_steps+1, self.num_envs, *self.obs_shape)
        self.act_batch = torch.zeros(self.num_steps, self.num_envs)
        self.rew_batch = torch.zeros(self.num_steps, self.num_envs)
        self.done_batch = torch.zeros(self.num_steps, self.num_envs)
        self.log_prob_act_batch = torch.zeros(self.num_steps, self.num_envs)
        self.value_batch = torch.zeros(self.num_steps+1, self.num_envs)
        self.return_batch = torch.zeros(self.num_steps, self.num_envs)
        self.adv_batch = torch.zeros(self.num_steps, self.num_envs)

        # For reward-normalization
        self.rolling_returns_batch = torch.zeros(self.num_steps, self.num_envs)
        self.rolling_mean_batch = torch.zeros(self.num_steps, self.num_envs)
        self.rolling_dev_batch = torch.zeros(self.num_steps, self.num_envs) # deviation
        self.eps_steps_batch = torch.zeros(self.num_steps, self.num_envs)
        self.step = 0


    def store(self, obs, act, rew, done, log_prob_act, value):
        self.obs_batch[self.step] = torch.from_numpy(obs)
        self.act_batch[self.step] = torch.from_numpy(act)
        self.rew_batch[self.step] = torch.from_numpy(rew)
        self.done_batch[self.step] = torch.from_numpy(done)
        self.log_prob_act_batch[self.step] = torch.from_numpy(log_prob_act)
        self.value_batch[self.step] = torch.from_numpy(value)

        self.eps_steps_batch[self.step] = self.eps_steps_batch[self.step - 1] * (1 - self.done_batch[self.step]) + 1
        self.step = (self.step + 1) % self.num_steps


    def store_last(self, last_obs, last_val):
        self.obs_batch[-1] = torch.from_numpy(last_obs)
        self.value_batch[-1] = torch.from_numpy(last_val)


    def scale_reward(self, gamma = 0.99):
        """
        rewards are divided through the standard deviation of a rolling discounted sum of the rewards
        (Appendix A.2: https://arxiv.org/abs/2005.12729)
        """
        R = self.rolling_returns_batch[-1]
        M = self.rolling_mean_batch[-1]
        V = self.rolling_dev_batch[-1]

        for i in range(self.num_steps):
            rew = self.rew_batch[i]
            done = self.done_batch[i]
            eps_steps = self.eps_steps_batch[i]

            R = rew + gamma * R * (1-done)
            M *= (1-done)
            V *= (1-done)
            oldM = M
            M = M + ((R - M) / eps_steps)
            V = V + (R - oldM) * (R - M)
            self.rolling_returns_batch[i] = R
            self.rolling_mean_batch[i] = M
            self.rolling_dev_batch[i] = V

        # Var(X) = (if n> 1) : dev(X)/(n -1)  (else) : x^2
        rolling_var_batch = (self.rolling_dev_batch / (self.eps_steps_batch - 1 + 1e-5)) * (self.eps_steps_batch != 1)
        rolling_var_batch += self.rolling_mean_batch.pow(2) * (self.eps_steps_batch == 1)
        rolling_std_batch = torch.sqrt(rolling_var_batch)

        return self.rew_batch / (rolling_std_batch + 1e-5)


    def compute_estimates(self, gamma = 0.99, lmbda = 0.95, use_gae = False, normalize_adv = True, scale_reward = False):
        if scale_reward == True:
            rew_batch = self.scale_reward(gamma)
        else:
            rew_batch = self.rew_batch

        if use_gae == True:
            A = 0
            for i in reversed(range(self.num_steps)):
                rew = rew_batch[i]
                done = self.done_batch[i]
                value = self.value_batch[i]
                next_value = self.value_batch[i+1]

                delta = (rew + gamma * next_value * (1 - done)) - value
                A = gamma * lmbda * A * (1 - done) + delta
                self.return_batch[i] = A + value
        else:
            G = self.value_batch[-1]
            for i in reversed(range(self.num_steps)):
                rew = rew_batch[i]
                done = self.done_batch[i]

                G = rew + gamma * G * (1 - done)
                self.return_batch[i] = G

        self.adv_batch = self.return_batch - self.value_batch[:-1]
        if normalize_adv == True:
            self.adv_batch = (self.adv_batch - torch.mean(self.adv_batch)) / (torch.std(self.adv_batch) + 1e-5)


    def fetch_train_generator(self, mini_batch_size = None):
        batch_size = self.num_steps * self.num_envs
        if mini_batch_size is None:
            mini_batch_size = batch_size
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)),
                               mini_batch_size,
                               drop_last=True)
        for indices in sampler:
            obs_batch = torch.FloatTensor(self.obs_batch[:-1]).reshape(-1, *self.obs_shape)[indices].to(self.device)
            act_batch = torch.FloatTensor(self.act_batch).reshape(-1)[indices].to(self.device)
            log_prob_act_batch = torch.FloatTensor(self.log_prob_act_batch).reshape(-1)[indices].to(self.device)
            value_batch = torch.FloatTensor(self.value_batch).reshape(-1)[indices].to(self.device)
            return_batch = torch.FloatTensor(self.return_batch).reshape(-1)[indices].to(self.device)
            adv_batch = torch.FloatTensor(self.adv_batch).reshape(-1)[indices].to(self.device)

            yield obs_batch, act_batch, log_prob_act_batch, value_batch, return_batch, adv_batch


    def fetch_log_data(self):
        rew_batch = self.rew_batch.numpy()
        done_batch = self.done_batch.numpy()

        return rew_batch, done_batch