from .base_agent import BaseAgent
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class PPO(BaseAgent):
    def __init__(self,
                 env,
                 policy,
                 logger,
                 storage,
                 device,
                 n_steps=128,
                 n_envs=8,
                 epoch=3,
                 mini_batch_size=32*8,
                 gamma=0.99,
                 lmbda=0.95,
                 learning_rate=2.5e-4,
                 grad_clip_norm=0.5,
                 eps_clip=0.1,
                 value_coef=1.0,
                 entropy_coef=0.01,
                 normalize_adv=True,
                 use_gae=True,
                 scale_reward=False,
                 **kwargs):

        super(PPO, self).__init__(env, policy, logger, storage, device)

        self.n_steps = n_steps
        self.n_envs = n_envs
        self.epoch = epoch
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.lmbda = lmbda
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate, eps=1e-5)
        self.grad_clip_norm = grad_clip_norm
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.normalize_adv = normalize_adv
        self.use_gae = use_gae
        self.scale_reward = scale_reward


    def predict(self, obs):
        obs = torch.FloatTensor(obs).to(device=self.device)
        dist = self.policy.pi(obs)
        act = dist.sample()
        log_prob_act = dist.log_prob(act).detach()
        value = self.policy.v(obs).detach()

        return act.cpu().numpy(), log_prob_act.cpu().numpy(), value.cpu().numpy()


    def optimize(self):
        pi_loss_epoch, value_loss_epoch, entropy_loss_epoch = 0, 0, 0
        num_updates = 0

        for e in range(self.epoch):
            generator = self.storage.fetch_train_generator(mini_batch_size=self.mini_batch_size)
            for sample in generator:
                obs_batch, act_batch, old_log_prob_act_batch, old_value_batch, return_batch, adv_batch = sample

                # Clipped Surrogate Objective
                dist_batch = self.policy.pi(obs_batch)
                log_prob_act_batch = dist_batch.log_prob(act_batch)
                ratio = torch.exp(log_prob_act_batch - old_log_prob_act_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * adv_batch
                pi_loss = -torch.min(surr1, surr2).mean()

                # Clipped Bellman-Error
                value_batch = self.policy.v(obs_batch)
                # clipped_value_batch = old_value_batch + (value_batch - old_value_batch).clamp(-self.eps_clip, self.eps_clip)
                # v_surr1 = (value_batch - return_batch.detach()).pow(2)
                # v_surr2 = (clipped_value_batch - return_batch.detach()).pow(2)
                # value_loss = 0.5 * torch.max(v_surr1, v_surr2).mean()
                value_loss = 0.5 * (value_batch - return_batch.detach()).pow(2).mean()

                # Policy Entropy
                entropy_loss = dist_batch.entropy().mean()

                self.optimizer.zero_grad()
                loss = pi_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
                self.optimizer.step()

                pi_loss_epoch += pi_loss.item()
                value_loss_epoch += value_loss.item()
                entropy_loss_epoch += entropy_loss.item()
                num_updates += 1


        pi_loss_epoch /= num_updates
        value_loss_epoch /= num_updates
        entropy_loss_epoch /= num_updates

        summary = {'Loss/pi': pi_loss_epoch,
                   'Loss/v': value_loss_epoch,
                   'Loss/entropy': entropy_loss_epoch}
        return summary


    def train(self, num_timesteps):
        obs = self.env.reset()

        while self.t < num_timesteps:
            # Run Policy
            for _ in range(self.n_steps):
                act, log_prob_act, value = self.predict(obs)
                next_obs, rew, done, _ = self.env.step(act)
                self.storage.store(obs, act, rew, done, log_prob_act, value)
                obs = next_obs
            last_obs = next_obs
            _, _, last_val = self.predict(last_obs)
            self.storage.store_last(last_obs, last_val)

            # Compute advantage estimates
            self.storage.compute_estimates(self.gamma, self.lmbda, self.use_gae, self.normalize_adv, self.scale_reward)

            # Optimize policy & value
            summary = self.optimize()

            # Log the training-procedure
            self.t += self.n_steps * self.n_envs
            rew_batch, done_batch = self.storage.fetch_log_data()
            self.logger.feed(rew_batch, done_batch)
            self.logger.write_summary(summary)
            self.logger.dump()

        self.env.close()
        torch.save({'state_dict': self.policy.state_dict()}, self.logger.logdir + '/model.pth')