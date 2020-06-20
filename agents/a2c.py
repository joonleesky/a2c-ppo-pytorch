from .base_agent import BaseAgent
import torch
import torch.optim as optim
import torch.nn.functional as F


class A2C(BaseAgent):
    def __init__(self,
                 env,
                 policy,
                 logger,
                 storage,
                 device,
                 n_steps=32,
                 n_envs=16,
                 gamma=0.99,
                 lmbda=0.95,
                 learning_rate=7e-4,
                 alpha = 0.99,
                 eps = 1e-5,
                 grad_clip_norm=0.5,
                 value_coef=0.5,
                 entropy_coef=0.01,
                 normalize_adv=True,
                 use_gae=True,
                 scale_reward = False,
                 **kwargs):

        super(A2C, self).__init__(env, policy, logger, storage, device)

        self.n_steps = n_steps
        self.n_envs = n_envs
        self.gamma = gamma
        self.lmbda = lmbda
        self.optimizer = optim.RMSprop(self.policy.parameters(), lr=learning_rate, alpha=alpha, eps=eps)
        self.grad_clip_norm = grad_clip_norm
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
        generator = self.storage.fetch_train_generator()

        pi_loss_epoch, value_loss_epoch, entropy_loss_epoch = 0, 0, 0
        num_updates = 0

        for sample in generator:
            obs_batch, act_batch, _, _, return_batch, adv_batch = sample

            # Advantageous policy-gradient
            dist_batch = self.policy.pi(obs_batch)
            log_prob_act_batch = dist_batch.log_prob(act_batch)
            pi_loss = - (log_prob_act_batch * adv_batch.detach()).mean()

            # Bellman-Error of value-function
            value_batch = self.policy.v(obs_batch)
            value_loss = F.smooth_l1_loss(value_batch, return_batch.detach())

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