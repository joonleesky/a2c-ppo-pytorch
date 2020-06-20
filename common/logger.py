import numpy as np
import pandas as pd
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import time

class Logger(object):
    
    def __init__(self, n_envs, logdir):
        self.start_time = time.time()
        self.n_envs = n_envs
        self.logdir = logdir

        self.episode_rewards = []
        for _ in range(n_envs):
            self.episode_rewards.append([])
        self.episode_len_buffer = deque(maxlen = 40)
        self.episode_reward_buffer = deque(maxlen = 40)
        
        self.log = pd.DataFrame(columns = ['timesteps', 'wall_time', 'num_episodes',
                               'max_episode_rewards', 'mean_episode_rewards','min_episode_rewards',
                               'max_episode_len', 'mean_episode_len', 'min_episode_len'])
        self.writer = SummaryWriter(logdir)
        self.timesteps = 0
        self.num_episodes = 0
        

    def feed(self, rew_batch, done_batch):
        steps = rew_batch.shape[0]
        rew_batch = rew_batch.T
        done_batch = done_batch.T

        for i in range(self.n_envs):
            for j in range(steps):
                self.episode_rewards[i].append(rew_batch[i][j])
                if done_batch[i][j]:
                    self.episode_len_buffer.append(len(self.episode_rewards[i]))
                    self.episode_reward_buffer.append(np.sum(self.episode_rewards[i]))
                    self.episode_rewards[i] = []
                    self.num_episodes += 1
        self.timesteps += (self.n_envs * steps)

    def write_summary(self, summary):
        if self.num_episodes > 0:
            summary['Rewards/max_episodes']  = np.max(self.episode_reward_buffer)
            summary['Rewards/mean_episodes'] = np.mean(self.episode_reward_buffer)

        for key, value in summary.items():
            self.writer.add_scalar(key, value, self.timesteps)

    
    def dump(self):
        wall_time = time.time() - self.start_time
        log = [self.timesteps] + [wall_time] + [self.num_episodes] + self._get_episode_statistics()
        self.log.loc[len(self.log)] = log

        # TODO: logger to append, not write!
        with open(self.logdir + '/log.csv', 'w') as f:
            self.log.to_csv(f, index = False)


    def _get_episode_statistics(self):
        if self.num_episodes == 0:
            return [None] * 6
        max_episode_rewards  = np.max(self.episode_reward_buffer)
        mean_episode_rewards = np.mean(self.episode_reward_buffer)
        min_episode_rewards  = np.min(self.episode_reward_buffer)
        max_episode_len  = np.max(self.episode_len_buffer)
        mean_episode_len = np.mean(self.episode_len_buffer)
        min_episode_len  = np.min(self.episode_len_buffer)

        return [max_episode_rewards, mean_episode_rewards, min_episode_rewards,
                max_episode_len, mean_episode_len, min_episode_len]