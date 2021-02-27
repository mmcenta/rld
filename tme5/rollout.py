from collections import deque

import numpy as np
from scipy.signal import lfilter
import torch


# Taken and adapted from SpinningUp (https://github.com/openai/spinningup/)
def discount_cumsum(x, discount):
    return lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class RolloutCollector:
    """
    Stores rollouts until a batch is complete. Computes TD(λ) targets AND
    GAE(λ) advantages (https://arxiv.org/abs/1506.02438).
    """
    def __init__(self, obs_size, act_size, batch_size, gamma=0.99, lam=0.97):
        if isinstance(obs_size, int):
            obs_size = (obs_size,)
        if isinstance(act_size, int):
            act_size = (act_size,)
        self.obs_buf = np.zeros((batch_size,) + obs_size, dtype=np.float32)
        self.act_buf = np.zeros((batch_size,) + act_size, dtype=np.float32)
        self.rew_buf = np.zeros(batch_size, dtype=np.float32)
        self.val_buf = np.zeros(batch_size, dtype=np.float32)
        self.logp_buf = np.zeros(batch_size, dtype=np.float32)
        self.adv_buf = np.zeros(batch_size, dtype=np.float32)
        self.tgt_buf = np.zeros(batch_size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, batch_size

    def store(self, obs, act, rew, done, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1
        if done:
            self.finish_path()

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # TD(λ) targets
        deltas = rews[:-1] + self.gamma * vals[1:]
        self.tgt_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # GAE(λ) advantages
        deltas -= vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0

        # advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        data = [self.obs_buf, self.act_buf, self.adv_buf, self.tgt_buf,
            self.logp_buf]
        return tuple([torch.as_tensor(t, dtype=torch.float32) for t in data])

    def is_full(self):
        return self.ptr == self.max_size