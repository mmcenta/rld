import argparse
import os
import random
import sys
import time

import gym
import gridworld
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.utils.tensorboard import SummaryWriter

from utils import *
from memory import Memory
from models import QNetwork


#matplotlib.use("Qt5agg")
matplotlib.use("TkAgg")


WARM_UP = 1000

class BaseAgent(nn.Module):
    """Base agent for DQN."""
    def __init__(self, env, opt):
        super(BaseAgent, self).__init__()
        self.opt = opt
        self.env = env
        if opt.fromFile is not None:
            self.load(opt.fromFile)
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)

    def act(self, observation):
        pass

    def learn(self, batch):
        pass

    def save(self, outputDir):
        pass

    def load(self, inputDir):
        pass


class DQN(BaseAgent):
    def __init__(self, env, opt, layers=[32], min_eps=0.05, eps_decay=0.01,
                gamma=0.99, lr=1e-3, per=False, target_update_freq=-1,
                clip_val=None, double=False, dueling=False, noisy=False):
        super(DQN, self).__init__(env, opt)
        self.eps = 1.0
        self.min_eps = opt.get('minEpsilon', min_eps)
        self.eps_decay = opt.get('epsilonDecay', eps_decay)
        self.gamma = opt.get('gamma', gamma)
        self.clip_val = opt.get('clipVal', clip_val)
        self.per = per
        self.double = double
        self.dueling = dueling
        self.noisy = noisy

        self.update_step_ = 0 # set to zero when reaches target_update_freq

        # Create neural networks
        obs_size, act_size = self.featureExtractor.outSize, self.action_space.n

        layers = opt.get('layers', layers)
        self.Q = QNetwork(obs_size, act_size, layers=layers, dueling=dueling,
            noisy=noisy)

        self.target_freq = None
        if target_update_freq > 0:
            self.target_Q = QNetwork(obs_size, act_size, layers=layers,
                dueling=dueling, noisy=noisy)
            self.target_freq = target_update_freq
        else:
            self.target_Q = self.Q

        # Optimizer and loss
        lr = opt.get('learningRate', lr)
        self.loss_fn = torch.nn.SmoothL1Loss(reduce=False)
        self.optim = torch.optim.Adam(self.Q.parameters(), lr)

    def learn(self, batch, w):
        # prepare tensors
        batch_size = batch.shape[0]
        ob_shape = (batch_size, self.featureExtractor.outSize)

        obs = torch.zeros(ob_shape, dtype=torch.float32)
        act = torch.zeros(batch_size, dtype=torch.int64)
        rews = torch.zeros(batch_size, dtype=torch.float32)
        next_obs = torch.zeros(ob_shape, dtype=torch.float32)
        dones = torch.zeros(batch_size, dtype=torch.bool)

        for i in range(batch_size):
            o, a, r, next_o, done = batch[i]
            o = torch.tensor(self.featureExtractor.getFeatures(o), dtype=torch.float32)
            next_o = torch.tensor(self.featureExtractor.getFeatures(next_o), dtype=torch.float32)
            obs[i], act[i], rews[i], next_obs[i], dones[i] = o, a, r, next_o, bool(done)

        # compute targets
        with torch.no_grad():
            next_values = None
            if self.double:
                q_values = self.Q(next_obs)
                target_q_values = self.target_Q(next_obs)
                argmax = torch.argmax(q_values, dim=-1, keepdim=True)
                next_values = torch.squeeze(
                    torch.gather(target_q_values, 1, argmax))
            else:
                target_q_values = self.target_Q(next_obs)
                next_values, _ = torch.max(target_q_values, dim=-1)
            y = torch.where(dones, rews, rews + self.gamma * next_values)

        # perform an optimization step
        self.optim.zero_grad()
        q_value = torch.squeeze(
            torch.gather(self.Q(obs), 1, act.unsqueeze(1)))
        tderr = self.loss_fn(q_value, y)
        if self.per:
            loss = torch.squeeze(torch.tensor(w)) @ tderr
        else:
            loss = torch.mean(tderr)
        loss.backward()
        if self.clip_val:
            torch.nn.utils.clip_grad_norm_(self.Q.parameters(), self.clip_val) # clip gradients
        self.optim.step()

        # update target
        if self.target_freq is not None:
            self.update_step_ += 1
            if self.update_step_ == self.target_freq:
                self.target_Q.load_state_dict(self.Q.state_dict())
                self.update_step_ = 0

        # update epsilon
        self.eps = max(self.eps - self.eps_decay, self.min_eps)

        return tderr.detach().numpy()

    def act(self, observation):
        obs = torch.tensor(
            self.featureExtractor.getFeatures(observation),
            dtype=torch.float32)

        if self.noisy:
            self.Q.reset_noise()
            action = torch.argmax(self.Q(obs), dim=-1).item()
        if self.training and random.random() < self.eps:
            action = self.action_space.sample()
        else:
            action = torch.argmax(self.Q(obs), dim=-1).item()
        return action


## Runner ##
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='gridworld',
        help="Runner environment, either 'gridworld', 'cartpole' or 'lunar'")
    parser.add_argument('-per', action='store_true',
        help='If set, Prioritized Experience Replay is used.')
    parser.add_argument('--target-update-freq', type=int, default=-1,
        help='Frequency in which ')
    parser.add_argument('-double', action='store_true',
        help='If set, uses double DQN objective.')
    parser.add_argument('-dueling', action='store_true',
        help='If set, dueling DQN is used.')
    parser.add_argument('-noisy', action='store_true',
        help='If set, noisy DQN is used.')
    parser.add_argument('--verbose', '-v', action='store_true',
        help='If set, will render an episode occasionally.')
    args = parser.parse_args()

    # load configs
    config = load_yaml('./configs/config_random_{}.yaml'.format(args.env))
    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]

    # create environment
    env = gym.make(config["env"])
    if hasattr(env, 'setPlan'):
        env.setPlan(config["map"], config["rewards"])

    # set experiment directory
    tstart = str(time.time())
    tstart = tstart.replace(".", "_")
    name = 'dqn{}{}{}{}'.format(
        '_per' if args.per else '', '_target' if args.target_update_freq > 0 else '',
        '_double' if args.double else '', '_dueling' if args.dueling else '',
        '_noisy' if args.noisy else '')
    outdir = "./XP/" + config["env"] + "/" + name + "_" + tstart

    # seed rngs
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    eps_decay = (1 - config['minEpsilon']) / (config['explorationFraction'] * config['nbTimesteps'])
    agent = DQN(env, config, per=args.per, eps_decay=eps_decay,
        target_update_freq=args.target_update_freq, double=args.double,
        dueling=args.dueling, noisy=args.noisy)

    memory = Memory(config.get('memSize'), prior=args.per)

    print("Saving in " + outdir)
    os.makedirs(outdir, exist_ok=True)
    save_src(os.path.abspath(outdir))
    write_yaml(os.path.join(outdir, 'info.yaml'), config)
    logger = LogMe(SummaryWriter(outdir))
    loadTensorBoard(outdir)

    episode, timestep = 0, 0
    rsum, mean = 0, 0
    verbose = True
    itest, test_start = 0, 0
    obs, reward, done = env.reset(), 0, False
    while timestep < config['nbTimesteps']:
        if (args.verbose and
            episode % int(config["freqVerbose"]) == 0 and episode >= config["freqVerbose"]):
            verbose = True
        else:
            verbose = False

        if episode % freqTest == 0 and episode >= freqTest:  ##### Same as train for now
            print("Test time! ")
            mean = 0
            agent.eval()

        if episode % freqTest == nbTest and episode > freqTest:
            print("End of test, mean reward=", mean / nbTest)
            itest += 1
            logger.direct_write("rewardTest", mean / nbTest, timestep)
            agent.train()

        if episode % freqSave == 0:
            agent.save(outdir + "/save_" + str(episode))

        ep_steps = 0
        if verbose:
            env.render()

        while True:
            if verbose:
                env.render()

            action = agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            ep_steps += 1

            if agent.training:
                truncated = info.get('TimeLimit.truncated', False)
                memory.store((obs, action, reward, next_obs, done and not truncated))
                if memory.nentities >= config.get('warmUp', WARM_UP):
                    idx, w, batch = memory.sample(config['batchSize'])
                    tderr = agent.learn(batch, w)
                    if args.per:
                        memory.update(idx, tderr)
                timestep += 1

            obs = next_obs
            rsum += reward
            if done:
                print(str(episode) + " rsum=" + str(rsum) + ", " + str(ep_steps) + " actions ")
                logger.direct_write("reward", rsum, timestep)
                agent.nbEvents = 0
                mean += rsum
                rsum = 0
                obs = env.reset()
                break

        episode += 1

    print('Done!')
    env.close()
