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
import torch.nn
import torch.optim
from torch.utils.tensorboard import SummaryWriter

from utils import *
from memory import Memory


#matplotlib.use("Qt5agg")
matplotlib.use("TkAgg")


WARM_UP = 1000

class BaseAgent(object):
    """Base agent for DQN."""
    def __init__(self, env, opt):
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
                clip_val=1.0, mem_size=50000, batch_size=32):
        super(DQN, self).__init__(env, opt)
        self.eps = 1.0
        self.min_eps = opt.get('minEpsilon', min_eps)
        self.eps_decay = opt.get('epsilonDecay', eps_decay)
        self.gamma = opt.get('gamma', gamma)
        self.batch_size = opt.get('batch_size', batch_size)
        self.per = per
        self.clip_val = opt.get('clipVal', clip_val)
        self.update_step_ = 0 # set to zero when reaches target_update_freq
        self.test = False # flag for testing mode

        # Replay Buffer
        self.memory = Memory(opt.get('memSize', mem_size), prior=per)

        # Q-Network
        layers = opt.get('layers', layers)
        in_size, out_size = self.featureExtractor.outSize, self.action_space.n
        self.Q = NN(in_size, out_size, layers=layers)

        # Target Network
        self.target_Q, self.target_freq = None, None
        if target_update_freq > 0:
            self.target_Q = NN(in_size, out_size, layers=layers)
            self.target_freq = target_update_freq

        # Optimizer and
        lr = opt.get('learningRate', lr)
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.optim = torch.optim.Adam(self.Q.parameters(), lr)

    def learn(self):
        # sample batch
        idx, w, batch = self.memory.sample(self.batch_size)

        # prepare input and targets for the Q network
        batch_size = batch.shape[0]
        ob_shape = (batch_size, self.featureExtractor.outSize)

        obs = torch.zeros(ob_shape, dtype=torch.float32)
        act = torch.zeros(batch_size, dtype=torch.int64)
        rews = torch.zeros(batch_size, dtype=torch.float32)
        next_obs = torch.zeros(ob_shape, dtype=torch.float32)
        dones = torch.zeros(batch_size, dtype=torch.bool)

        with torch.no_grad():
            for i in range(batch_size):
                o, a, r, next_o, done = batch[i]
                o = torch.tensor(self.featureExtractor.getFeatures(o), dtype=torch.float32)
                next_o = torch.tensor(self.featureExtractor.getFeatures(next_o), dtype=torch.float32)
                obs[i], act[i], rews[i], next_obs[i], dones[i] = o, a, r, next_o, bool(done)

            next_values = None
            if self.target_Q is None:
                next_values = self.Q(next_obs)
            else:
                next_values = self.target_Q(next_obs)
            next_values, _ = torch.max(next_values, dim=-1)
            y = torch.where(dones, rews, rews + self.gamma * next_values)

        # perform an optimization step
        self.optim.zero_grad()
        q_value = torch.squeeze(torch.gather(self.Q(obs), 1, act.unsqueeze(1)))
        tderr = self.loss_fn(q_value, y)
        if self.per:
            self.memory.update(idx, tderr)
            tderr *= w
        tderr.backward()
        torch.nn.utils.clip_grad_norm_(self.Q.parameters(), self.clip_val) # clip gradients
        self.optim.step()

        # update target
        self.update_step_ += 1
        if self.update_step_ == self.target_freq:
            self.target_Q.load_state_dict(self.Q.state_dict())
            self.update_step_ = 0

        # update epsilon
        self.eps = max(self.eps - self.eps_decay, self.min_eps)

        return tderr.item()

    def act(self, observation):
        obs = torch.tensor(
            self.featureExtractor.getFeatures(observation),
            dtype=torch.float32)
        action = None
        if not self.test and random.random() < self.eps:
            action = self.action_space.sample()
        else:
            logits = self.Q(obs)
            action = torch.argmax(logits, dim=-1).item()
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
    name = 'dqn{}{}'.format(
        '_per' if args.per else '', '_target' if args.target_update_freq > 0 else '')
    outdir = "./XP/" + config["env"] + "/" + name + "_" + tstart

    # seed rngs
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    obs = env.reset()

    eps_decay = (1 - config['minEpsilon']) / (config['explorationFraction'] * config['nbTimesteps'])
    agent = DQN(env, config, eps_decay=eps_decay,
        target_update_freq=args.target_update_freq)

    print("Saving in " + outdir)
    os.makedirs(outdir, exist_ok=True)
    save_src(os.path.abspath(outdir))
    write_yaml(os.path.join(outdir, 'info.yaml'), config)
    logger = LogMe(SummaryWriter(outdir))
    loadTensorBoard(outdir)

    i, t = 0, 0
    rsum = 0
    mean = 0
    verbose = True
    itest, test_start = 0, 0
    reward = 0
    done = False
    while t < config['nbTimesteps']:
        if (args.verbose and
            i % int(config["freqVerbose"]) == 0 and i >= config["freqVerbose"]):
            verbose = True
        else:
            verbose = False

        if i % freqTest == 0 and i >= freqTest:  ##### Same as train for now
            print("Test time! ")
            mean = 0
            agent.test = True

        if i % freqTest == nbTest and i > freqTest:
            print("End of test, mean reward=", mean / nbTest)
            itest += 1
            logger.direct_write("rewardTest", mean / nbTest, itest)
            agent.test = False

        if i % freqSave == 0:
            agent.save(outdir + "/save_" + str(i))

        j = 0
        if verbose:
            env.render()

        while True:
            if verbose:
                env.render()

            action = agent.act(obs)
            next_obs, reward, done, _ = env.step(action)
            j += 1

            if not agent.test:
                agent.memory.store((obs, action, reward, next_obs, done))
                if agent.memory.nentities >= config.get('warmUp', WARM_UP):
                    agent.learn()
                t += 1

            obs = next_obs
            rsum += reward
            if done:
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                logger.direct_write("reward", rsum, i)
                agent.nbEvents = 0
                mean += rsum
                rsum = 0
                obs = env.reset()
                break

        i += 1

    env.close()
