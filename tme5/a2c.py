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
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from utils import *
from rollout import RolloutCollector


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


class A2C(BaseAgent):
    def __init__(self,
        env, opt, p_layers=[32], v_layers=[32], gamma=0.99, lr=7e-4,
        clip_val=1.0, batch_size=32, v_coef=0.25, ent_coef=0.01, alpha=0.99,
        momentum=0.0, rmsprop_eps=1e-5):

        super(A2C, self).__init__(env, opt)

        # training options
        self.gamma = opt.get('gamma', gamma)
        self.batch_size = opt.get('batch_size', batch_size)
        self.clip_val = opt.get('clipVal', clip_val)
        self.v_coef = opt.get('valueCoef', v_coef)
        self.ent_coef = opt.get('entropyCoef', ent_coef)

        # optimizer options
        self.lr = opt.get('learningRate', lr)
        self.alpha = opt.get('alpha', alpha)
        self.momentum = opt.get('momentum', momentum)
        self.rmsprop_eps = opt.get('RMSPropEpsilon', rmsprop_eps)

        self.test = False # flag for testing mode

        # policy network
        obs_size, out_size = self.featureExtractor.outSize, env.action_space.n
        p_layers =  opt.get('policyLayers', p_layers)
        self.P = NN(obs_size, out_size, layers=p_layers)

        # value networks
        v_layers = opt.get('valueLayers', v_layers)
        self.V = NN(obs_size, 1, layers=v_layers)

        # optimizer and value loss
        self.v_loss_fn = torch.nn.SmoothL1Loss()
        parameters = list(self.P.parameters()) + list(self.V.parameters())
        self.optim = torch.optim.RMSprop(parameters, self.lr,
            alpha=self.alpha, momentum=self.momentum, eps=self.rmsprop_eps)

    def learn(self, batch):
        obs, act, adv, tgt, old_logp = batch

        # compute policy loss and entropy
        dist, logp = self._compute_policy_dist(obs, act)
        p_loss, entropy = torch.mean(- adv * logp), torch.mean(dist.entropy())

        # compute value loss
        v_loss = self.v_loss_fn(self.V(obs), tgt)

        # optimize
        self.optim.zero_grad()
        loss = p_loss + self.v_coef * v_loss + self.ent_coef * entropy
        loss.backward()
        parameters = list(self.P.parameters()) + list(self.V.parameters())
        torch.nn.utils.clip_grad_norm_(parameters, self.clip_val) # clip gradients
        self.optim.step()

        return loss.detach().item()

    def _compute_policy_dist(self, obs, act=None):
        logits = self.P(obs)
        dist = Categorical(logits=logits)
        if act is not None:
            logp = dist.log_prob(act)
            return dist, logp
        return dist, None

    def act(self, observation):
        with torch.no_grad():
            obs = torch.tensor(
                self.featureExtractor.getFeatures(observation),
                dtype=torch.float32)
            value = self.V(obs)
            dist, _ = self._compute_policy_dist(obs)
            action = dist.sample()
            logp = dist.log_prob(action)
        return action.item(), value, logp


## Runner ##
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='gridworld',
        help="Runner environment, either 'gridworld', 'cartpole' or 'lunar'")
    parser.add_argument('--lam', type=float, default=0.97,
        help='Lambda parameter for calculating targets and advantages.')
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
    name = 'a2c_lam{}'.format(args.lam)
    outdir = "./XP/" + config["env"] + "/" + name + "_" + tstart

    # seed rngs
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    agent = A2C(env, config)

    obs_shape, act_shape = env.observation_space.shape, env.action_space.shape
    collector = RolloutCollector(obs_shape, act_shape, config['batchSize'],
        gamma=config['gamma'], lam=args.lam)

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
            episode % int(config["freqVerbose"]) == 0 and
            episode >= config["freqVerbose"]):
            verbose = True
        else:
            verbose = False

        if episode % freqTest == 0 and episode >= freqTest:  ##### Same as train for now
            print("Test time! ")
            mean = 0
            agent.test = True

        if episode % freqTest == nbTest and episode > freqTest:
            print("End of test, mean reward=", mean / nbTest)
            itest += 1
            logger.direct_write("rewardTest", mean / nbTest, itest)
            agent.test = False

        if episode % freqSave == 0:
            agent.save(outdir + "/save_" + str(episode))

        ep_steps = 0
        if verbose:
            env.render()

        while True:
            if verbose:
                env.render()

            action, value, logp = agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            ep_steps += 1

            if not agent.test:
                if collector.is_full():
                    collector.finish_path(value)
                    batch = collector.get()
                    agent.learn(batch)
                truncated = info.get('TimeLimit.truncated', False)
                collector.store(obs, action, reward, done and not truncated, value, logp)
                timestep += 1

            obs = next_obs
            rsum += reward
            if done:
                print(str(episode) + " rsum=" + str(rsum) + ", " + str(ep_steps) + " actions ")
                logger.direct_write("reward", rsum, episode)
                agent.nbEvents = 0
                mean += rsum
                rsum = 0
                obs = env.reset()
                break

        episode += 1

    env.close()
