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


class AdaptativePPO(BaseAgent):
    def __init__(self,
        env, opt, p_layers=[32], v_layers=[32], gamma=0.99, lr=7e-4,
        clip_val=1.0, batch_size=32, target_kl=0.25, ent_coef=0.01,
        p_train_steps=32, v_train_steps=32):

        super(AdaptativePPO, self).__init__(env, opt)

        # training options
        self.gamma = opt.get('gamma', gamma)
        self.batch_size = opt.get('batch_size', batch_size)
        self.clip_val = opt.get('clipVal', clip_val)
        self.ent_coef = opt.get('entropyCoef', ent_coef)
        self.p_train_steps = opt.get('policyTrainSteps', p_train_steps)
        self.v_train_steps = opt.get('valueTrainSteps', v_train_steps)
        self.kl_coef = 1. # adaptative
        self.target_kl = opt.get('targetKL', target_kl)

        # optimizer options
        self.lr = opt.get('learningRate', lr)

        self.test = False # flag for testing mode

        # policy network
        obs_size, out_size = self.featureExtractor.outSize, env.action_space.n
        p_layers =  opt.get('policyLayers', p_layers)
        self.P = NN(obs_size, out_size, layers=p_layers)

        # value network
        v_layers = opt.get('valueLayers', v_layers)
        self.V = NN(obs_size, 1, layers=v_layers)

        # optimizer and value loss
        self.v_loss_fn = torch.nn.SmoothL1Loss()
        self.p_optim = torch.optim.Adam(self.P.parameters(), self.lr)
        self.v_optim = torch.optim.Adam(self.V.parameters(), self.lr)

    def learn(self, batch):
        obs, act, adv, tgt, old_logp = batch

        # take K training steps for the policy network
        for _ in range(self.p_train_steps):
            dist, logp = self._compute_policy_dist(obs, act)

            p_loss = torch.mean(- adv * logp)
            kl_div = torch.mean(torch.exp(old_logp) * (old_logp - logp))
            entropy = torch.mean(dist.entropy())

            self.p_optim.zero_grad()
            loss = p_loss + self.kl_coef * kl_div + self.ent_coef * entropy
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.P.parameters(), self.clip_val)
            self.p_optim.step()

        # adapt KL divergence coeficient
        with torch.no_grad():
            _, logp = self._compute_policy_dist(obs, act)
            kl_div = torch.mean(torch.exp(old_logp) * (old_logp - logp)).item()
            if kl_div >= 1.5 * self.target_kl:
                self.kl_coef *= 2
            if kl_div <= self.target_kl / 1.5:
                self.kl_coef *= 0.5

        # fit the value network
        for _ in range(self.v_train_steps):
            self.v_optim.zero_grad()
            v_loss = self.v_loss_fn(self.V(obs), tgt)
            v_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.V.parameters(), self.clip_val)
            self.v_optim.step()

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

class ClippedPPO(BaseAgent):
    def __init__(self,
        env, opt, p_layers=[32], v_layers=[32], gamma=0.99, lr=7e-4,
        grad_clip_val=1.0, batch_size=32, clip_ratio=0.01, ent_coef=0.01,
        p_train_steps=32, v_train_steps=32):

        super(ClippedPPO, self).__init__(env, opt)

        # training options
        self.gamma = opt.get('gamma', gamma)
        self.batch_size = opt.get('batch_size', batch_size)
        self.clip_val = opt.get('clipVal', grad_clip_val)
        self.ent_coef = opt.get('entropyCoef', ent_coef)
        self.p_train_steps = opt.get('policyTrainSteps', p_train_steps)
        self.v_train_steps = opt.get('valueTrainSteps', v_train_steps)
        self.clip_ratio = opt.get('clipRatio', clip_ratio)

        # optimizer options
        self.lr = opt.get('learningRate', lr)

        self.test = False # flag for testing mode

        # policy network
        obs_size, out_size = self.featureExtractor.outSize, env.action_space.n
        p_layers =  opt.get('policyLayers', p_layers)
        self.P = NN(obs_size, out_size, layers=p_layers)

        # value network
        v_layers = opt.get('valueLayers', v_layers)
        self.V = NN(obs_size, 1, layers=v_layers)

        # optimizer and value loss
        self.v_loss_fn = torch.nn.SmoothL1Loss()
        self.p_optim = torch.optim.Adam(self.P.parameters(), self.lr)
        self.v_optim = torch.optim.Adam(self.V.parameters(), self.lr)

    def learn(self, batch):
        obs, act, adv, tgt, old_logp = batch

        # take K training steps for the policy network
        for _ in range(self.p_train_steps):
            dist, logp = self._compute_policy_dist(obs, act)

            ratio = torch.exp(logp - old_logp)
            clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv

            p_loss = -(torch.min(ratio * adv, clip_adv)).mean()
            entropy = torch.mean(dist.entropy())

            self.p_optim.zero_grad()
            loss = p_loss + self.ent_coef * entropy
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.P.parameters(), self.clip_val)
            self.p_optim.step()

        # fit the value network
        for _ in range(self.v_train_steps):
            self.v_optim.zero_grad()
            v_loss = self.v_loss_fn(self.V(obs), tgt)
            v_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.V.parameters(), self.clip_val)
            self.v_optim.step()

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
    parser.add_argument('--variant', type=str, default='adaptative',
        help="PPO variant to use, either 'adaptative' or 'clipped'.")
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
    name = '{}_ppo_lam{}'.format(args.variant, args.lam)
    outdir = "./XP/" + config["env"] + "/" + name + "_" + tstart

    # seed rngs
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    if args.variant == 'adaptative':
        agent = AdaptativePPO(env, config)
    elif args.variant == 'clipped':
        agent = ClippedPPO(env, config)
    else:
        raise ValueError("Argument 'variant' must be either 'adaptative' or 'clipped'.")

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
