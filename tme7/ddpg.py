import argparse
import os
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


WARM_UP = 1000 # Fill replay buffer
START_AFTER = 10000 # Samples actions randomly at the start

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


class DDPG(BaseAgent):
    def __init__(self, env, opt, q_layers=[32], p_layers=[32], noise_scale=0.1,
                gamma=0.99, p_lr=1e-3, q_lr=1e-4, q_weight_decay=1e-2,
                update_decay=0.999, clip_val=None):
        super(DDPG, self).__init__(env, opt)
        self.noise_scale = opt.get('noiseScale', noise_scale)
        self.update_decay = opt.get('updateDecay', update_decay)
        self.gamma = opt.get('gamma', gamma)
        self.clip_val = opt.get('clipVal', clip_val)
        self.act_low = self.action_space.low[0]
        self.act_high = self.action_space.high[0]
        self.act_range = (self.act_high - self.act_low) / 2

        self.test = False # flag for testing mode

        # Policy Network
        obs_size, act_size = self.featureExtractor.outSize, self.action_space.shape[0]
        p_layers = opt.get('policyLayers', p_layers)
        self.P = NN(obs_size, act_size, layers=p_layers)
        self.target_P = NN(obs_size, act_size, layers=p_layers)

        # Q-Networks
        q_layers = opt.get('qLayers', q_layers)
        self.Q = NN(obs_size + act_size, 1, layers=q_layers)
        self.target_Q = NN(obs_size + act_size, 1, layers=q_layers)

        # Optimizers
        p_lr = opt.get('policyLearningRate', p_lr)
        q_lr = opt.get('qLearningRate', q_lr)
        self.p_optim = torch.optim.Adam(self.P.parameters(), p_lr)
        self.q_optim = torch.optim.Adam(self.Q.parameters(), q_lr)

    def _get_action(self, obs, p_fn):
        return self.act_range * torch.tanh(p_fn(obs))

    def _get_values(self, obs, act, q_fn):
        return torch.squeeze(q_fn(torch.cat([obs, act], dim=-1)))

    def learn(self, batch):
        # prepare batch tensors
        batch_size = batch.shape[0]
        obs_shape = (batch_size, self.featureExtractor.outSize)
        act_shape = (batch_size,) + self.action_space.shape

        obs = torch.zeros(obs_shape, dtype=torch.float32)
        act = torch.zeros(act_shape, dtype=torch.float32)
        rews = torch.zeros(batch_size, dtype=torch.float32)
        next_obs = torch.zeros(obs_shape, dtype=torch.float32)
        dones = torch.zeros(batch_size, dtype=torch.bool)

        for i in range(batch_size):
            o, a, r, next_o, done = batch[i]
            o = torch.tensor(self.featureExtractor.getFeatures(o), dtype=torch.float32)
            next_o = torch.tensor(self.featureExtractor.getFeatures(next_o), dtype=torch.float32)
            a = torch.tensor(a, dtype=torch.float32)
            obs[i], act[i], rews[i], next_obs[i], dones[i] = o, a, r, next_o, bool(done)

        # compute targets
        with torch.no_grad():
            next_act = self._get_action(next_obs, self.target_P)
            next_values = self._get_values(next_obs, next_act, self.target_Q)
            y = torch.where(dones, rews, rews + self.gamma * next_values)

        # optimize Q
        self.q_optim.zero_grad()
        values = self._get_values(obs, act, self.Q)
        q_loss = ((values - y) ** 2).mean()
        q_loss.backward()
        if self.clip_val is not None:
            torch.nn.utils.clip_grad_norm_(self.Q.parameters(), self.clip_val)
        self.q_optim.step()

        # optimize policy
        for w in self.Q.parameters(): # freeze Q parameters
            w.requires_grad = False

        self.p_optim.zero_grad()
        values = self._get_values(obs, self._get_action(obs, self.P), self.Q)
        p_loss = - values.mean()
        p_loss.backward()
        if self.clip_val is not None:
            torch.nn.utils.clip_grad_norm_(self.P.parameters(), self.clip_val)
        self.p_optim.step()

        for w in self.Q.parameters(): # freeze Q parameters
            w.requires_grad = True

        # update target
        with torch.no_grad():
            for w, target_w in zip(self.Q.parameters(), self.target_Q.parameters()):
                target_w.data.mul_(self.update_decay)
                target_w.data.add_((1 - self.update_decay) * w.data)
            for w, target_w in zip(self.P.parameters(), self.target_P.parameters()):
                target_w.data.mul_(self.update_decay)
                target_w.data.add_((1 - self.update_decay) * w.data)

    def act(self, observation):
        with torch.no_grad():
            obs = torch.tensor(
                self.featureExtractor.getFeatures(observation),
                dtype=torch.float32)
            action = (self.act_range * self.P(obs))[0].numpy()
            if not self.test:
                action += self.noise_scale * np.random.randn(*action.shape)
                action = np.clip(action, self.act_low, self.act_high)
        return action


## Runner ##
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='pendulum',
        help="Runner environment, either 'pendulum', 'cartpole' or 'lunar'")
    parser.add_argument('-per', action='store_true',
        help='If set, Prioritized Experience Replay is used.')
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
    name = 'ddpg{}'.format('_per' if args.per else '')
    outdir = "./XP/" + config["env"] + "/" + name + "_" + tstart

    # seed rngs
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    obs = env.reset()

    agent = DDPG(env, config)

    memory = Memory(config.get('memSize'), prior=False)

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

            if timestep < config.get('startAfter', START_AFTER):
                action = env.action_space.sample()
            else:
                action = agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            ep_steps += 1

            if not agent.test:
                truncated = info.get('TimeLimit.truncated', False)
                memory.store((obs, action, reward, next_obs, done and not truncated))


                if (memory.nentities >= config.get('warmUp', WARM_UP) and
                   timestep % config['updateEvery'] == 0):
                   for _ in range(config['updateSteps']):
                        _, _, batch = memory.sample(config['batchSize'])
                        agent.learn(batch)
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