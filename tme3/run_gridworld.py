from collections import defaultdict
import datetime
import json
import os
import pickle

import matplotlib
matplotlib.use('TkAgg')
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
from numpy.random import default_rng


FPS = 0.0001
AGENTS = ('random', 'q-learning', 'sarsa', 'dyna-q')


def get_epsilon_greedy_explorer(rng, epsilon):
    def explorer(episode, Q):
        if rng.random() < epsilon:
            return rng.integers(Q.shape[0])
        return np.argmax(Q)
    return explorer


def get_decay_epsilon_greedy_explorer(rng, epsilon, total_episodes):
    def explorer(episode, Q):
        decayed_epsilon = epsilon * (total_episodes - episode) / total_episodes
        if rng.random() < decayed_epsilon:
            return rng.integers(Q.shape[0])
        return np.argmax(Q)
    return explorer


class BaseAgent:
    """Base agent for tabular Q-Learning"""
    def __init__(self, action_space, states, gamma=0.999, explorer=None):
        self.action_space = action_space
        self.states = states # state str -> state index
        self.gamma = gamma # MDP discount rate
        self.explorer = explorer # exploring strategy
        if explorer is None:
            self.explorer = lambda Q: np.argmax(Q) # default to greedy (no exploration)

        # Initialize Q-values
        self.Q = np.zeros((len(states), action_space.n))

    def get_state_index(self, state):
        return self.states[gridworld.GridworldEnv.state2str(state)]

    def act(self, episode, observation):
        return self.explorer(episode, self.Q[self.get_state_index(observation)])

    def act_greedy(self, observation):
        return np.argmax(self.Q[self.get_state_index(observation)])

    def learn(self, transition):
        pass


class Random(BaseAgent):
    """Random baseline agent."""
    def __init__(self, action_space, states, **kwargs):
        super(Random, self).__init__(action_space, states, **kwargs)

    def act(self, episode, observation):
        return self.action_space.sample()

    def act_greedy(self, observation):
        return self.act(0, observation)


class QLearning(BaseAgent):
    """Vanilla Q-Learning agent."""
    def __init__(self, action_space, states, lr=0.01, lam=0.0, **kwargs):
        super(QLearning, self).__init__(action_space, states, **kwargs)
        self.lr = lr # Q-value update learning rate
        self.lam = lam # Eligibility traces weight parameter
        if lam > 0:
            self.e = defaultdict(float) # State-Action elibility

    def learn(self, transition):
        obs, act, next_obs, reward, done = transition
        s, next_s = self.get_state_index(obs), self.get_state_index(next_obs)

        # Get TD error
        delta = 0.
        if done:
            delta = reward - self.Q[s, act]
        else:
            delta = reward + reward + self.gamma * np.max(self.Q[next_s]) - self.Q[s, act]

        # Update Q-values (use eligibilty traces if lambda > 0)
        if self.lam <= 0:
            self.Q[s, act] = self.Q[s, act] + self.lr * delta
        else:
            self.e[(s, act)] += 1
            for s_prime, a_prime in self.e.keys():
                self.Q[s_prime, a_prime] = self.Q[s_prime, a_prime] + self.lr * delta * self.e[(s_prime, a_prime)]
                self.e[(s_prime, a_prime)] = self.lam * self.gamma * self.e[(s_prime, a_prime)] + int(s_prime == s)
            if done:
                self.e.clear() # Clear elibility traces after episode


class SARSA(BaseAgent):
    """Tabular SARSA agent."""
    def __init__(self, action_space, states, lr=0.01, lam=0.0, **kwargs):
        super(SARSA, self).__init__(action_space, states, **kwargs)
        self.lr = lr # Q-value update learning rate
        self.lam = lam # Eligibility traces weight parameter
        if lam > 0:
            self.e = defaultdict(float) # State-Action elibility

    def learn(self, transition, next_act):
        obs, act, next_obs, reward, done = transition
        s, next_s = self.get_state_index(obs), self.get_state_index(next_obs)

        # Get error using next action from policy
        delta = 0.
        if done:
            delta = reward - self.Q[s, act]
        else:
            delta = reward + reward + self.gamma * self.Q[next_s, next_act] - self.Q[s, act]

        # Update Q-values (use eligibilty traces if lambda > 0)
        if self.lam <= 0:
            self.Q[s, act] = self.Q[s, act] + self.lr * delta
        else:
            self.e[(s, act)] += 1
            for s_prime, a_prime in self.e.keys():
                self.Q[s_prime, a_prime] = self.Q[s_prime, a_prime] + self.lr * delta * self.e[(s_prime, a_prime)]
                self.e[(s_prime, a_prime)] = self.lam * self.gamma * self.e[(s_prime, a_prime)] + int(s_prime == s)
            if done:
                self.e.clear() # Clear elibility traces after episode


class DynaQ(BaseAgent):
    """Tabular hybrid model-based/value-based Dyna-Q agent."""
    def __init__(self, n_planning_steps, action_space, states, lr=0.01, mlr=0.01, rng=None, **kwargs):
        super(DynaQ, self).__init__(action_space, states, **kwargs)
        self.n_planning_steps = n_planning_steps # Number of planning steps using the learned model
        self.lr = lr # Q-value learning rate
        self.mlr = mlr # Model learning rate

        self.seen = set() # Stores seen state-action pairs
        self.seen_list = [] # A list containing state-action pairs for faster sampling
        self.rng = rng  # For sampling
        if self.rng is None:
            self.rng = np.random.default_rng()

        self.P = np.zeros((len(states), action_space.n, len(states))) # P[s, a, s'] := P(s'| s, a)
        self.R = np.zeros((len(states), action_space.n, len(states))) # R[s, a, s'] := reward of transition (s, a, s')

    def learn(self, transition):
        obs, act, next_obs, reward, done = transition
        s, next_s = self.get_state_index(obs), self.get_state_index(next_obs)

        # Update Q-value directly
        if done:
            self.Q[s, act] = self.Q[s, act] + self.lr * (reward - self.Q[s, act])
        else:
            self.Q[s, act] = self.Q[s, act] + self.lr * (reward + self.gamma * np.max(self.Q[next_s]) - self.Q[s, act])

        # Update model
        self.R[s, act, next_s] = self.R[s, act, next_s] + self.mlr * (reward - self.R[s, act, next_s])
        target = np.zeros(len(self.states))
        target[next_s] = 1.
        self.P[s, act] = self.P[s, act] + self.mlr * (target - self.P[s, act])

        # Compute planning and update Q-values
        if len(self.seen_list) >= self.n_planning_steps:
            samples = self.rng.choice(self.seen_list, size=(self.n_planning_steps,))
            for s, a in samples:
                self.Q[s, a] = (self.Q[s, a] +
                    self.lr * (np.sum(self.P[s, a] * (self.R[s, a] + self.gamma * np.max(self.Q, axis=-1))) - self.Q[s, a]))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', '-e', type=int, default=1000,
        help='Number of training episodes. Defaults to 1000.')
    parser.add_argument('--test-episodes', '-te', type=int, default=100,
        help='Number of test epsodes. Defaults to 100.')
    parser.add_argument('--render-freq', '-rf', type=int, default=-1,
        help='Frequency of rendered episodes. Defaults to -1 (no rendering).')
    parser.add_argument('--print-freq', '-pf', type=int, default=1,
        help='Frequency of printing epsisode metrics. Defaults to 1 (once per epsisode).')
    parser.add_argument('--seed', type=int, default=0,
        help='Random seed for PRNGs. Defaults to 0.')
    parser.add_argument('--gamma', type=float, default=0.999,
        help='Reward discount factor. Defaults to 0.999.')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.01,
        help='Learning rate for Q-value updates. Defaults to 0.1.')
    parser.add_argument('--model-learning-rate', '-mlr', type=float, default=0.01,
        help='Learning rate for updating the model for Dyna-Q. Defaults tp=o 0.01.')
    parser.add_argument('--epsilon', type=float, default=0.1,
        help='Rate for epsilon greedy policy. Defaults to 0.1.')
    parser.add_argument('--lam', type=float, default=0.0,
        help='Eligibility traces exponential weight parameter. Defaults to 0.')
    parser.add_argument('--logdir', type=str, default='./logs/',
        help='Log directory to save metrics and hyperparameters.')
    parser.add_argument('--agent', type=str, default='random',
        help="Type of agent to run. Can be either 'random', 'q-learning' and 'sarsa', defaulting to random.")
    parser.add_argument('--n-planning-steps', type=int, default=5,
        help='Number of planning steps for Dyna-Q. Defaults to 5.')
    parser.add_argument('-decay', action='store_true',
        help='If set, the exploration parameter will be decayed linearly during training.')
    parser.add_argument('-use-gamma', action='store_true',
        help='If set, the discount parmameter will be used to calculate the cumulative reward.')
    args = parser.parse_args()

    # Get current time
    current_time = datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")

    # Create and initialize environment
    env = gym.make('gridworld-v0')
    env.setPlan('gridworldPlans/plan0.txt', {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
    env.seed(0)

    # Print some information of the environment
    print('Action Space: {}'.format(env.action_space))
    states, _ = env.getMDP()  # gets states : state str -> state index
    print('Nb of states: {}'.format(len(states)))  # number of states

    # Get exploration policy
    rng = default_rng(args.seed)
    if args.decay:
        explorer = get_decay_epsilon_greedy_explorer(rng, args.epsilon, args.episodes)
    else:
        explorer = get_epsilon_greedy_explorer(rng, args.epsilon)

    # Build agent
    agent = None
    if args.agent in AGENTS:
        if args.agent == 'random':
            agent = Random(env.action_space, states)
        elif args.agent == 'q-learning' or args.agent == 'sarsa':
            agent_class = QLearning if args.agent == 'q-learning' else SARSA
            agent = agent_class(env.action_space, states,
                gamma=args.gamma, lr=args.learning_rate, lam=args.lam,
                explorer=explorer)
        elif args.agent == 'dyna-q':
            agent = DynaQ(args.n_planning_steps, env.action_space,
                states, gamma=args.gamma, lr=args.learning_rate,
                mlr=args.model_learning_rate, explorer=explorer)
    else:
        raise ValueError('Agent {} not supported.'.format(args.agent))

    # Train agent
    reward_sums = []
    episode_lengths = []
    for i in range(args.episodes):
        env.verbose = (args.render_freq > 0 and
            i % args.render_freq == 0 and i > 0)  # render every render_freq episodes
        if env.verbose:
            env.render(FPS)

        step = 0
        reward_sum = 0
        discount = 1.

        obs = env.reset().copy()
        act = agent.act(i, obs) # choose starting action
        while True:
            # Take action
            next_obs, reward, done, _ = env.step(act)

            # Plan next action and learn
            transition = (obs, act, next_obs, reward, done)
            act = agent.act(i, next_obs)
            if args.agent == 'sarsa':
                agent.learn(transition, act)
            else:
                agent.learn(transition)

            # Update observation
            obs = next_obs.copy()

            # Update metrics
            if args.use_gamma:
                discount *= args.gamma
                reward_sum = discount * reward
            else:
                reward_sum += reward
            step += 1

            if env.verbose:
                env.render(FPS)
            if done:
                reward_sums.append(reward_sum)
                episode_lengths.append(step)
                if (args.print_freq > 0 and
                    i % args.print_freq == 0 and i > 0):
                    print('Episode {}: reward_sum={}, length={}'.format(i, reward_sum, step))
                break

    # Test agent
    test_reward_sums = []
    test_episode_lengths = []
    for i in range(args.test_episodes):
        obs = env.reset()

        step = 0
        reward_sum = 0
        discount = 1.
        while True:
            # Take action
            act = agent.act_greedy(obs)
            obs, reward, done, _ = env.step(act)

            # Update metrics
            if args.use_gamma:
                discount *= args.gamma
                reward_sum = discount * reward
            else:
                reward_sum += reward
            step += 1

            if done:
                test_reward_sums.append(reward_sum)
                test_episode_lengths.append(step)
                break

    # Print statistics
    reward_sums = np.array(reward_sums)
    episode_lengths = np.array(episode_lengths)
    print('Train Cumulative Reward: {} ± {}'.format(reward_sums.mean(), reward_sums.std()))
    print('Train Episode Lengths: {} ± {}'.format(episode_lengths.mean(), episode_lengths.std()))

    test_reward_sums = np.array(test_reward_sums)
    test_episode_lengths = np.array(test_episode_lengths)
    print('Test Cumulative Reward: {} ± {}'.format(test_reward_sums.mean(), test_reward_sums.std()))
    print('Test Episode Lengths: {} ± {}'.format(test_episode_lengths.mean(), test_episode_lengths.std()))
    env.close()

    # Save log
    logs_dir = os.path.join(args.logdir, "{}_{}/".format(args.agent, current_time))
    os.makedirs(logs_dir, exist_ok=True)
    with open(os.path.join(logs_dir, 'hyperparams.json'), 'w') as f:
        json.dump(vars(args), f)
    with open(os.path.join(logs_dir, 'metrics.pkl'), 'wb') as f:
        pickle.dump({
            'train_reward_sums': reward_sums,
            'train_episode_lengths': episode_lengths,
            'test_reward_sums': test_reward_sums,
            'test_episode_lengths': test_episode_lengths,
        }, f)
    print('Saved hyperparameters and metrics to {}.'.format(logs_dir))
    print('Done!')
