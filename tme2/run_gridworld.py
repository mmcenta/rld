import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy


FPS = 0.0001
MAX_POLICY_ITERATION = 10000
MAX_POLICY_EVALUATION = 10000
MAX_VALUE_ITERATION = 10000


class RandomAgent:
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        return self.action_space.sample()


class BaseAgent:
    "Base Agent for Policy Iteration and Value Iteration"
    def __init__(self, action_space, states, mdp, gamma=0.999):
        self.action_space = action_space
        self.states = states
        self.gamma = gamma

        # Initialize values and policy
        self.V = np.random.normal(size=(len(states)))
        self.policy = np.zeros(len(states))
        for s in range(len(states)):
            self.policy[s] = action_space.sample()
        self.policy = self.policy.astype(np.int32)

        # Compute transition matrix and reward matrix
        self.R = np.zeros((len(states), action_space.n, len(states)))
        self.P = np.zeros((len(states), action_space.n, len(states)))
        seen = dict()
        for key in mdp.keys():
            s = states[key]
            for a in range(action_space.n):
                for t in mdp[key][a]:
                    p, next_key, r, _ = t
                    next_s = states[next_key]
                    self.P[s, a, next_s] += p # += intead of = because of repeated transitions
                    self.R[s, a, next_s] = r

    def act(self, observation):
        s = states[gridworld.GridworldEnv.state2str(observation)]
        return self.policy[s]

    def build(self):
        pass


class PIAgent(BaseAgent):
    """Policy Iteration Agent"""
    def __init__(self, action_space, states, mdp, gamma=0.999):
        super(PIAgent, self).__init__(action_space, states, mdp, gamma=gamma)

    def policy_evaluation(self):
        delta = 0
        for s in range(len(self.states)):
            a, v = self.policy[s], self.V[s]
            self.V[s] = np.sum(self.P[s, a, :] * (self.R[s, a, :] + self.gamma * self.V))
            delta = max(delta, np.abs(self.V[s] - v))
        return delta

    def policy_improvement(self):
        prev_policy = self.policy.copy()
        for s in range(len(self.states)):
            self.policy[s] = np.argmax(np.sum(self.P[s, :, :] * (self.R[s, :, :] + self.gamma * self.V), axis=-1))
        return np.all(self.policy == prev_policy)

    def build(self, threshold=0.01):
        done = False
        for _ in range(MAX_POLICY_ITERATION):
            # Policy Evaluation until convergence
            delta = 0
            for _ in range(MAX_POLICY_EVALUATION):
                delta = self.policy_evaluation()
                if delta < threshold:
                    break
            if delta >= threshold:
                print("Maximum number of policy evaluation iterations ({}) "
                      "reached.".format(MAX_POLICY_EVALUATION))

            # Policy Improvement
            done = self.policy_improvement()
            if done:
                break
        if not done:
            print("Maximum number of policy iterations ({}) "
                  "reached.".format(MAX_POLICY_ITERATION))


class VIAgent(BaseAgent):
    """Value Iteration Agent"""
    def __init__(self, action_space, states, mdp, gamma=0.999):
        super(VIAgent, self).__init__(action_space, states, mdp, gamma=gamma)
        self.mdp = mdp
        self.states = states

    def build(self, threshold=0.01):
        # Value Iteration
        delta = 0
        for _ in range(MAX_VALUE_ITERATION):
            delta = 0
            for s in range(len(self.states)):
                v = self.V[s]
                self.V[s] = np.max(np.sum(self.P[s, :, :] * (self.R[s, :, :] + self.gamma * self.V), axis=-1))
                delta = max(delta, np.abs(self.V[s] - v))
            if delta < threshold:
                break
        if delta >= threshold:
            print("Maximum number of policy iterations ({}) "
                  "reached.".format(MAX_VALUE_ITERATION))

        # Extract Policy
        for s in range(len(self.states)):
            self.policy[s] = np.argmax(np.sum(self.P[s, :, :] * (self.R[s, :, :] + self.gamma * self.V), axis=-1))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', '-e', type=int, default=1000,
        help="Number of training episodes. Defaults to 1000.")
    parser.add_argument('--render-freq', '-rf', type=int, default=-1,
        help="Frequency of rendered episodes. Defaults to -1 (no rendering).")
    parser.add_argument('--gamma', type=float, default=0.999,
        help="Reward discount factor. Defaults to 0.999.")
    parser.add_argument('--threshold', type=float, default=1e-12,
        help="Threshold for value convergence. Defaults to 1e-12.")
    parser.add_argument('--agent', type=str, default='random',
        help="Type of agent to run. Can be either 'random', 'pi' and 'vi', defaulting to random.")
    parser.add_argument('-use-gamma', action='store_true',
        help="If set, the discount parmameter will be used to calculate the cumulative reward.")
    args = parser.parse_args()

    # Create and initialize environment
    env = gym.make("gridworld-v0")
    env.setPlan("gridworldPlans/plan0.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
    env.seed(0)

    # Print some information of the environment
    print('Action Space: {}'.format(env.action_space))
    states, mdp = env.getMDP()  # gets the MDP, states : state -> state nb
    print("Nb of states: {}".format(len(states)))  # number of states

    # Build agent
    if args.agent == 'pi':
        agent = PIAgent(env.action_space, states, mdp, gamma=args.gamma)
        agent.build(threshold=args.threshold)
    elif args.agent == 'vi':
        agent = VIAgent(env.action_space, states, mdp, gamma=args.gamma)
        agent.build(threshold=args.threshold)
    else:
        agent = RandomAgent(env.action_space)

    # Run agent
    reward_sums = []
    episode_lengths = []
    for i in range(args.episodes):
        obs = env.reset()
        env.verbose = (args.render_freq > 0 and
            i % args.render_freq == 0 and i > 0)  # render every render_freq episodes
        if env.verbose:
            env.render(FPS)
        length = 0
        reward_sum = 0
        discount = 1.
        while True:
            # Take action
            act = agent.act(obs)
            obs, reward, done, _ = env.step(act)

            # Record rewards and length
            if args.use_gamma:
                discount *= args.gamma
                reward_sum = discount * reward
            else:
                reward_sum += reward
            length += 1

            if env.verbose:
                env.render(FPS)
            if done:
                reward_sums.append(reward_sum)
                episode_lengths.append(length)
                print("Episode {}: reward_sum={}, length={}".format(i, reward_sum, length))
                break

    # Print statistics
    reward_sums = np.array(reward_sums)
    ep_lengths = np.array(episode_lengths)
    print("Cumulative Reward: {} ± {}".format(reward_sums.mean(), reward_sums.std()))
    print("Episode Lengths: {} ± {}".format(ep_lengths.mean(), ep_lengths.std()))

    print("done")
    env.close()