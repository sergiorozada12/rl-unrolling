import gymnasium as gym
import torch


class CliffWalkingEnv:
    def __init__(self):
        self.env = gym.make("CliffWalking-v0").unwrapped
        self.nS = self.env.observation_space.n
        self.nA = self.env.action_space.n

        goal_state = 47
        for a in range(self.nA):
            self.env.P[goal_state][a] = [(1.0, goal_state, 0.0, True)]

        self.P = torch.zeros(self.nS * self.nA, self.nS)
        self.r = torch.zeros(self.nS * self.nA)
        for s in range(self.nS):
            for a in range(self.nA):
                idx = s * self.nA + a
                for prob, next_s, reward, done in self.env.P[s][a]:
                    self.P[idx, next_s] += prob
                    self.r[idx] += prob * reward

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
