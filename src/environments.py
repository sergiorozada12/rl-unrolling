import torch
import gymnasium as gym
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete


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


class MirroredCliffWalkingEnv:
    def __init__(self):
        self.env = gym.make("CliffWalking-v0").unwrapped
        self.nS = self.env.observation_space.n
        self.nA = self.env.action_space.n

        self.start_state = 0        # (0, 0)
        self.goal_state = 11        # (0, 11)

        for a in range(self.nA):
            self.env.P[self.goal_state][a] = [(1.0, self.goal_state, 0.0, True)]

        for col in range(1, 11):
            cliff_state = col  # row 0, col = col
            for a in range(self.nA):
                self.env.P[cliff_state][a] = [(1.0, self.start_state, -100.0, False)]

        self.P = torch.zeros(self.nS * self.nA, self.nS)
        self.r = torch.zeros(self.nS * self.nA)
        for s in range(self.nS):
            for a in range(self.nA):
                idx = s * self.nA + a
                for prob, next_s, reward, done in self.env.P[s][a]:
                    self.P[idx, next_s] += prob
                    self.r[idx] += prob * reward

    def reset(self):
        self.env.s = self.start_state
        return self.env.s

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


class HighResCliffWalkingEnv(Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, rows=8, cols=24):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.nS = rows * cols
        self.nA = 4  # 0: right, 1: down, 2: left, 3: up

        self.start_state = (rows - 1) * cols + 0        # bottom-left
        self.goal_state = (rows - 1) * cols + (cols - 1)  # bottom-right

        self.state = self.start_state
        self.observation_space = Discrete(self.nS)
        self.action_space = Discrete(self.nA)

        self._build_env()
        self._build_model()

    def _build_env(self):
        self.P_env = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        for r in range(self.rows):
            for c in range(self.cols):
                s = r * self.cols + c
                for a in range(self.nA):
                    if s == self.goal_state:
                        self.P_env[s][a] = [(1.0, s, 0.0, True)]
                        continue

                    next_r, next_c = r, c
                    if a == 0 and r > 0:              # Up
                        next_r -= 1
                    elif a == 1 and c < self.cols - 1:  # Right
                        next_c += 1
                    elif a == 2 and r < self.rows - 1:  # Down
                        next_r += 1
                    elif a == 3 and c > 0:              # Left
                        next_c -= 1

                    next_s = next_r * self.cols + next_c

                    # Cliff zone
                    is_cliff = (r == self.rows - 1 and 1 <= c <= self.cols - 2)

                    if is_cliff:
                        self.P_env[s][a] = [(1.0, self.start_state, -100.0, False)]
                    else:
                        done = next_s == self.goal_state
                        reward = 0.0 if done else -1.0
                        self.P_env[s][a] = [(1.0, next_s, reward, done)]

    def _build_model(self):
        self.P = torch.zeros(self.nS * self.nA, self.nS)
        self.r = torch.zeros(self.nS * self.nA)
        for s in range(self.nS):
            for a in range(self.nA):
                idx = s * self.nA + a
                for prob, next_s, reward, done in self.P_env[s][a]:
                    self.P[idx, next_s] += prob
                    self.r[idx] += prob * reward

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.start_state
        return self.state, {}  # gymnasium format: (obs, info)

    def step(self, action):
        transitions = self.P_env[self.state][action]
        prob, next_state, reward, terminated = transitions[0]
        truncated = False
        self.state = next_state
        return next_state, reward, terminated, truncated, {}

    def render(self):
        grid = np.full((self.rows, self.cols), '.', dtype=str)
        for c in range(1, self.cols - 1):
            grid[self.rows - 1, c] = 'C'
        grid[self.rows - 1, 0] = 'S'
        grid[self.rows - 1, self.cols - 1] = 'G'
        r, c = divmod(self.state, self.cols)
        grid[r, c] = 'A'
        print('\n'.join(''.join(row) for row in grid))

    def close(self):
        pass
