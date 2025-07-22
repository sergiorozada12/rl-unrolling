import gymnasium as gym
import torch


ACT_UP = 0
ACT_RIGHT = 1
ACT_DOWN = 2
ACT_LEFT = 3


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
        self.nS = self.env.observation_space.n  # Number of states
        self.nA = self.env.action_space.n       # Number of actions

        cliff_states = [c for c in range(1, 11)]

        self.start_state = 0        # Top-left corner (0, 0)
        self.goal_state = 11        # Top-right corner (0, 11)

        # Make goal state absorbing with 0 reward
        for a in range(self.nA):
            self.env.P[self.goal_state][a] = [(1.0, self.goal_state, 0.0, True)]

        # Remove the original cliff in the bottom row (row 3, columns 1 to 10)
        ## Removing transitions from last row
        for col in range(0, 12):
            state = 3 * 12 + col  # row 3, col = col
            for a in range(self.nA):
                # Restore default transition: step with -1 reward and continue
                next_state = self.env.P[state][a][0][1]
                self.env.P[state][a] = [(1.0, next_state, -1.0, False)]
                
        ## Removing transitions from states above the cliff
        for col in range(1, 11):
            state_above_cliff = 2 * 12 + col  # row 2, col = col → states 25 to 34
            for i, (prob, next_s, reward, done) in enumerate(self.env.P[state_above_cliff][ACT_DOWN]):
                if reward == -100.0:
                    # Replace with normal step: move down with -1.0 reward and not done
                    next_state = next_s  # usually this is 3*12 + col
                    self.env.P[state_above_cliff][ACT_DOWN][i] = (1.0, next_state, -1.0, False)
        
        # Add mirrored cliff in the top row (row 0, columns 1 to 10)
        for col in range(0, 11):
            from_state = col
            for a in range(self.nA):
                for i, (prob, next_s, reward, done) in enumerate(self.env.P[from_state][a]):
                    if next_s in cliff_states:
                        self.env.P[from_state][a][i] = (1.0, self.start_state, -100.0, False)
                    elif next_s == self.goal_state:
                        self.env.P[from_state][a][i] = (1.0, self.goal_state, -1, True)

        # Modify transitions INTO the new cliff from below
        for col in range(1, 12):
            from_state = 1 * 12 + col  # row 1
            for i, (prob, next_s, reward, done) in enumerate(self.env.P[from_state][ACT_UP]):
                if next_s in cliff_states:
                    self.env.P[from_state][ACT_UP][i] = (1.0, self.start_state, -100.0, False)
                elif next_s == self.goal_state:
                    self.env.P[from_state][ACT_UP][i] = (1.0, self.goal_state, -1, True)

        ### PLOT FOR DEBUG ###
        # for row in range(4):
        #     for col in range(12):
        #         state = row * 12 + col
        #         for a in range(self.nA):
        #             print(f"State {state}, Action {a}: {self.env.P[state][a]}")

        # Convert transition dynamics and rewards into torch tensors
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
