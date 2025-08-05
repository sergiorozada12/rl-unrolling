"""Environment implementations for BellNet.

This module contains custom environment implementations including
CliffWalking and MirroredCliffWalking environments with modified
reward structures and dynamics.
"""

import gymnasium as gym
import torch


# Action constants
ACT_UP = 0
ACT_RIGHT = 1  
ACT_DOWN = 2
ACT_LEFT = 3


class CliffWalkingEnv:
    """Standard CliffWalking environment with modified goal state.
    
    This environment wraps the OpenAI Gym CliffWalking environment
    and modifies the goal state to be absorbing with zero reward.
    
    Attributes:
        nS: Number of states
        nA: Number of actions
        P: Transition probability tensor of shape (nS * nA, nS)
        r: Reward tensor of shape (nS * nA,)
    """
    def __init__(self) -> None:
        self.env = gym.make("CliffWalking-v1", render_mode="rgb_array").unwrapped
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

    def reset(self) -> int:
        """Reset environment to initial state.
        
        Returns:
            Initial state
        """
        return self.env.reset()

    def step(self, action: int) -> tuple:
        """Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        return self.env.step(action)

    def render(self):
        """Render the environment.
        
        Returns:
            RGB array of the rendered environment
        """
        return self.env.render()

    def close(self) -> None:
        """Close the environment."""
        self.env.close()


class MirroredCliffWalkingEnv:
    """Modified CliffWalking environment with mirrored cliff.
    
    This environment modifies the standard CliffWalking by:
    1. Removing the cliff from the bottom row
    2. Adding a cliff to the top row (mirrored)
    3. Changing start/goal positions accordingly
    
    The cliff is now in the top row (states 1-10) instead of bottom row.
    Start state is at (0,0) and goal state is at (0,11).
    
    Attributes:
        nS: Number of states
        nA: Number of actions
        P: Transition probability tensor of shape (nS * nA, nS)
        r: Reward tensor of shape (nS * nA,)
        start_state: Starting state (top-left)
        goal_state: Goal state (top-right)
    """
    def __init__(self) -> None:
        self.env = gym.make("CliffWalking-v1", render_mode="rgb_array").unwrapped
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

        
    def reset(self) -> int:
        """Reset environment to initial state.
        
        Returns:
            Initial state
        """
        self.env.s = self.start_state
        self.env.lastaction = None  # inicializar
        return self.env.s

    def step(self, action: int) -> tuple:
        """Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        self.env.lastaction = action
        return self.env.step(action)

    def render(self):
        """Render the environment.
        
        Returns:
            RGB array of the rendered environment
        """
        return self.env.render()

    def close(self) -> None:
        """Close the environment."""
        self.env.close()
