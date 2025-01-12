import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, epsilon=1.0, alpha=0.1):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = 1

    def select_action(self, state, i_episode):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        self.epsilon = self.epsilon / i_episode
        if np.random.random() > self.epsilon:
            return np.argmax(self.Q[state])
        else:
            return np.random.choice(np.arange(self.nA))

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        if done:
            self.Q[state][action] = self.Q[state][action] + self.alpha * (
                reward - self.Q[state][action]
            )
        else:
            G_estimate = reward + self.gamma * np.max(self.Q[next_state])
            self.Q[state][action] = self.Q[state][action] + self.alpha * (
                    G_estimate - self.Q[state][action]
                )

