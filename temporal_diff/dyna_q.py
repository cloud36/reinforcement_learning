import numpy as np
from collections import defaultdict

class Dyna_Q:
    def __init__(self, nA=6, eps=.1, alpha=.1, gamma=.9, n=50):
        """ Initialize agent.
        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.total_actions = 0 
        self.n = n
        # model of universe
        self.history = []

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        # according to e-greedy policy 
        self.total_actions += 1
        self.eps = 1 / self.total_actions

        action = np.argmax(self.Q[state])

        if np.random.random() < self.eps:
            return np.random.choice(self.nA)
        else:
            return action 


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
        
        next_value = max(self.Q[next_state])

        # td_error = reward + self.gamma * next_value - self.Q[state][0][action]
        td_error = reward + self.gamma * next_value - self.Q[state][action]
        self.Q[state][action] = self.Q[state][action] + self.alpha * td_error 
        
        self.history.append((state, action, reward, next_state, done))
    
    def simulate(self):
        for i in range(self.n):
            state, action, reward, next_state, done = self.history[np.random.randint(0,high=len(self.history))]
            self.step(state, action, reward, next_state, done)

        

