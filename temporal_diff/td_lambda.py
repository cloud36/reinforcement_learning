import numpy as np
from collections import defaultdict

class TD_Lambda_Agent:
    def __init__(self, nA=6, eps=.1, alpha=.1, gamma=.9, lamb=.4):
        """ Initialize agent.
        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: [np.zeros(self.nA), 0])
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.lamb = lamb
        self.total_actions = 0 

        
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
        #self.eps * .99
        self.lamb *.99
        action = np.argmax(self.Q[state][0])

        if np.random.random() < self.eps:
            return np.random.choice(self.nA)
        else:
            return action 


    def step(self, state, action, reward, next_state, done, episode):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        # Update eligibilities
        for s, v in self.Q.items():
            v[1] *= self.lamb * self.gamma
        self.Q[state][1] = 1
        
        next_value = max(self.Q[next_state][0])

        td_error = reward + self.gamma * next_value - self.Q[state][0][action]
        for state, value in self.Q.items():
            value[0][action] = value[0][action] + self.alpha * td_error * value[1]
        #if episode == 100:
        #    for k, v in self.Q.items():
        #        return self.Q, state, action, reward, next_state
        #        break
