import sys
import gym
import numpy as np
from collections import defaultdict

def mc_prediction_q(env, num_episodes, generate_episode, gamma=1.0):
    # initialize empty dictionaries of arrays
    returns_sum = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        
        episode = generate_episode(env)
        for t in range(len(episode)):
            state  = episode[t][0]
            action = episode[t][1]
            reward = episode[t][2]
            if state not in N:
                N[state][action] = 1
            if state in N:
                N[state][action] += 1
                
            discounted_sum_rewards = reward + sum([ii[2]*gamma for ii in episode[t+1:]])
            returns_sum[state][action] += discounted_sum_rewards
        
        # normalize
        for s in N:
            for i in range(env.action_space.n):
                Q[s][i] =  returns_sum[s][i] / N[s][i]
                
    return Q

def mc_control(env, num_episodes, alpha, gamma=1.0):
    nA = env.action_space.n
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(nA))
    # loop over episodes
    for i_episode in range(1, num_episodes+1):

        # monitor progress
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
            
        policy = dict([(state, np.argmax(value)) for state, value in Q.items()])
        state = env.reset()
        episode = []
        
        # generate episode 
        while True:
            if state not in policy:
                action = np.random.choice(nA)
            else:
                action = policy[state]
                
            next_state, reward, done, info = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            if done:
                break
        #print(episode)
        # update q-function
        visited = []
        for t in range(len(episode)):
            state  = episode[t][0]
            action = episode[t][1]
            reward = episode[t][2]

            # for first-visit MC Control 
            if state not in visited:
                # calculate G
                G = reward + sum([ii[2]*gamma for ii in episode[t+1:]])
                current_value = Q[state][action]
                Q[state][action] = current_value + alpha*(G-current_value)
                visited.append(state)
                
    return policy, Q
