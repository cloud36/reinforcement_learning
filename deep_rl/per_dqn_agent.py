import numpy as np
import random
from collections import namedtuple, deque
from recordtype import recordtype

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


###
'''
NEED TO UPDATE IMPORTANCE SAMPLE WEIGHTS 
'''
###

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, priority_eps=.1):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.priority_eps = priority_eps

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        #self.criterion = nn.MSELoss()
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done, self.priority_eps)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
            
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, errs = experiences
        
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        
        Q_expected_is = Q_expected * (( 1 / errs)*(1/errs.shape[0])) #**beta
        Q_targets_is  = Q_targets  * (( 1 / errs)*(1/errs.shape[0])) #**beta 
        # Compute loss
        loss = F.mse_loss(Q_expected_is, Q_targets_is)
 
        # append loss to replay samples 
        td_errors = (Q_expected - Q_targets)**2
        for e in range(len(experiences)):
            #print(td_errors[e].detach().numpy())
            #print(td_errors[e][0].detach().numpy())
            self.memory.add(states[e], actions[e], rewards[e], next_states[e], dones[e], td_errors[e][0].detach().numpy())
            #experiences[e] = experiences[e]._replace(td_error = td_errors[e]+self.priority_eps)    
            
        #self.memory.add(experiences)
        #_ = [self.memory.append(e) for e in experiences]
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, priority_eps=.1, priority_alpha=.5):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "td_error"])
        self.seed = random.seed(seed)
        self.priority_eps = priority_eps
        self.priority_alpha = priority_alpha
    
    def add(self, state, action, reward, next_state, done, priority_eps):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, priority_eps)
        self.memory.append(e)
        
    def batch_add(self, experiences):
        self.memory.append(experiences)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
                          
        # update priorities 
        errorsT = [(e.td_error+self.priority_eps)**self.priority_alpha for e in self.memory if e is not None]

        #print(type(errorsT))
        #print(errorsT)
        sample_prob = np.array(errorsT) / sum(errorsT) #  np.array(errors) / sum(errors) 
        #print(sum(sample_prob))
        # sample based on priority 
        idx = np.random.choice(len(self.memory), self.batch_size, p = sample_prob)
        experiences = deque([self.memory[i] for i in idx])

        # remove sampled experiences from memory. They will be added back with new td_errors.
        idxs = list(set(idx))
        idxs.sort()
        c=0
        for ix in idxs:
            del self.memory[ix-c]
            c+=1
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
      
        err = torch.from_numpy(np.vstack([e.td_error for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones, err)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
