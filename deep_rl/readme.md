# Value Based Deep Reinforcement Learning

## Overview 

#### DQN Algorithm

The first and most well known deep reinforcement learning algorithm is a DQN. This reinforcement learning algorithm was made popular in February 2015, by researchers at DeepMind. It combines deep learning and reinforcement learning to achieve super-human level control on various Atari games from raw pixels.

DQN's extend traditional Q-Learning by using function approximation represented by a deep neural network, hence the name DQN. In addition to using a DNN to represent the state-action space, two other important modifications were made: experience replay and use of a target network. Both of these modifications were to help overcome what is known as the "deadly triad" coined by Richard Sutton. The deadly triad occurs when three elements are combined: bootstrapping, function approximation and off-policy learning. DQN has all three of these ingredients.

- Experience Replay: Not only does experience replay help us decorrelate state-action pairs, which helps prevent the learning from oscillating or diverging, but it can also help the learning algorithm converge faster and make more efficient use of its past experience.

- Target Network: Another important addition is the use of a target network. Without the target network, a single DNN would be used to estimate both the current state value and the expected next state value. The problem arises when updating the parameters of the DNN. With the updated parameters, both the current state estimate and next state estimate may change, which in turn would make minimizing the loss function difficult -- think of chasing a moving target.

Original Paper: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

### Double DQN Algorithm

Double DQN is an improvement on the original DQN and it attempts to address overly optimistic q-function that is observed in DQN's. It is easy for a DQN to overestiamte the q-function because of the argmax operator in the TD Error update. To handle this Double DQN makes a simple modification by having another DNN estimate the value of the argmax operator i.e. we use DNN-1 to select the actions in th argmax operator and use another DNN (DNN-2) to estimate the value. To implement this we simple use our target and local DNNs.

Read more about Double DQN's here: https://arxiv.org/pdf/1509.06461.pdf

- Note: You may be wondering why the title of this page is "value-based" drl. It can be said there are two general types of reinforcement learning: value-based and policy based. Value-based method aim to understand the value of action an action in any given state, once this is known, an optimal policy can be derived just my taking greedy actions. Policy based method on the other hand skip the state-action value appoximation and directly look for an optimal policy. 
