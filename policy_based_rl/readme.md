# Policy Based Reinforcement Learning

## Introduction 

In policy based reinforcement learning we aim to directly find an optimal policy and skip the step of estimated a state-action value as with value-based reinforcement learning. Linked below are some additional resources that go into further detail.

    - Link 1
    - Link 2

### General Purpose Optimization Methods

At the most basic level, once can use any generic optimization algorithm to find an optimial policy. In this repository, we highlight several general purpose optimization algorithms that can be used to directly find an optimal policy such as hill-climbing, hill-climbing with adaptive noise scaling, simulated annealing, cross-entropy and genetic algorithms. 

### Gradient Based Methods i.e Policy Gradient Methods

Unlike the methods dicssed above, gradient based methods, as one can imagine, use the graident of a policy to find the optimal policy. 

- REINFORCE
    - This algorithm works by directly looking for the optimal policy by collecting a number of sample trajectories (state, action, reward) then computing the gradient and optimizing via gradient ascent. 
