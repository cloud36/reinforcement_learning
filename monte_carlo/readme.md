# Monte Carlo Methods 

## Monte Carlo Prediction

Monte Carlo methods are some of the simplest approaches in reinforcement learning. They work by observing an entire espisode and simple taking the average discounted reward for each state and recording that as the states value. For example, imagine the agent takes the follwoing trafectory below and observes the following rewards:

    (State_2, Reward:5) --> (State_5, Reward:2) --> (State_7, Reward:-1) --> (State_9, terminal state)

Furthermore, let us assume the discount rate is .9. Then the following values are assigned to each state.

    State_2 = 5 + .9*2+ .81*-1 = 5.99
    State_5 = 2 + .9*-1        = 1.1
    State_7 = -1               = -1

This is what is known as Monte Carlo prediction in that it predicts the value of each state visited by the agent. As the agent experiences more epsidoes the law of large numbers kicks in and the predicted value will converge to the theoretical mean. 

## Monte Carlo Control

Monte Carlo control is a simple modification to MC prediction and there are two basic approachs 1) if the model of the world is knonw and 2) if the model of the world is not known.

1. If there is a known model of the work an agent can act greedily by selecting the action that takes it to the state with the highest value.

2. If a model of the world is not known. An agent must keep track of action values for each state i.e the average discounted reward for taking a specific action in each state. Once this table is built an agent can act greedily with respect to the action-values.


## Some Facts about Monte Carlo Methods.

1. Monte Carlo methods are known to have high bias, but no variance. 
