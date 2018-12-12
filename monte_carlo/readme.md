# Monte Carlo Methods 

## Overview 

Monte Carlo methods are some of the simplest approaches in reinforcement learning. They work by observing an entire espisode and simple take the average discounted reward for each state and recording that as the states value. For example, imagine takes the trafectory below and observes the following rewards:

(State, Reward)
(State_2, 5)
(State_5, 2)
(State_7, -1)
(State_9, terminal state)

Let's also assume the discount rate is .9. Then the following values are assigned to each state.

State_2 = 5 + .9*2+ .9*-1 = 5.9
State_5 = 2 + .9*-1       = 1.1
State_7 = -1              = -1

This is what is known as Monte Carlo prediction in that it predicts the value of each state visited by the agent. As the agent experiences more epsidoes the law of large numbers kicks in and the predicted value will converge to the theoretical mean. 

*Monte Carlo Prediction* 

## Insights

Monte Carlo methods are known to have high variance, but no bias. This is an important consideration to take into account when ...
