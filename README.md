# Simple test for using MCTS for finding failure stats in simulator

Based on the work done by Lee, Ritchie et al. in "Adaptive Stress Testing: Finding Likely Failure Events with Reinforcement Learning". Used as a proof of concept before doing adaptive stress testing of more advanced agents and simulators.

Simulator is dumb boat going in straight line, while disturbance is steerable boat trying to crash in to the streerable boat. The boats operate in a 100x100 world with a step length of 5.

The MCTS has no knowledge of the workings of the sim-world, and can only pass in a random action-seed and is returned a set of variables to estimate the reward of the current state of the sim-world.

## MCTS

The monte carlo tree search is quite simple, where one loop searches down through the tree, finds a leaf node and does a single rollout from that node. This project is based on pseudocode on page 13 in Lee, Ritchie et al., "Adaptive Stress Testing: Finding Likely Failure Events with Reinforcement Learning"

## Results and effectivity of the model

Using MCTS to search for failure states seems to highly increase the chances of finding them. Through testing, over 500,000 random paths were generated of which none resulted in a crash, while about 70-80% of searches using MCTS with 4000 simulations/rollouts resulted in multiple failure states.

### Figures

----

![Image of a generated, most likely crash path](figs/crashPath.png?raw=true "Crash Path")

An example of a generated, most likely crash path. Steerable boat starts at x=17, y=0. Connected dots are positions in the same time step, colors indicate distances between boats.

----

![Image of a route with random seed actions](figs/randomPath.png?raw=true "Random path")

An example of a random path without using MCTS.

----

![Reward over time](figs/scoreOverTime.png?raw=true "Reward over time")

An example of the score of each simulation as it develops over time. First crash is found after 717 simulations.
