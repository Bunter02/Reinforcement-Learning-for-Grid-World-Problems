# Reinforcement-Learning-for-Grid-World-Problems
Overview

The 2 versions are a normal python file(easier to run) and a juypter notebook fill (easier to see whats happening in each section)

This code implements a Q-learning reinforcement learning agent that learns to navigate through a 5x5 grid world environment. The environment includes obstacles, a special "jump" state that teleports the agent to another location with a reward, and a terminal state that the agent aims to reach.
Features

Complete implementation of a grid world environment
Q-learning agent with customizable learning parameters
Support for multiple learning rates
Visualization of state values and learning progress
Comprehensive training process with configurable stopping criteria

Requirements
numpy
matplotlib
seaborn
random
Installation

Ensure you have Python 3.6 or higher installed
Install the required packages:
pip install numpy matplotlib seaborn

Download the script: grid_world_q_learning.py

Usage
Run the script directly to train and visualize the Q-learning agent:
python grid_world_q_learning.py
Code Structure

GridWorldEnv: Defines the grid world environment with its states, actions, and dynamics
QLearningAgent: Implements the Q-learning algorithm with epsilon-greedy exploration
train_agent: Handles the training process for a given agent and environment
visualize_state_values: Creates a heatmap visualization of the learned state values
plot_rewards: Plots the rewards per episode during training
main: Coordinates the overall execution, testing different learning rates

Parameters
The following parameters can be adjusted to experiment with different learning configurations:

learning_rate: Controls how quickly the agent adapts to new information (default: tests [1.0, 0.8, 0.5, 0.2])
discount_factor: Determines the importance of future rewards (default: 0.95)
epsilon: Initial exploration rate (default: 1.0)
epsilon_decay: Rate at which exploration decreases (default: 0.995)
min_epsilon: Minimum exploration rate (default: 0.01)
max_episodes: Maximum number of training episodes (default: 100)
max_steps: Maximum steps per episode (default: 100)

Environment Details

5x5 grid world
Actions: North(0), South(1), East(2), West(3)
Starting position: [2,1] (second row, first column)
Terminal state: [5,5] (bottom-right corner) with +10 reward
Special jump: [2,4] to [4,4] with +5 reward
Obstacles at positions [3,3], [3,4], [4,3]
All other actions result in -1 reward

Example Output
The script produces:

Training progress information for each learning rate
A plot of rewards per episode
A heatmap visualization of the final state values
Comparison of learning performance across different learning rates

Author
Daire Bunting
COM762: Deep Learning and Its Application
University of Ulster
May 2025
