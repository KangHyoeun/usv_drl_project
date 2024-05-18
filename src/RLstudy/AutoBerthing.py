#!/usr/bin/env python3
import torch
print(torch.cuda.is_available())
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import time
import math
import stable_baselines3
from stable_baselines3 import A2C

class GridWorldEnv(gym.Env):
    def __init__(self):
        super(GridWorldEnv, self).__init__()
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Box(low=0, high=4, shape=(2,), dtype=np.int32)
        self.state = None
        self.goal = (4, 4)

    def reset(self):
        self.state = (0, 0)
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 0:  # Up
            x = max(x - 1, 0)
        elif action == 1:  # Down
            x = min(x + 1, 4)
        elif action == 2:  # Left
            y = max(y - 1, 0)
        elif action == 3:  # Right
            y = min(y + 1, 4)
        
        self.state = (x, y)
        done = self.state == self.goal
        reward = 1 if done else 0
        return self.state, reward, done, {}

    def render(self, mode='human', reward=None):
        grid = np.zeros((5, 5))
        grid[self.state] = 1
        grid[self.goal] = 0.5
        print("Current State Grid: ")
        print(grid)
        if reward is not None:
            print(f"Reward Received: {reward}")

    def close(self):
        pass

env = GridWorldEnv()
obs = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # 무작위 행동 선택
    obs, reward, done, info = env.step(action)
    env.render(reward=reward)
    
env.close()
