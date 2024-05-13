#!/usr/bin/env python3

# customEnv.py

import gymnasium as gym 
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

import mmgdynamics.calibrated_vessels as cvs
from mmgdynamics.maneuvers import *
from mmgdynamics.structs import Vessel

import os

class VesselEnv(gym.Env):
    def __init__(self, vessel: Vessel, initial_state: np.ndarray, dT: float, target_position: np.ndarray):
        super(VesselEnv, self).__init__()
        self.vessel = vessel
        self.initial_state = initial_state
        self.state = np.copy(initial_state)
        self.dT = dT
        self.target_position = target_position
        self.psi = 0.0  # Assume initial heading is zero
        self.position = np.array([0.0, 0.0])  # Assume initial position is at the origin
        self.history = {'x': [], 'y': [], 'rewards': []}

        # Action space: [propeller revs per second(nps), rudder angle(delta)]
        self.action_space = spaces.Box(low=np.array([0.0, np.deg2rad(-30)]),
                                       high=np.array([13.4, np.deg2rad(30)]),
                                       dtype=np.float32)

        # Observation space: [u, v, r, distance to target, heading to target, x, y]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

    def step(self, action):
        nps, delta = action

        next_state = step(
            X=self.state,
            vessel=self.vessel,
            dT=self.dT,
            nps=nps,
            delta=delta,
            psi=self.psi
        )
        self.state = np.squeeze(next_state)

        # Update heading
        self.psi += self.state[2] * self.dT

        # Update position
        self.position[0] += self.state[0] * np.cos(self.psi) * self.dT - self.state[1] * np.sin(self.psi) * self.dT
        self.position[1] += self.state[0] * np.sin(self.psi) * self.dT + self.state[1] * np.cos(self.psi) * self.dT

        # Calculate distance and heading to target
        distance_to_target = np.linalg.norm(self.target_position - self.position)
        heading_to_target = np.arctan2(self.target_position[1] - self.position[1],
                                       self.target_position[0] - self.position[0]) - self.psi

        # Observation
        observation = np.array([self.state[0], self.state[1], self.state[2], distance_to_target, heading_to_target, self.position[0], self.position[1]])

        # Reward
        reward = -distance_to_target

        # Done condition
        terminated = distance_to_target < 1.0  # Consider terminated if within 1 meter of target
        truncated = False  # You can define your own condition or logic here

        done = terminated or truncated  # Combine conditions for done

        # Record history for rendering
        self.history['x'].append(self.position[0])
        self.history['y'].append(self.position[1])
        self.history['rewards'].append(reward)

        return observation, reward, done, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.copy(self.initial_state)
        self.position = np.array([0.0, 0.0])
        self.psi = 0.0
        self.history = {'x': [], 'y': [], 'rewards': []}
        distance_to_target = np.linalg.norm(self.target_position - self.position)
        heading_to_target = np.arctan2(self.target_position[1] - self.position[1], self.target_position[0] - self.position[0])
        return np.array([self.state[0], self.state[1], self.state[2], distance_to_target, heading_to_target, self.position[0], self.position[1]]), {}


    def render(self, mode='human', save_path='./plots'):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.history['x'], self.history['y'], label='Trajectory')
        plt.scatter([self.target_position[0]], [self.target_position[1]], color='red', label='Goal')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Ship Trajectory')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.history['rewards'], label='Rewards')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.title('Rewards Over Time')
        plt.legend()

        plt.savefig(f"{save_path}/trajectory_and_rewards.png")
        plt.close()