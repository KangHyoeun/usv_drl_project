#!/usr/bin/env python3

# customEnv.py

import gymnasium as gym 
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 폰트 설정
matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # 폰트 패밀리
matplotlib.rcParams['font.size'] = 12  # 기본 폰트 크기

import mmgdynamics.calibrated_vessels as cvs
from mmgdynamics.maneuvers import *
from mmgdynamics.structs import Vessel
from mmgdynamics import step

import os
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class VesselEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, vessel: Vessel, initial_state: np.ndarray, dT: float, target_position: np.ndarray, render_mode='human', max_steps=500):
        super(VesselEnv, self).__init__()
        self.vessel = vessel
        self.render_mode = render_mode
        self.initial_state = initial_state
        self.state = np.copy(initial_state)
        self.dT = dT
        self.target_position = target_position
        self.psi = 0.0  # Assume initial heading is zero
        self.position = np.array([0.0, 0.0])  # Assume initial position is at the origin

        self.max_steps = max_steps
        self.current_step = 0
        self.history = {'x': [], 'y': [], 'rewards': []}
        logging.info("Environment initialized.")

        # Calculate the initial distance to the target position to initialize last_distance
        self.last_distance = np.linalg.norm(self.target_position - self.position)

        # Action space: [propeller revs per second(nps), rudder angle(delta)]
        # Normalize action space to [-1, 1]
        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float64)

        # Actual action bounds
        self.nps_bounds = (0.0, 13.4)
        self.delta_bounds = (np.deg2rad(-35), np.deg2rad(35))

        # Observation space: [u, v, r, distance to target, heading to target, x, y]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64)

    def step(self, action):
        logging.info(f"Step action received: {action}")
        self.current_step += 1

        # Rescale action from [-1, 1] to actual action bounds
        nps = (action[0] * (self.nps_bounds[1] - self.nps_bounds[0]) / 2.0 + (self.nps_bounds[1] + self.nps_bounds[0]) / 2.0)
        delta = (action[1] * (self.delta_bounds[1] - self.delta_bounds[0]) / 2.0 + (self.delta_bounds[1] + self.delta_bounds[0]) / 2.0)

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
        truncated = self.current_step >= self.max_steps  # You can define your own condition or logic here

        done = terminated or truncated  # Combine conditions for done

        # Record history for rendering
        self.history['x'].append(self.position[0])
        self.history['y'].append(self.position[1])
        self.history['rewards'].append(reward)

        logging.info(f"Step completed. Position: {self.position}, Reward: {reward}, Done: {done}")


        return observation, reward, done, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.copy(self.initial_state)
        self.position = np.array([0.0, 0.0])
        self.psi = 0.0
        self.history = {'x': [], 'y': [], 'rewards': []}

        self.current_step = 0
        distance_to_target = np.linalg.norm(self.target_position - self.position)
        heading_to_target = np.arctan2(self.target_position[1] - self.position[1], self.target_position[0] - self.position[0])
        logging.info("Environment reset.")

        return np.array([self.state[0], self.state[1], self.state[2], distance_to_target, heading_to_target, self.position[0], self.position[1]]), {}


    def render(self, render_mode='human', save_path='./plots'):
        if render_mode == 'human':
            logging.info("Rendering the environment.")

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

            logging.info("Render completed and saved.")

    def close(self):
        pass
