#!/usr/bin/env python3
# customEnv.py

import gymnasium as gym 
from gymnasium import spaces

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import multivariate_normal

# 폰트 설정
matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # 폰트 패밀리
matplotlib.rcParams['font.size'] = 12  # 기본 폰트 크기

from mmgdynamics.maneuvers import *
from mmgdynamics.structs import Vessel
from mmgdynamics import step
import math
import os
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VesselEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, vessel: Vessel, initial_state: np.ndarray, dT: float, waypoints: list, render_mode='human', max_steps=500, slow_down_distance=10.0):
        super(VesselEnv, self).__init__()
        self.vessel = vessel
        self.render_mode = render_mode
        self.initial_state = initial_state
        self.state = np.copy(initial_state)
        self.dT = dT
        self.waypoints = waypoints
        self.current_waypoint_index = 0
        self.target_position, self.target_psi = self.waypoints[self.current_waypoint_index]

        self.psi = 0.0  # Assume initial heading is zero
        self.position = np.array([0.0, 0.0])  # Assume initial position is at the origin

        self.fl_psi = np.deg2rad(5)
        self.fl_vel = 0.1
        self.w_vel = 0.1
        self.beta_w = 0.1

        self.max_steps = max_steps
        self.current_step = 0
        self.slow_down_distance = slow_down_distance
        self.history = {'x': [], 'y': [], 'rewards': [], 'heading': []}
        logging.info("Environment initialized.")

        # Action space: [propeller revs per second(nps), rudder angle(delta)] # Normalize action space to [-1, 1]
        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float64)

        # Actual action bounds
        self.nps_bounds = (0.0, 13.4)
        self.delta_bounds = (np.deg2rad(-35), np.deg2rad(35))

        # Observation space: [x, y, u, v, r, distance to target, heading to target, heading error]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64)

    def step(self, action):
        logging.info(f"Step action received: {action}")
        self.current_step += 1

        # Rescale action from [-1, 1] to actual action bounds
        nps = (action[0] * (self.nps_bounds[1] - self.nps_bounds[0]) / 2.0 + (self.nps_bounds[1] + self.nps_bounds[0]) / 2.0)
        delta = (action[1] * (self.delta_bounds[1] - self.delta_bounds[0]) / 2.0 + (self.delta_bounds[1] + self.delta_bounds[0]) / 2.0)

        logging.info(f"nps: {nps}, delta: {delta}")

        next_state = step(
            X=self.state,
            vessel=self.vessel,
            dT=self.dT,
            nps=nps,
            delta=delta,
            psi=self.psi,
            fl_psi=self.fl_psi,
            fl_vel=self.fl_vel,
            w_vel=self.w_vel,
            beta_w=self.beta_w
        )
        self.state = np.squeeze(next_state)

        # Update heading
        self.psi += self.state[2] * self.dT

        # Update position
        self.position[0] += self.state[0] * np.cos(self.psi) * self.dT - self.state[1] * np.sin(self.psi) * self.dT
        self.position[1] += self.state[0] * np.sin(self.psi) * self.dT + self.state[1] * np.cos(self.psi) * self.dT

        # Calculate distance and heading to target
        distance_to_target = np.linalg.norm(self.target_position - self.position)
        heading_error = self.target_psi - self.psi

        # Slow down near the target
        if distance_to_target < self.slow_down_distance:
            nps *= 0.9

        # Observation
        observation = np.array([self.position[0], self.position[1], self.state[0], self.state[1], self.state[2], distance_to_target, heading_error])

        # Reward
        reward = 0

        # 1. 목표 위치에 도달하는 것에 대한 보상 증가
        if distance_to_target < 1.0:
            reward += 1000.0

        # 2. 방위각 유지 보상
        if abs(heading_error) < np.deg2rad(5):
            reward += 500.0

        # 3. 목표 위치와의 거리에 따른 보상
        reward += -distance_to_target

        # 4. 방위각 오차에 따른 보상
        reward += -abs(heading_error)

        # 5. 안정적인 경로 유지 보상 (조작량 줄이기)
        reward += -abs(delta) * 10.0
        
        # 다변량 가우시안 보상 함수
        # reward = self.multivariate_gaussian_reward(distance_to_target, heading_error)
        
        # Done condition
        terminated = bool(distance_to_target < 1.0 and abs(heading_error) < np.deg2rad(5))  # Consider terminated if within 1 meter and 5 degrees of target
        truncated = bool(self.current_step >= self.max_steps)

        done = terminated or truncated  # Combine conditions for done

        if terminated and self.current_waypoint_index < len(self.waypoints) - 1:
            self.current_waypoint_index += 1
            self.target_position, self.target_psi = self.waypoints[self.current_waypoint_index]
            done = False

        # Record history for rendering
        self.history['x'].append(self.position[0])
        self.history['y'].append(self.position[1])
        self.history['rewards'].append(reward)
        self.history['heading'].append(self.psi)

        logging.info(f"Step completed. Position: {self.position}, Reward: {reward}, Done: {done}")

        return observation, reward, done, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.copy(self.initial_state)
        self.position = np.array([0.0, 0.0])
        self.psi = 0.0
        self.history = {'x': [], 'y': [], 'rewards': [], 'heading': []}
        self.current_waypoint_index = 0
        self.target_position, self.target_psi = self.waypoints[self.current_waypoint_index]

        self.current_step = 0
        distance_to_target = np.linalg.norm(self.target_position - self.position)
        heading_error = self.target_psi - self.psi
        logging.info("Environment reset.")

        return np.array([self.position[0], self.position[1], self.state[0], self.state[1], self.state[2], distance_to_target, heading_error]), {}


    def render(self, render_mode='human', save_path='./plots', env_index=None):
        if render_mode == 'human':
            logging.info("Rendering the environment.")
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            plt.figure(figsize=(15, 10))

            # Plot trajectory and goal points
            plt.subplot(3, 1, 1)
            plt.plot(self.history['x'], self.history['y'], label='Trajectory')
            plt.scatter(self.target_position[0], [self.target_position[1]], color='red')
            plt.xlim(0, 60)
            plt.ylim(-5, 5)
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.title(f'Ship Trajectory (Env {env_index})')
            plt.legend()

            # Plot rewards over time
            plt.subplot(3, 1, 2)
            plt.plot(self.history['rewards'], label='Rewards')
            plt.xlabel('Step')
            plt.ylabel('Reward')
            plt.title('Rewards Over Time')
            plt.legend()

            # Plot heading angle over time
            plt.subplot(3, 1, 3)
            plt.plot(self.history['heading'], label='Heading Angle')
            plt.xlabel('Step')
            plt.ylabel('Heading Angle (rad)')
            plt.title('Heading Angle Over Time')
            plt.legend()

            plt.tight_layout()
            plt.savefig(f"{save_path}/trajectory_and_rewards_env{env_index}.png")

            logging.info(f"Render completed and saved for environment {env_index}.")
            
    def multivariate_gaussian_reward(self, distance, heading_error):
        mean = np.array([0, 0])
        cov = np.array([[1, 0], [0, 1]])
        pos = np.array([distance, heading_error])
        rv = multivariate_normal(mean, cov)
        reward = rv.pdf(pos)
        return reward * 1000  # 스케일링을 통해 보상 값 조정
    
    def close(self):
        pass
