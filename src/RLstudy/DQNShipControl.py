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

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ShipControlEnv(gym.Env):
    MAX_STEPS = 100
    NUM_ACTIONS = 4
    GOAL_POSITION = np.array([10.0, 10.0])
    GRID_LIMITS = (0, 20)  # X and Y limits for the position grid
    GRID_SIZE = 20

    def __init__(self):
        super(ShipControlEnv, self).__init__()
        self.action_space = spaces.Discrete(self.NUM_ACTIONS)  # 4 방향 제어
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.state = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # 초기 상태
        self.goal = self.GOAL_POSITION  # 목표 위치 초기화
        self.current_step = 0
        self.figure, self.axis = plt.subplots(figsize=(5, 5))
        self.axis.set_xlabel('X Position')
        self.axis.set_ylabel('Y Position')
        self.axis.grid(True)
        self.reset()

    def reset(self):
        self.state = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.current_step = 0
        return self.normalize_state(self.state)

    def step(self, action):
        x, y, theta = self.state
        if action == 1:  # 전진
            x += np.cos(np.radians(theta))
            y += np.sin(np.radians(theta))
        elif action == 2:  # 좌회전
            theta += 10.0
        elif action == 3:  # 우회전
            theta -= 10.0
        elif action == 4:  # 후진
            x -= np.cos(np.radians(theta))
            y -= np.sin(np.radians(theta))

        # 경계 조건 확인 및 조정
        x = max(self.GRID_LIMITS[0], min(x, self.GRID_LIMITS[1]))
        y = max(self.GRID_LIMITS[0], min(y, self.GRID_LIMITS[1]))
        theta = theta % 360  # Ensure theta is within 0-360 degrees
        
        self.state = np.array([x, y, theta], dtype=np.float32)
        normalized_state = self.normalize_state(self.state)
        distance_to_goal = np.linalg.norm(self.goal - self.state[:2])
        done = distance_to_goal < 1 or self.current_step >= self.max_steps
        reward = -1 if not done else 1000
        reward -= 0.1 * distance_to_goal
        self.current_step += 1
        return normalized_state, reward, done, {}
    
    def normalize_state(self, state):
        x, y, theta = state
        x_norm = (x - self.GRID_LIMITS[0]) / self.GRID_SIZE
        y_norm = (y - self.GRID_LIMITS[0]) / self.GRID_SIZE
        theta_norm = (theta / 360.0) * 2 * math.pi  # Normalize and convert to radians
        return np.array([x_norm, y_norm, np.sin(theta_norm), np.cos(theta_norm)], dtype=np.float32)

    def render(self, mode='human'):
        if mode == 'human':
            self.axis.clear()
            self.axis.set_xlim(self.GRID_LIMITS[0], self.GRID_LIMITS[1])
            self.axis.set_ylim(self.GRID_LIMITS[0], self.GRID_LIMITS[1])
            agent = self.axis.scatter(self.state[0], self.state[1], color='blue', label='Agent', s=100)
            goal = self.axis.scatter(self.goal[0], self.goal[1], color='red', label='Goal', s=100)
            self.axis.set_xlabel('X Position')
            self.axis.set_ylabel('Y Position')
            self.axis.set_title('Ship Control Simulation')  # 그림 제목 설정
            self.axis.legend(handles=[agent, goal], loc='upper right')
            plt.pause(0.01)

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        ).to(device)

    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=10000)
        self.batch_size = 128
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0002)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_dim)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
            target = reward + (self.gamma * torch.max(self.model(next_state)).item() * (not done))
            current_q = self.model(state)[0][action]
            loss = nn.functional.mse_loss(current_q, torch.tensor(target, dtype=torch.float32, device=device))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

env = ShipControlEnv()
agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)

import matplotlib.pyplot as plt
import time

def train_dqn(episodes=100):
    rewards = []  # 에피소드별 총 보상을 기록
    epsilons = []  # 에피소드별 탐색 비율을 기록
    start_time = time.time()

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                rewards.append(total_reward)
                epsilons.append(agent.epsilon)
                print(f"에피소드: {e+1}, 보상: {total_reward}, 탐색: {agent.epsilon}")
                break
            agent.replay()
            env.render()

    elapsed_time = time.time() - start_time
    print(f"학습 완료! 총 걸린 시간: {elapsed_time:.2f} 초")

    # 그래프 그리기
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Episode vs Total Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epsilons)
    plt.title('Episode vs Epsilon')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # 모델 저장
    torch.save(agent.model.state_dict(), 'trained_model.pth')
    print("모델 저장 완료! 파일 이름: 'trained_model.pth'")

# 환경 및 에이전트 초기화
env = ShipControlEnv()
agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
train_dqn(500)
