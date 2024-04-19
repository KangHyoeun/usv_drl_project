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

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ShipControlEnv(gym.Env):
    def __init__(self):
        super(ShipControlEnv, self).__init__()
        self.action_space = spaces.Discrete(4)  # 4 방향 제어
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.state = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.goal = np.array([10.0, 10.0])
        self.max_steps = 100
        self.current_step = 0
        self.figure, self.axis = plt.subplots(figsize=(5, 5))
        self.axis.set_xlabel('X Position')
        self.axis.set_ylabel('Y Position')
        self.axis.grid(True)

    def reset(self):
        self.state = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.current_step = 0
        return self.state

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
        x = max(0, min(x, 20))  # x가 0보다 작으면 0, 20보다 크면 20
        y = max(0, min(y, 20))  # y가 0보다 작으면 0, 20보다 크면 20
        
        self.state = np.array([x, y, theta], dtype=np.float32)
        distance_to_goal = np.linalg.norm(self.goal - self.state[:2])
        done = distance_to_goal < 1 or self.current_step >= self.max_steps
        reward = -1 if not done else 1000  # 목표에 도달하면 큰 보상
        reward = -0.1 * distance_to_goal # 목표에 더 가까워질수록 보상 증가
        self.current_step += 1
        return self.state, reward, done, {}

    def render(self, mode='human'):
        if mode == 'human':
            self.axis.clear()
            self.axis.set_xlim(0, 20)
            self.axis.set_ylim(0, 20)
            agent = self.axis.scatter(self.state[0], self.state[1], color='blue', label='Agent', s=100)
            goal = self.axis.scatter(self.goal[0], self.goal[1], color='red', label='Goal', s=100)
            self.axis.set_xlabel('X Position')
            self.axis.set_ylabel('Y Position')
            self.axis.set_title('선박 제어 시뮬레이션')  # 그림 제목 설정
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

def train_dqn(episodes=100):
    start_time = time.time() # 학습 시작 시간

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
                print(f"에피소드: {e+1}, 보상: {total_reward}, 탐색: {agent.epsilon}")
                break
            agent.replay()
            env.render()

    elapsed_time = time.time() - start_time  # 전체 학습 시간 계산
    print(f"학습 완료! 총 걸린 시간: {elapsed_time:.2f} 초")

    # 모델 저장
    torch.save(agent.model.state_dict(), 'trained_model.pth')
    print("모델 저장 완료! 파일 이름: 'trained_model.pth'")

train_dqn(500)
