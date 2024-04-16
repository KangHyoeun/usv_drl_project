import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)  # Ensure consistent input shape
        act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            target = reward + (1 - done) * self.gamma * torch.max(self.model(next_state)).item()
            prediction = self.model(state)
            target_f = prediction.clone()
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(prediction, target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

env = gym.make('CartPole-v1')
agent = Agent(state_size=4, action_size=2)
episodes = 1000
scores = []

for e in range(episodes):
    state = env.reset()
    state = np.array(state[0])
    total_reward = 0
    while True:
        action = agent.act(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = np.array(next_state[0])
        done = terminated or truncated
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            scores.append(total_reward)
            print(f'Episode {e+1}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}')
            break
        agent.replay(32)

env.close()

plt.figure(figsize=(10, 5))
plt.plot(scores)
plt.title('DQN Agent Performance on CartPole-v1')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.show()
