# src/usv_drl_project/utils/replay_buffer.py
import random
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)

        grid_map = torch.tensor(np.array([s['grid_map'] for s in state]), dtype=torch.float32).to(self.device)
        state_vec = torch.tensor(np.array([s['state_vec'] for s in state]), dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        done = torch.tensor(done, dtype=torch.float32).to(self.device)

        next_grid_map = torch.tensor(np.array([s['grid_map'] for s in next_state]), dtype=torch.float32).to(self.device)
        next_state_vec = torch.tensor(np.array([s['state_vec'] for s in next_state]), dtype=torch.float32).to(self.device)

        return (grid_map, state_vec, action, reward, next_grid_map, next_state_vec, done)

    def __len__(self):
        return len(self.buffer)
