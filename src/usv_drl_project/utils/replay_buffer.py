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
        state_grid = torch.tensor(state['grid_map'], dtype=torch.float32, device=self.device)
        state_vec  = torch.tensor(state['state_vec'], dtype=torch.float32, device=self.device)
        next_grid  = torch.tensor(next_state['grid_map'], dtype=torch.float32, device=self.device)
        next_vec   = torch.tensor(next_state['state_vec'], dtype=torch.float32, device=self.device)
        
        action = torch.tensor(action, dtype=torch.long, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        done   = torch.tensor(done, dtype=torch.float32, device=self.device)

        transition = (state_grid, state_vec, action, reward, next_grid, next_vec, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return tuple(torch.stack(x) for x in zip(*batch))

    def __len__(self):
        return len(self.buffer)
