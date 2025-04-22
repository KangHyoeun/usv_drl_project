# src/usv_drl_project/agent.py
import torch
import numpy as np

class DQNAgent:
    def __init__(self, policy_net, target_net, optimizer, config):
        self.policy_net = policy_net
        self.target_net = target_net
        self.optimizer = optimizer
        self.config = config
        self.device = config['device']

        self.epsilon = config['epsilon_start']
        self.epsilon_final = config['epsilon_final']
        self.epsilon_decay = config['epsilon_decay']
        self.n_actions = config.get('n_actions', 3)
        self.global_step = 0

    def update_epsilon(self):
        decay = self.global_step / self.epsilon_decay
        self.epsilon = max(self.epsilon_final, self.config['epsilon_start'] - decay)

    def select_action(self, obs):
        self.update_epsilon()
        self.global_step += 1

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)

        grid = torch.tensor(obs['grid_map'], dtype=torch.float32).unsqueeze(0).to(self.device)
        vec = torch.tensor(obs['state_vec'], dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.policy_net(grid, vec)
            return torch.argmax(q_values, dim=1).item()
        
    def select_action_batch(self, batch_obs):
        self.update_epsilon()
        self.global_step += len(batch_obs['grid_map'])  # 환경 수만큼 증가

        # Exploration (벡터화)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions, size=len(batch_obs['grid_map']))

        # 벡터화된 tensor 입력
        grid = torch.tensor(batch_obs['grid_map'], dtype=torch.float32).to(self.device)
        vec = torch.tensor(batch_obs['state_vec'], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            q_values = self.policy_net(grid, vec)
            return torch.argmax(q_values, dim=1).cpu().numpy()


    def learn(self, batch):
        grid_map, state_vec, action_batch, reward_batch, next_grid_map, next_state_vec, done_batch = batch

        with torch.no_grad():
            next_q = self.target_net(next_grid_map, next_state_vec).max(1)[0]
            target_q = reward_batch + (1 - done_batch) * self.config['gamma'] * next_q

        current_q = self.policy_net(grid_map, state_vec).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        loss = torch.nn.functional.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
