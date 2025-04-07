# src/usv_drl_project/train.py
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from tqdm import trange
from models.dueling_dqn import DuelingDQN
from utils.replay_buffer import ReplayBuffer
from utils.logger import CSVLogger
from config import CONFIG
from envs.usv_collision_env import USVCollisionEnv
from stable_baselines3.common.vec_env import SubprocVecEnv

def make_env(config):
    return lambda: USVCollisionEnv(config)

def train():
    env = USVCollisionEnv(CONFIG)
    obs, _ = env.reset()

    input_shape = (3, *CONFIG['grid_size'])
    state_vec_dim = 6
    n_actions = 3

    policy_net = DuelingDQN(input_shape, state_vec_dim, n_actions).to(CONFIG['device'])
    target_net = DuelingDQN(input_shape, state_vec_dim, n_actions).to(CONFIG['device'])
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=CONFIG['lr'])
    buffer = ReplayBuffer(CONFIG['buffer_size'], CONFIG['device'])

    epsilon = CONFIG['epsilon_start']
    global_step = 0

    os.makedirs(os.path.dirname(CONFIG['save_path']), exist_ok=True)
    logger = CSVLogger('./logs/train_log.csv')

    for _ in trange(CONFIG['total_timesteps']):
        # Epsilon 업데이트
        epsilon = max(CONFIG['epsilon_final'], CONFIG['epsilon_start'] - global_step / CONFIG['epsilon_decay'])

        # 행동 선택
        if np.random.rand() < epsilon:
            action = np.random.choice([0, 1, 2])
        else:
            with torch.no_grad():
                grid = torch.tensor(obs['grid_map'], dtype=torch.float32).unsqueeze(0).to(CONFIG['device'])
                vec = torch.tensor(obs['state_vec'], dtype=torch.float32).unsqueeze(0).to(CONFIG['device'])
                q_values = policy_net(grid, vec)
                if obs['encounter_type'] == 'Static':
                    action = torch.argmax(q_values[0, 1:]).item() + 1
                else:
                    action = torch.argmax(q_values, dim=1).item()

        env.set_epsilon(epsilon)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        curr_obs = {
            'grid_map': obs['grid_map'],
            'state_vec': obs['state_vec'],
            'encounter_type': obs['encounter_type']
        }
        next_o = {
            'grid_map': next_obs['grid_map'],
            'state_vec': next_obs['state_vec'],
            'encounter_type': next_obs['encounter_type']
        }

        buffer.push(curr_obs, action, reward, next_o, done)
        obs = next_obs
        global_step += 1

        if len(buffer) < CONFIG['start_learning']:
            continue

        if global_step % CONFIG['train_freq'] == 0:
            grid_map, state_vec, action_batch, reward_batch, next_grid_map, next_state_vec, done_batch = buffer.sample(CONFIG['batch_size'])

            with torch.no_grad():
                next_q = target_net(next_grid_map, next_state_vec).max(1)[0]
                target_q = reward_batch + (1 - done_batch) * CONFIG['gamma'] * next_q

            current_q = policy_net(grid_map, state_vec).gather(1, action_batch.unsqueeze(1)).squeeze(1)
            loss = nn.MSELoss()(current_q, target_q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger.log(global_step, reward_batch.mean().item(), loss.item())

        if global_step % CONFIG['target_update_interval'] == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if global_step % CONFIG['log_interval'] == 0:
            print(f"Step {global_step} | Epsilon: {epsilon:.3f} | ReplayBuffer: {len(buffer)}")

    torch.save(policy_net.state_dict(), CONFIG['save_path'])
    logger.close()


