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
    envs = SubprocVecEnv([make_env(CONFIG) for _ in range(CONFIG['n_envs'])])
    obs = envs.reset()

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
        actions = []
        for i in range(CONFIG['n_envs']):
            # ❌ TCPA 판단 제거
            if np.random.rand() < epsilon:
                actions.append(np.random.choice([0, 1, 2]))  # 💡 무작위 행동 선택
                # actions.append(0)  # always 경로추종 명령
            else:
                encounter_type = obs['encounter_type'][i]
                with torch.no_grad():
                    grid = torch.tensor(obs['grid_map'][i], dtype=torch.float32).unsqueeze(0).to(CONFIG['device'])
                    vec = torch.tensor(obs['state_vec'][i], dtype=torch.float32).unsqueeze(0).to(CONFIG['device'])
                    q_values = policy_net(grid, vec)
                    if encounter_type == 'Static':
                        avoid_action = torch.argmax(q_values[0, 1:]).item() + 1
                    else:
                        avoid_action = torch.argmax(q_values, dim=1).item()
                    actions.append(avoid_action)

        # 먼저 epsilon 업데이트
        epsilon = max(CONFIG['epsilon_final'], CONFIG['epsilon_start'] - global_step / CONFIG['epsilon_decay'])
        epsilons = [epsilon] * CONFIG['n_envs']
        next_obs, rewards, terminateds, truncateds, infos = envs.step(actions, epsilons)

        for i in range(CONFIG['n_envs']):
            curr_obs = {
                'grid_map': obs['grid_map'][i],
                'state_vec': obs['state_vec'][i]
            }
            next_o = {
                'grid_map': next_obs['grid_map'][i],
                'state_vec': next_obs['state_vec'][i]
            }
            done = np.logical_or(terminateds[i], truncateds[i])
            buffer.push(curr_obs, actions[i], rewards[i], next_o, done=done)

        obs = next_obs
        global_step += 1

        if len(buffer) < CONFIG['start_learning']:
            continue

        if global_step % CONFIG['train_freq'] == 0:
            grid_map, state_vec, action, reward, next_grid_map, next_state_vec, done = buffer.sample(CONFIG['batch_size'])

            with torch.no_grad():
                next_q = target_net(next_grid_map, next_state_vec).max(1)[0]
                target_q = reward + (1 - done) * CONFIG['gamma'] * next_q

            current_q = policy_net(grid_map, state_vec).gather(1, action.unsqueeze(1)).squeeze(1)
            loss = nn.MSELoss()(current_q, target_q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger.log(global_step, reward.mean().item(), loss.item())

        if global_step % CONFIG['target_update_interval'] == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if global_step % CONFIG['log_interval'] == 0:
            print(f"Step {global_step} | Epsilon: {epsilon:.3f} | ReplayBuffer: {len(buffer)}")

    torch.save(policy_net.state_dict(), CONFIG['save_path'])
    logger.close()


