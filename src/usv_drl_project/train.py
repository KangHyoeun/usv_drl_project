# src/usv_drl_project/train.py
import torch
import torch.nn as nn
import torch.optim as optim
import os
import psutil
import os as osys
import random
import numpy as np
import copy
from tqdm import trange
from models.dueling_dqn import DuelingDQN
from utils.replay_buffer import ReplayBuffer
from utils.logger import CSVLogger
from config import CONFIG
from envs.usv_collision_env import USVCollisionEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from agent import DQNAgent

def make_env(rank, config, seed):
    def _init():
        config_copy = copy.deepcopy(config)  # ← 얕은 복사 대신 깊은 복사
        config_copy['env_seed'] = seed + rank
        env = USVCollisionEnv(config_copy)
        return env
    return _init

def train_with_seed(seed):
    print(f"\n=== Starting training with SEED: {seed} ===")

    # Set seeds globally
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Apply seed to config
    CONFIG['seed'] = seed
    CONFIG['save_path'] = f'./checkpoints/seed_{seed}.pt'

    envs = SubprocVecEnv([make_env(i, CONFIG, seed) for i in range(CONFIG['n_envs'])])
    obs = envs.reset()

    input_shape = (3, *CONFIG['grid_size'])
    state_vec_dim = 6
    n_actions = 3

    policy_net = DuelingDQN(input_shape, state_vec_dim, n_actions).to(CONFIG['device'])
    print("PolicyNet device:", next(policy_net.parameters()).device)
    target_net = DuelingDQN(input_shape, state_vec_dim, n_actions).to(CONFIG['device'])
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=CONFIG['lr'])
    buffer = ReplayBuffer(CONFIG['buffer_size'], CONFIG['device'])
    agent = DQNAgent(policy_net, target_net, optimizer, CONFIG)

    os.makedirs(os.path.dirname(CONFIG['save_path']), exist_ok=True)
    logger = CSVLogger(f'./logs/train_seed_{seed}.csv')

    for _ in trange(CONFIG['total_timesteps'], desc=f"Seed {seed}"):
        actions = agent.select_action_batch(obs)
        envs.env_method('set_epsilon', agent.epsilon)
        next_obs, rewards, dones, infos = envs.step(actions)

        for i in range(CONFIG['n_envs']):
            curr_obs = {
                'grid_map': obs['grid_map'][i],
                'state_vec': obs['state_vec'][i],
            }
            next_o = {
                'grid_map': next_obs['grid_map'][i],
                'state_vec': next_obs['state_vec'][i],
            }
            done = dones[i]
            buffer.push(curr_obs, actions[i], rewards[i], next_o, done=done)

        obs = next_obs

        if len(buffer) < CONFIG['start_learning']:
            continue

        if agent.global_step % CONFIG['train_freq'] == 0:
            batch = buffer.sample(CONFIG['batch_size'])
            loss = agent.learn(batch)

            # reward curve logging
            recent_rewards = [b[2].cpu().item() for b in buffer.buffer[-CONFIG['batch_size']:]]
            mean_reward = float(np.mean(recent_rewards))
            logger.log(agent.global_step, mean_reward, loss)

        if agent.global_step % CONFIG['target_update_interval'] == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if agent.global_step % CONFIG['log_interval'] == 0:
            print(f"Step {agent.global_step} | Epsilon: {agent.epsilon:.3f} | ReplayBuffer: {len(buffer)}")
            # mem = psutil.Process(osys.getpid()).memory_info().rss / 1024**2
            # print(f"[Memory] RAM: {mem:.2f} MB")
            # if torch.cuda.is_available():
            #     print(f"[GPU] Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
            # ✅ 자선 위치 및 heading 확인
            own_states = envs.get_attr("own_state")  # 각 env의 own_state 딕셔너리 리스트
            for i, s in enumerate(own_states):
                x, y, psi = s["x"], s["y"], s["psi"]
                print(f"[Env {i}] x={x:.2f}, y={y:.2f}, ψ={np.rad2deg(psi):.1f}°")


    torch.save(policy_net.state_dict(), CONFIG['save_path'])
    logger.close()
    envs.close()

def sweep_seeds(seed_list):
    for seed in seed_list:
        train_with_seed(seed)

if __name__ == '__main__':
    SEED_LIST = [0, 10, 20, 30, 40]  # 원하는 seed 목록
    sweep_seeds(SEED_LIST)