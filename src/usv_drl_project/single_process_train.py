# src/usv_drl_project/single_process_train.py
import torch
import torch.nn as nn
import torch.optim as optim
import os
import psutil
import os as osys
import numpy as np
import random
from tqdm import trange
from models.dueling_dqn import DuelingDQN
from utils.replay_buffer import ReplayBuffer
from utils.logger import CSVLogger
from config import CONFIG
from envs.usv_collision_env import USVCollisionEnv
from agent import DQNAgent

def train(seed):

    prev_in_avoidance = False

    # Set seeds globally
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Apply seed to config
    CONFIG['seed'] = seed
    CONFIG['save_path'] = f'./checkpoints/seed_{seed}.pt'

    env = USVCollisionEnv(CONFIG)
    obs, _ = env.reset(seed)

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
        env.set_epsilon(agent.epsilon)
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        done = terminated or truncated

        buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs

        if done:
            obs, _ = env.reset(seed=seed)

        if len(buffer) < CONFIG['start_learning']:
            continue

        if agent.global_step % CONFIG['train_freq'] == 0:
            batch = buffer.sample(CONFIG['batch_size'])
            loss = agent.learn(batch)

            recent_rewards = [b[2].cpu().item() for b in buffer.buffer[-CONFIG['batch_size']:]]
            mean_reward = float(np.mean(recent_rewards))
            logger.log(agent.global_step, mean_reward, loss)

        if agent.global_step % CONFIG['target_update_interval'] == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if agent.global_step % CONFIG['log_interval'] == 0:
            print(f"Step {agent.global_step} | Epsilon: {agent.epsilon:.3f} | ReplayBuffer: {len(buffer)}")
            if not prev_in_avoidance and env.in_avoidance:
                print("→ 회피 시작!")
            prev_in_avoidance = env.in_avoidance
            # mem = psutil.Process(osys.getpid()).memory_info().rss / 1024**2
            # print(f"[Memory] RAM: {mem:.2f} MB")
            # if torch.cuda.is_available():
            #     print(f"[GPU] Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
            
            x, y, psi = env.own_state["x"], env.own_state["y"], env.own_state["psi"]
            has_avoid_target = env.avoid_obs is not None
            tcpa_now = getattr(env, "tcpa_now", None)
            tcpa_str = f"{tcpa_now:.2f}s" if tcpa_now is not None else "N/A"    
            print(f"x={x:.2f}, y={y:.2f}, ψ={np.rad2deg(psi):.1f}°, in_avoidance={env.in_avoidance}, "
                  f"action={action}, avoid_target={has_avoid_target}, TCPA={tcpa_str}")
            # print("▼ 장애물 목록:")
            # for i, obs in enumerate(env.obs_state):
            #     x_o, y_o, psi_o, is_dyn = obs['x'], obs['y'], obs['psi'], obs.get('dynamic', False)
            #     print(f"  - [{i}] x={x_o:.1f}, y={y_o:.1f}, ψ={np.rad2deg(psi_o):.1f}°, dynamic={is_dyn}")


    torch.save(policy_net.state_dict(), CONFIG['save_path'])
    logger.close()

if __name__ == '__main__':
    train()