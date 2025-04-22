# src/usv_drl_project/test.py
import torch
import numpy as np
from models.dueling_dqn import DuelingDQN
from envs.usv_collision_env import USVCollisionEnv
from config import CONFIG
import time
from agent import DQNAgent

# 로드된 정책으로 1 에피소드 실행
def run_test_episode(model_path, render=False):
    env = USVCollisionEnv(CONFIG)
    obs, _ = env.reset()

    input_shape = (3, *CONFIG['grid_size'])
    state_vec_dim = 6
    n_actions = 3

    policy_net = DuelingDQN(input_shape, state_vec_dim, n_actions).to(CONFIG['device'])
    target_net = DuelingDQN(input_shape, state_vec_dim, n_actions).to(CONFIG['device'])
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=CONFIG['lr'])

    agent = DQNAgent(policy_net, target_net, optimizer, CONFIG)

    total_reward = 0
    step = 0

    for step in range(200):
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step {step} | Action: {action} | Reward: {reward:.3f} | Done: {terminated or truncated}")
        if terminated or truncated:
            break

    print(f"\nTest episode finished after {step} steps with total reward {total_reward:.2f}")

if __name__ == '__main__':
    run_test_episode(CONFIG['save_path'])