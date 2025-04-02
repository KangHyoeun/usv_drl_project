# src/usv_drl_project/test.py
import torch
import numpy as np
from models.dueling_dqn import DuelingDQN
from envs.usv_collision_env import USVCollisionEnv
from config import CONFIG
import time

# 로드된 정책으로 1 에피소드 실행
def run_test_episode(model_path, render=False):
    env = USVCollisionEnv(CONFIG)
    obs, _ = env.reset()

    input_shape = (3, *CONFIG['grid_size'])
    state_vec_dim = 6
    n_actions = 3

    model = DuelingDQN(input_shape, state_vec_dim, n_actions).to(CONFIG['device'])
    model.load_state_dict(torch.load(model_path, map_location=CONFIG['device']))
    model.eval()

    total_reward = 0
    step = 0
    done = False

    while not done:
        grid = torch.tensor(obs['grid_map'], dtype=torch.float32).unsqueeze(0).to(CONFIG['device'])
        vec = torch.tensor(obs['state_vec'], dtype=torch.float32).unsqueeze(0).to(CONFIG['device'])

        with torch.no_grad():
            q_values = model(grid, vec)
            action = torch.argmax(q_values, dim=1).item()

        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        step += 1

        print(f"Step {step:3} | Action: {action} | Reward: {reward:.3f}")
        if render:
            time.sleep(0.1)

    print(f"\nTest episode finished after {step} steps with total reward {total_reward:.2f}")

if __name__ == '__main__':
    run_test_episode(CONFIG['save_path'])