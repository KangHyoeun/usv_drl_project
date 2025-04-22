# src/usv_drl_project/config.py
import torch
import numpy as np

CONFIG = {
    'grid_size': (84, 84),
    'n_envs': 4,  # 병렬 환경 개수
    'n_actions': 3,
    'buffer_size': 65000,
    'batch_size': 128,
    'gamma': 0.99,
    'lr': 1e-4,
    'target_update_interval': 200,
    'train_freq': 4,
    'start_learning': 1000,
    'max_episode_steps': 1000,
    'total_timesteps': 300000,
    'epsilon_start': 1.0,
    'epsilon_final': 0.05,
    'epsilon_decay': 20000,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'log_interval': 1000,
    'save_path': './checkpoints/seed_{seed}.pt',
    'tcpa_exploration_factor': 0.3, # ε 역할
    'vfg_chi_inf_deg': 45.0,   # chi_inf in degrees
    'vfg_k': -0.5,              # tanh 경사도
    'vfg_chi_path_deg': 0.0,   # 경로 진행 방향 (0도면 y축)
    'dcpa_thresh': 30.0,  # DCPA 임계값 (단위: 미터)
    'tcpa_thresh': 60.0,  # TCPA 임계값 (단위: 초)
    'max_speed': 3.0,
    "max_yaw_rate": np.deg2rad(10.0),
    "max_rpm": 993,
    "scenario": 'HO',
}
