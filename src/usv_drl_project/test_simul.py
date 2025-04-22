# src/usv_drl_project/test.py
import torch
import matplotlib.pyplot as plt
import numpy as np
from models.dueling_dqn import DuelingDQN
from envs.usv_collision_env import USVCollisionEnv
from config import CONFIG
import time
from agent import DQNAgent

env = USVCollisionEnv(CONFIG)
obs, _ = env.reset(seed=0)

positions = []

# 초기 상태 지정
env.own_state["psi"] = np.deg2rad(90)  # +90도 = 동쪽
env.own_state["x"] = 0.0
env.own_state["y"] = 0.0

for step in range(1000):
    env.set_epsilon(0.0)  # 탐험 방지
    action = 0  # 경로추종만 확인
    obs, reward, terminated, truncated, info = env.step(action)

    x, y, psi = env.own_state["x"], env.own_state["y"], env.own_state["psi"]
    positions.append((x, y))
    print(f"[{step:03}] x={x:.2f}, y={y:.2f}, ψ={np.rad2deg(psi):.1f}°")

# 궤적 시각화
positions = np.array(positions)
plt.figure(figsize=(10, 10))
plt.plot(positions[:,1], positions[:,0], 'b-')  # Y=East, X=North
plt.xlabel('East (m)')
plt.ylabel('North (m)')
plt.title('USV Trajectory (ψ=90° → East)')
plt.grid(True)
plt.axis('equal')
plt.show()