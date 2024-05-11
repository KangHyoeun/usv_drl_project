#!/usr/bin/env python3
import torch
print(torch.cuda.is_available())
from IPython import display
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


# Parallel environments
vec_env = make_vec_env("CartPole-v1", n_envs=4)
# 환경 및 PPO 모델 설정
model = PPO("MlpPolicy", vec_env, verbose=1)

# 학습 시작 시간 기록
start_time = time.time()

# 모델 학습
model.learn(total_timesteps=25000)

# 학습 종료 시간 기록 및 출력
end_time = time.time()
elapsed_time = end_time - start_time
print(f"학습 완료! 총 걸린 시간: {elapsed_time//60:.0f} 분 {elapsed_time%60:.0f} 초")

# 모델 저장
model.save("ppo_example")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_example")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")