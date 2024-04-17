import gym
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# 환경을 생성합니다.
env = gym.make('CartPole-v1')

# 학습된 모델을 불러옵니다.
# 이전에 저장한 모델 가중치를 'cartpole_dqn.h5'로 가정합니다.
model = load_model('cartpole_dqn.h5')

# 제어 성공 여부를 평가하기 위한 시험 에피소드 수
num_episodes = 100
steps_list = []  # 각 에피소드에서의 시간 단계를 저장할 리스트

# 시뮬레이션 루프
for i in range(num_episodes):
    state = env.reset()
    state = np.reshape(state, [1, env.observation_space.shape[0]])
    done = False
    steps = 0
    
    # 에피소드가 끝날 때까지 루프를 돕니다.
    while not done:
        env.render()  # 환경을 시각화합니다 (실제 제어 상황에서는 생략 가능).
        steps += 1
        
        # 학습된 정책을 사용하여 행동을 선택합니다.
        q_values = model.predict(state)
        action = np.argmax(q_values[0])
        
        # 선택된 행동을 적용하고 다음 상태를 관찰합니다.
        next_state, _, done, _ = env.step(action)
        state = np.reshape(next_state, [1, env.observation_space.shape[0]])
        
    steps_list.append(steps)

env.close()  # 환경을 닫습니다.

# 제어 성공 여부 시각화
plt.plot(steps_list)
plt.xlabel('Episode')
plt.ylabel('Time Steps')
plt.title('Performance of DQN-Controlled System')
plt.show()
