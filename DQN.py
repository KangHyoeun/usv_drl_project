import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random

# 환경 생성
env = gym.make('CartPole-v1')

# DQN 모델 정의
class DQN(tf.keras.Model):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(24, activation='relu')
        self.dense2 = tf.keras.layers.Dense(24, activation='relu')
        self.dense3 = tf.keras.layers.Dense(action_size, activation='linear')

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return self.dense3(x)

# 하이퍼파라미터
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
learning_rate = 0.001
epsilon = 1.0  # 탐험률
epsilon_decay = 0.995
epsilon_min = 0.01
batch_size = 64
memory = deque(maxlen=2000)

# 모델 생성
model = DQN(action_size)
optimizer = tf.keras.optimizers.Adam(learning_rate)

# 경험 리플레이
def train_model():
    if len(memory) < batch_size:
        return
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = map(np.asarray, zip(*batch))
    model_params = model.trainable_variables
    with tf.GradientTape() as tape:
        tape.watch(model_params)
        current_q = model(states)
        max_next_q = tf.reduce_max(model(next_states), axis=1)
        target_q = rewards + (1 - dones) * 0.99 * max_next_q
        one_hot_actions = tf.keras.utils.to_categorical(actions, action_size)
        predict_q = tf.reduce_sum(one_hot_actions * current_q, axis=1)
        loss = tf.reduce_mean(tf.square(target_q - predict_q))
    gradients = tape.gradient(loss, model_params)
    optimizer.apply_gradients(zip(gradients, model_params))

# 주 학습 루프
rewards_list = []
for e in range(1, 501):  # 총 에피소드 수
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    total_reward = 0
    while not done:
        if np.random.rand() <= epsilon:
            action = env.action_space.sample()
        else:
            q_values = model(state)
            action = np.argmax(q_values[0])

        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        if done:
            rewards_list.append(total_reward)
            print(f"에피소드: {e}/{500}, 점수: {total_reward}, epsilon: {epsilon:.2}")
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
        
        train_model()

# 학습된 제어 정책 시각화
plt.plot(rewards_list)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()
