# src/usv_drl_project/utils/logger.py
import csv
import os

class CSVLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.file = open(filepath, mode='w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['step', 'reward', 'loss'])
        print(f"[Logger] Logging to {filepath}")

    def log(self, step, reward, loss):
        self.writer.writerow([step, reward, loss])
        self.file.flush()  # 실시간 디스크 기록

    def close(self):
        self.file.close()

# 시각화용
import matplotlib.pyplot as plt

def plot_csv_log(filepath):
    steps, rewards, losses = [], [], []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row['step']))
            rewards.append(float(row['reward']))
            losses.append(float(row['loss']))

    seed_str = os.path.splitext(os.path.basename(filepath))[0].split('_')[-1]

    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    ax[0].plot(steps, rewards)
    ax[0].set_ylabel(f'Reward (Seed {seed_str})')
    ax[1].plot(steps, losses)
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Step')
    plt.tight_layout()
    plt.savefig(f'./logs/plot_seed_{seed_str}.png')
    plt.close()
