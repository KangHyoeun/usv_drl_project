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

    def log(self, step, reward, loss):
        self.writer.writerow([step, reward, loss])

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

    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    ax[0].plot(steps, rewards)
    ax[0].set_ylabel('Reward')
    ax[1].plot(steps, losses)
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Step')
    plt.tight_layout()
    plt.show()