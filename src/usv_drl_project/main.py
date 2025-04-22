# src/usv_drl_project/main.py
# from train import train
from config import CONFIG
from single_process_train import train
from utils.logger import plot_csv_log
import random
import numpy as np
import torch

seed = 0

if __name__ == '__main__':
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    CONFIG['seed'] = seed
    CONFIG['save_path'] = f'./checkpoints/seed_{seed}.pt'

    train(seed)
    plot_csv_log(f'./logs/train_seed_{seed}.csv')