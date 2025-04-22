# src/usv_drl_project/seed.py
from train import sweep_seeds
# from single_process_train import train
from utils.logger import plot_csv_log

if __name__ == '__main__':
    SEED_LIST = [20, 30, 40]  # 원하는 seed 목록
    sweep_seeds(SEED_LIST)
    for seed in SEED_LIST:
        plot_csv_log(f'./logs/train_seed_{seed}.csv')