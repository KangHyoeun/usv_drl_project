# src/usv_drl_project/main.py
from train import train
from utils.logger import plot_csv_log

if __name__ == '__main__':
    train()
    plot_csv_log('./logs/train_log.csv')