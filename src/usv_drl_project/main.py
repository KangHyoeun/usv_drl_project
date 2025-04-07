# src/usv_drl_project/main.py
# from train import train
from single_process_train import train
from utils.logger import plot_csv_log
from envs.usv_collision_env import USVCollisionEnv
from config import CONFIG

if __name__ == '__main__':
    env = USVCollisionEnv(CONFIG)  # 단일 환경 인스턴스 생성
    # test_env_compatibility(env)  # ✅ 호환성 검증 함수 추가
    train()
    plot_csv_log('./logs/train_log.csv')