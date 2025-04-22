import torch
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.usv_collision_env import USVCollisionEnv
from models.dueling_dqn import DuelingDQN
from config import CONFIG

def make_env(config):
    return lambda: USVCollisionEnv(config)

def test():
    env = DummyVecEnv([make_env(CONFIG)])
    obs = env.reset()

    input_shape = (3, *CONFIG['grid_size'])
    state_vec_dim = 6
    n_actions = 3

    model = DuelingDQN(input_shape, state_vec_dim, n_actions).to(CONFIG['device'])
    model.load_state_dict(torch.load(CONFIG['save_path'], map_location=CONFIG['device']))
    model.eval()

    done = False
    total_reward = 0
    while not done:
        with torch.no_grad():
            grid = torch.tensor(obs['grid_map'], dtype=torch.float32).to(CONFIG['device'])
            vec = torch.tensor(obs['state_vec'], dtype=torch.float32).to(CONFIG['device'])
            action = torch.argmax(model(grid, vec), dim=1).item()

        obs, reward, done, info = env.step([action])
        reward = reward[0]
        done = done[0]
        total_reward += reward
        # env.render()  # matplotlib 기반이면 여기서 업데이트 가능

    print(f"Total test reward: {total_reward}")

if __name__ == '__main__':
    test()
