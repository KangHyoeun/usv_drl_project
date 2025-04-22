# src/usv_drl_project/utils/gridmap_utils.py
import numpy as np
import cv2

def generate_grid_map(own_state, obs_state, config):
    grid_size = config['grid_size']  # (H, W) = (84, 84)
    grid_map = np.zeros((3, *grid_size), dtype=np.float32)

    # layer 0: path
    grid_map[0] = _draw_path_layer(own_state, grid_size)

    # layer 1: dynamic obstacles
    grid_map[1] = _draw_dynamic_obstacles(obs_state, grid_size)

    # layer 2: static obstacles
    grid_map[2] = _draw_static_obstacles(obs_state, grid_size)

    return grid_map

def _draw_path_layer(state, grid_size):
    img = np.zeros(grid_size, dtype=np.uint8)
    cx, cy = grid_size[1]//2, grid_size[0]//2
    cv2.line(img, (cx, 0), (cx, grid_size[0]-1), 255, 3)
    return img.astype(np.float32) / 255.0

def _draw_dynamic_obstacles(state, grid_size):
    img = np.zeros(grid_size, dtype=np.uint8)
    for obs in state:
        if not obs['dynamic']: continue
        x, y = _world_to_grid(obs['x'], obs['y'], grid_size)
        cv2.circle(img, (x, y), 3, 255, -1)
        vx = obs['u'] * np.sin(obs['psi'])
        vy = obs['u'] * np.cos(obs['psi'])
        end_x = int(x + vx * 30)
        end_y = int(y - vy * 30)
        cv2.line(img, (x, y), (end_x, end_y), 200, 1)
    return img.astype(np.float32) / 255.0

def _draw_static_obstacles(state, grid_size):
    img = np.zeros(grid_size, dtype=np.uint8)
    for obs in state:
        if obs['dynamic']: continue
        x, y = _world_to_grid(obs['x'], obs['y'], grid_size)
        cv2.circle(img, (x, y), 3, 255, -1)
    return img.astype(np.float32) / 255.0

def _world_to_grid(x, y, grid_size):
    scale = grid_size[0] / 336.0  # assuming square: 84 / 336
    col = int((y + 168) * scale)  # +y가 오른쪽 → grid column
    row = grid_size[0] - int((x + 168) * scale)  # +x가 위쪽 → flip y축
    col = np.clip(col, 0, grid_size[1] - 1)
    row = np.clip(row, 0, grid_size[0] - 1)
    return col, row