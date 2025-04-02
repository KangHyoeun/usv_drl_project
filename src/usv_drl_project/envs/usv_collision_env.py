import gymnasium as gym
import numpy as np
import math
import random
from gymnasium import spaces
from utils.gridmap_utils import generate_grid_map
from utils.encounter_classifier import classify_encounter
from utils.cpa_utils import compute_cpa, is_risk
from utils.vo_utils import is_inside_vo, classify_velocity_region
from utils.reward_utils import compute_avoidance_reward
from utils.guidance import vector_field_guidance
from utils.control import steering_controller
from utils.angle import ssa

class USVCollisionEnv(gym.Env):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.grid_size = (84, 84)
        self.observation_space = spaces.Dict({
            "grid_map": spaces.Box(low=0, high=1, shape=(3, *self.grid_size), dtype=np.float32),
            "state_vec": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        })
        self.action_space = spaces.Discrete(3)
        self.prev_state = None
        self.prev_action = None
        self.state = None
        self.in_avoidance = False
        self.avoid_action = 0
        self.prev_psi = None

    def reset(self, seed=None, options=None):
        self.t = 0
        self.done = False
        self.in_avoidance = False
        self.avoid_action = 0
        self.state = self._init_state()
        self.prev_state = self.state.copy()
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        if not self.in_avoidance:
            for obs in self.state['obstacles']:
                dcpa, tcpa = compute_cpa(self.state, obs)
                if is_risk(dcpa, tcpa):
                    tcpa_exp = self.config.get('tcpa_exploration_factor', 0.5) * 20.0
                    if tcpa < tcpa_exp:
                        self.in_avoidance = True
                        self.avoid_action = random.choice([1, 2])
                        break

        if self.in_avoidance:
            action = self.avoid_action

        self._apply_action(action)
        self._simulate_dynamics()
        reward, terminated = self._compute_reward_and_done(action)
        self.t += 1

        obs = self._get_obs()
        self.prev_state = self.state.copy()
        self.prev_action = action
        return obs, reward, terminated, False, {}

    def _get_obs(self):
        grid_map = generate_grid_map(self.state, self.config)
        state_vec = np.array([
            self.state["u"], self.state["v"], self.state["r"],
            self.state["rpm1"], self.state["rpm2"],
            self._estimate_env_bias()
        ])
        return {"grid_map": grid_map, "state_vec": state_vec}

    def _estimate_env_bias(self):
        if self.prev_state is None:
            return 0.0
        dt = 0.1
        u = self.prev_state['u']
        psi = self.prev_state['psi']
        r = (self.prev_state['rpm2'] - self.prev_state['rpm1']) * 1.0
        psi_pred = psi + r * dt
        dx_pred = u * math.cos(psi_pred) * dt
        dy_pred = u * math.sin(psi_pred) * dt
        x_expected = self.prev_state['x'] + dx_pred
        y_expected = self.prev_state['y'] + dy_pred
        dx_err = self.state['x'] - x_expected
        dy_err = self.state['y'] - y_expected
        return np.linalg.norm([dx_err, dy_err])

    def _compute_reward_and_done(self, action):
        x, y = self.state['x'], self.state['y']
        psi = self.state['psi']
        done = False
        reward = 0.0
        e_cross = abs(x)
        reward += math.exp(-e_cross)
        env_bias = self._estimate_env_bias()

        for obs in self.state['obstacles']:
            dx = x - obs['x']
            dy = y - obs['y']
            dist = math.hypot(dx, dy)

            encounter = classify_encounter(self.state, obs)
            dcpa, tcpa = compute_cpa(self.state, obs)
            risk = is_risk(dcpa, tcpa)

            if dist < 5.0:
                reward = -1.0
                done = True
                break

            if risk:
                ref_psi = math.atan2(obs['y'] - y, obs['x'] - x)
                delta_psi = abs(self.state['psi'] - ref_psi)
                reward += compute_avoidance_reward(encounter, delta_psi, tcpa, env_bias)

                # ✅ VO 방향 기반 보상
                pA = np.array([x, y])
                vA = np.array([self.state['u'] * math.cos(psi), self.state['u'] * math.sin(psi)])
                pB = np.array([obs['x'], obs['y']])
                vB = np.array([obs['vx'], obs['vy']])
                v_rel = vA - vB
                p_rel = pB - pA

                if is_inside_vo(vA, pA, pB, 2.0, 2.0, vB):
                    region = classify_velocity_region(v_rel, p_rel)
                    if region == 'V1':
                        reward -= 0.5
                    elif region == 'V2':
                        reward += 0.5

        return reward, done

    def _init_state(self):
        return {
            "u": 1.5, "v": 0.0, "r": 0.0,
            "rpm1": 0.5, "rpm2": 0.5,
            "x": 0.0, "y": 0.0, "psi": 0.0,
            "obstacles": self._spawn_obstacles()
        }

    def _spawn_obstacles(self):
        types = ['HO', 'GW', 'SO', 'OT', 'Static']
        selected = random.choices(types, k=3)
        obs_list = []
        for t in selected:
            if t == 'HO':
                obs_list.append({"x": 0, "y": 50, "vx": 0, "vy": -0.5, "dynamic": True, "psi": -np.pi/2})
            elif t == 'GW':
                obs_list.append({"x": 15, "y": 50, "vx": 0, "vy": -0.3, "dynamic": True, "psi": -np.pi/2})
            elif t == 'SO':
                obs_list.append({"x": -15, "y": 50, "vx": 0, "vy": -0.3, "dynamic": True, "psi": -np.pi/2})
            elif t == 'OT':
                obs_list.append({"x": 0, "y": 20, "vx": 0.0, "vy": 0.5, "dynamic": True, "psi": np.pi/2})
            elif t == 'Static':
                obs_list.append({"x": random.uniform(-20, 20), "y": random.uniform(30, 70),
                                 "vx": 0.0, "vy": 0.0, "dynamic": False, "psi": 0.0})
        return obs_list

    def _simulate_dynamics(self):
        rpm_avg = (self.state['rpm1'] + self.state['rpm2']) / 2
        self.state['u'] = rpm_avg * 3.0
        self.state['r'] = (self.state['rpm2'] - self.state['rpm1']) * 1.0
        dt = 0.1
        self.state['psi'] += self.state['r'] * dt
        dx = self.state['u'] * math.cos(self.state['psi']) * dt
        dy = self.state['u'] * math.sin(self.state['psi']) * dt
        self.state['x'] += dx
        self.state['y'] += dy

    def _apply_action(self, action):
        pA = np.array([self.state['x'], self.state['y']])
        vA = np.array([
            self.state['u'] * math.cos(self.state['psi']),
            self.state['u'] * math.sin(self.state['psi'])
        ])
        rA = 2.0

        for obs in self.state['obstacles']:
            pB = np.array([obs['x'], obs['y']])
            vB = np.array([obs['vx'], obs['vy']])
            rB = 2.0
            if is_inside_vo(vA, pA, pB, rA, rB, vB):
                v_rel = vA - vB
                p_rel = pB - pA
                region = classify_velocity_region(v_rel, p_rel)
                if action == 1 and region == 'V1':
                    action = 2
                    break

        if action == 0:
            y = self.state['x']
            psi = self.state['psi']
            dt = 0.1

            chi_inf_rad = math.radians(self.config.get("vfg_chi_inf_deg", 45.0))
            chi_path_rad = math.radians(self.config.get("vfg_chi_path_deg", 0.0))
            k = self.config.get("vfg_k", 1.0)

            chi_d = vector_field_guidance(y, chi_path=chi_path_rad, chi_inf=chi_inf_rad, k=k)

            kp = self.config.get("steering_kp", 1.0)
            kd = self.config.get("steering_kd", 0.2)

            delta = steering_controller(chi_d, psi, self.prev_psi, dt, kp, kd)
            self.prev_psi = psi  # 업데이트

            self.state['rpm1'] = 0.5 - delta
            self.state['rpm2'] = 0.5 + delta
        elif action == 1:
            self.state['rpm1'] = 0.3
            self.state['rpm2'] = 0.6
        elif action == 2:
            self.state['rpm1'] = 0.6
            self.state['rpm2'] = 0.3
