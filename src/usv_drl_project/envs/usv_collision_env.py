import gymnasium as gym
import numpy as np
import math
import random
from gymnasium import spaces
from scipy.integrate import solve_ivp
from models.otter import otter
from utils.gridmap_utils import generate_grid_map
from utils.encounter_classifier import classify_encounter
from utils.cpa_utils import compute_cpa, is_risk
from utils.vo_utils import select_avoidance_heading, get_available_avoid_directions
from utils.reward_utils import compute_avoidance_reward, compute_path_reward
from utils.guidance import vector_field_guidance
from utils.control import Binv, T, K, Kp, Td, Ti, T_n, r_max, zeta_d, wn_d
from utils.angle import ssa
from utils.gnc import ref_model

class USVCollisionEnv(gym.Env):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.grid_size = (84, 84)
        self.observation_space = spaces.Dict({
            "grid_map": spaces.Box(low=0, high=1, shape=(3, *self.grid_size), dtype=np.float32),
            "state_vec": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
        })
        self.action_space = spaces.Discrete(3)
        self.max_episode_steps = config.get("max_episode_steps", 1000)  # default 1000
        self.t = 0
        self.prev_state = None
        self.state = None
        self.in_avoidance = False
        self.avoid_obs = None
        self.avoid_action = 0
        self.z_psi = 0.0
        self.psi_d = 0.0
        self.psi_ref = self.psi_d
        self.chi_ref_prev = self.psi_ref
        self.r_d = 0.0
        self.a_d = 0.0
        self.n = np.array([0.0, 0.0])

    def reset(self, seed=None, options=None):
        self.t = 0
        self.in_avoidance = False
        self.avoid_obs = None
        self.avoid_action = 0
        self.tcpa_init = None
        self.state = self._init_state()
        # 장애물 정보 확인
        print("Initial obstacles:", self.state['obstacles'])
        self.prev_state = self.state.copy()
        obs = self._get_obs()
        return obs, {}

    def step(self, action):

        if self.in_avoidance and self.avoid_obs is None:
            print("[FIXED] in_avoidance True인데 avoid_obs 없어서 초기화함")
            self.in_avoidance = False
            self.avoid_action = 0

        if not self.in_avoidance:
            for obs in self.state['obstacles']:
                dcpa, tcpa = compute_cpa(self.state, obs)
                if is_risk(dcpa, tcpa):
                    tcpa_exp = self.epsilon * tcpa
                    if tcpa < tcpa_exp:
                        pA = np.array([self.state['x'], self.state['y']])
                        vA = np.array([
                            self.state['u'] * math.cos(self.state['psi']),
                            self.state['u'] * math.sin(self.state['psi'])
                        ])
                        pB = np.array([obs['x'], obs['y']])
                        vB = np.array([obs['vx'], obs['vy']])
                        can_left, can_right = get_available_avoid_directions(vA, pA, pB, vB)
                        self.avoid_obs = obs  # 장애물 할당

                        if can_left and can_right:
                            self.avoid_action = random.choice([1, 2])
                        elif can_left:
                            self.avoid_action = 1
                        elif can_right:
                            self.avoid_action = 2
                        else:
                            self.avoid_action = random.choice([1, 2])  # fallback

                        self.in_avoidance = True
                        break

        if self.in_avoidance:
            action = self.avoid_action

        if action in [1, 2]:
            obs = self.avoid_obs
            if obs is None:
                print("[WARN] in_avoidance=True but avoid_obs is None — fallback to path following")
                action = 0
                self.in_avoidance = False
                self.avoid_action = 0
                self.avoid_obs = None

        if action == 0:
            self.chi_ref_prev = self.psi_ref

        self._compute_control_inputs(action)

        self._simulate_dynamics()
        self._simulate_obstacles()
        reward, terminated = self._compute_reward_and_done(action)

        self.t += 1
        truncated = self.t >= self.max_episode_steps

        done = terminated or truncated

        obs = self._get_obs()
        self.prev_state = self.state.copy()
        return obs, reward, terminated, truncated, {}
    
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def _get_obs(self):
        grid_map = generate_grid_map(self.state, self.config)
        state_vec = np.array([
            self.state["u"], self.state["v"], self.state["r"],
            self.state["rpm1"], self.state["rpm2"],
            self._estimate_env_bias()
        ])

        # Encounter type 판단 (기본적으로 가장 위험한 장애물 기준)
        if self.avoid_obs is not None:
            encounter_type = classify_encounter(self.state, self.avoid_obs)
        else:
            encounter_type = "None"

        return {
            "grid_map": grid_map,
            "state_vec": state_vec,
            "encounter_type": encounter_type  # ✅ 추가됨
        }

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
        terminated = False
        reward = 0.0
        e_cross = abs(x)

        if action == 0:
            reward += compute_path_reward(e_cross, krpath=-1.0)
        else:
            obs = self.avoid_obs  # 단일 객체로 사용
            if obs is not None:
                dx = x - obs['x']
                dy = y - obs['y']
                dist = math.hypot(dx, dy)

                if dist < 2.0:
                    return -1.0, True  # 충돌 → 종료

                dcpa, tcpa = compute_cpa(self.state, obs)
                encounter_type = classify_encounter(self.state, obs)
                risk = is_risk(dcpa, tcpa)
                if risk:
                    # 회피변침각 = 현재 psi - 과거 경로추종 침로
                    delta_psi = abs(psi - self.chi_ref_prev)
                    reward += compute_avoidance_reward(encounter_type, delta_psi, tcpa)

        return reward, terminated

    def _init_state(self):
        return {
            "u": 0.0, "v": 0.0, "w": 0.0, "p": 0.0, "q": 0.0, "r": 0.0, 
            "x": 0.0, "y": 0.0, "z": 0.0, "phi": 0.0, "theta": 0.0, "psi": 0.0,
            "rpm1": 0.0, "rpm2": 0.0,
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
        dt = 0.1
        keys = ["u", "v", "w", "p", "q", "r", "x", "y", "z", "phi", "theta", "psi"]
        x0 = np.array([self.state[k] for k in keys])
        n_input = np.array([self.state["rpm1"], self.state["rpm2"]])

        def f(t, x):
            dx, *_ = otter(x, n_input, 25.0, np.array([0.05, 0, -0.35]), 0.3, np.deg2rad(30))
            return dx
        sol = solve_ivp(f, [0, dt], x0, method='RK45', t_eval=[dt])
        x_next = sol.y[:, -1]
        # 상태값 업데이트
        for i, key in enumerate(keys):
            self.state[key] = x_next[i]

    def _simulate_obstacles(self, dt=0.1):
        for obs in self.state['obstacles']:
            if obs['dynamic']:  # 동적 장애물만 이동
                obs['x'] += obs['vx'] * dt  # 속도 * 시간
                obs['y'] += obs['vy'] * dt  # 직선 운동


    def _compute_control_inputs(self, action):
        psi = self.state['psi']
        r = self.state['r']
        dt = 0.1
        
        if action == 0:

            chi_inf_rad = math.radians(self.config.get("vfg_chi_inf_deg", 45.0))
            chi_path_rad = math.radians(self.config.get("vfg_chi_path_deg", 0.0))
            k = self.config.get("vfg_k", 1.0)

            self.psi_ref = vector_field_guidance(self.state['x'], chi_path=chi_path_rad, chi_inf=chi_inf_rad, k=k)

            # Reference model propagation
            self.psi_d, self.r_d, self.a_d = ref_model(self.psi_d, self.r_d, self.a_d, self.psi_ref, r_max, zeta_d, wn_d, dt, 1)

            # PID heading (yaw moment) autopilot and forward thrust
            tau_X = 100                              # Constant forward thrust
            tau_N = (T/K) * self.a_d + (1/K) * self.r_d - Kp * (ssa(psi - self.psi_d) + Td * (r - self.r_d) + (1/Ti) * self.z_psi) # Derivative and integral terms

            # Control allocation
            u = Binv @ np.array([tau_X, tau_N])      # Compute control inputs for propellers
            n_c = np.sign(u) * np.sqrt(np.abs(u))  # Convert to required propeller speeds

            # Euler's method
            self.n = self.n + dt/T_n * (n_c - self.n)              # Update propeller speeds
            self.z_psi = self.z_psi + dt * ssa(psi - self.psi_d)   # Update integral state

            self.state['rpm1'] = self.n[0]
            self.state['rpm2'] = self.n[1]

        elif action in [1, 2]:
            # 장애물 정보 가져오기
            obs = self.avoid_obs
            pA = np.array([self.state['x'], self.state['y']])
            vA = np.array([
                self.state['u'] * math.cos(psi),
                self.state['u'] * math.sin(psi)
            ])
            pB = np.array([obs['x'], obs['y']])
            vB = np.array([obs['vx'], obs['vy']])

            direction = 'left' if action == 1 else 'right'
            self.psi_ref = select_avoidance_heading(vA, pA, pB, vB, direction)
            # Reference model propagation
            self.psi_d, self.r_d, self.a_d = ref_model(self.psi_d, self.r_d, self.a_d, self.psi_ref, r_max, zeta_d, wn_d, dt, 1)

            # PID heading (yaw moment) autopilot and forward thrust
            tau_X = 100                              # Constant forward thrust
            tau_N = (T/K) * self.a_d + (1/K) * self.r_d - Kp * (ssa(psi - self.psi_d) + Td * (r - self.r_d) + (1/Ti) * self.z_psi) # Derivative and integral terms

            # Control allocation
            u = Binv @ np.array([tau_X, tau_N])      # Compute control inputs for propellers
            n_c = np.sign(u) * np.sqrt(np.abs(u))  # Convert to required propeller speeds

            # Euler's method
            self.n = self.n + dt/T_n * (n_c - self.n)              # Update propeller speeds
            self.z_psi = self.z_psi + dt * ssa(psi - self.psi_d)   # Update integral state

            self.state['rpm1'] = self.n[0]
            self.state['rpm2'] = self.n[1]
