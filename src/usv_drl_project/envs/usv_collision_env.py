import gymnasium as gym
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from gymnasium import spaces
from scipy.integrate import solve_ivp
from models.otter import otter
from utils.gridmap_utils import generate_grid_map
from utils.encounter_classifier import classify_encounter
from utils.cpa_utils import compute_cpa, is_risk
from utils.reward_utils import compute_avoidance_reward, compute_path_reward
from utils.guidance import vector_field_guidance, select_avoidance_heading, get_available_avoid_directions
from utils.control import r_max, zeta_d, wn_d, yaw_pid_controller, control_allocation
from utils.angle import ssa
from utils.gnc import ref_model

metadata = {"render_modes": ["human"], "render_fps": 10}

class USVCollisionEnv(gym.Env):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.grid_size = config.get('grid_size', (84, 84))
        self.dt = 0.1
        self.step_count = 0
        self.render_step = 0

        if 'env_seed' in self.config:
            self.np_random = np.random.default_rng(self.config['env_seed'])
            self.random = random.Random(self.config['env_seed'])

        self.observation_space = spaces.Dict({
            "grid_map": spaces.Box(low=0, high=1, shape=(3, *self.grid_size), dtype=np.float32),
            "state_vec": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
        })
        self.action_space = spaces.Discrete(3)

        self.own_state = None
        self.prev_state = None
        self.obs_state = None
        self.avoid_obs = None
        self.in_avoidance = False

        self.z_psi = 0.0
        self.r_d = 0.0
        self.a_d = 0.0
        self.n = np.array([0.0, 0.0])

        self.epsilon = 1.0
        self.max_episode_steps = config.get('max_episode_steps', 1000)

    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
            self.np_random = np.random.default_rng(seed)  # ✅ 고쳐야 함
            self.random = random.Random(seed)

        self.step_count = 0
        self.in_avoidance = False
        self.own_state = self._init_own_state()
        self.prev_state = self.own_state.copy()
        self.obs_state = self._spawn_obstacles()
        obs = self._get_obs()
        self._update_avoidance_info()
        if self.avoid_obs:
            self.tcpa_init = self.tcpa_now
            self.tcpa_exp = self.epsilon * self.tcpa_init
        else:
            self.tcpa_init = None
            self.tcpa_exp = None
        return obs, {}

    def _init_own_state(self):
        return {
            "u": 0.0, "v": 0.0, "w": 0.0, "p": 0.0, "q": 0.0, "r": 0.0, 
            "x": 0.0, "y": 0.0, "z": 0.0, "phi": 0.0, "theta": 0.0, "psi": 0.0,
            "rpm1": 0.0, "rpm2": 0.0,
            "enc_type": 0.0,
        }

    def _spawn_obstacles(self):
        scenario = self.config.get('scenario')
        if scenario == 'HO':
            return self._spawn_headon_obstacles()
        elif scenario == 'GW':
            return self._spawn_giveway_obstacles()
        elif scenario == 'SO':
            return self._spawn_standon_obstacles()
        elif scenario == 'OT':
            return self._spawn_overtaking_obstacles()
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
    
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def _get_obs(self):
        grid_map = generate_grid_map(self.own_state, self.obs_state, self.config)
        state_vec = np.array([
            self.own_state["u"] / self.config['max_speed'],          # 정방향 속도 (0~1)
            self.own_state["v"] / self.config['max_speed'],          # 측면 속도 (보통 0에 가까움)
            self.own_state["r"] / self.config['max_yaw_rate'],       # 회전속도 (-1 ~ 1 정도)
            self.own_state["rpm1"] / self.config['max_rpm'],         # 좌 프로펠러 RPM (-1 ~ 1)
            self.own_state["rpm2"] / self.config['max_rpm'],         # 우 프로펠러 RPM (-1 ~ 1)
            self._estimate_env_bias(self.dt)
        ])
        return {
            "grid_map": grid_map,
            "state_vec": state_vec,
        }
    
    def _update_avoidance_info(self):
        candidates = []
        for i, target_obs in enumerate(self.obs_state):
            dcpa, tcpa = compute_cpa(self.own_state, target_obs)
            is_static = not target_obs.get("dynamic", True)
            risk = is_risk(dcpa, tcpa, dcpa_thresh=30.0, tcpa_thresh=60.0, is_static=is_static)
            # print(f"[RISK] obs[{i}] static={is_static}, DCPA={dcpa:.2f}, TCPA={tcpa:.2f}, RISK={risk}")
            if risk:
                candidates.append((target_obs, dcpa, tcpa))
        if candidates:
            candidates.sort(key=lambda x: x[2])  # TCPA 기준
            self.avoid_obs, self.dcpa_now, self.tcpa_now = candidates[0]
            encounter_type = classify_encounter(self.own_state, self.avoid_obs)
        else:
            self.avoid_obs = None
            encounter_type = 'None'

        enc_map = {
            "SF": 0, "HO": 1, "GW": 2, "OT": 3, "SO": 4, "Static": 5, "None": 0
        }
        self.own_state["enc_type"] = enc_map.get(encounter_type, 0)

    def step(self, action):
        # 0. 이전 상태 저장 (optional)
        self.prev_state = self.own_state.copy()

        # 1. 조우 분석: 위험 정보 갱신, 조우유형 판단
        self._update_avoidance_info()

        # 2. TCPA 기반 회피기동 전환
        if not self.in_avoidance and self.avoid_obs is not None:
            if self.tcpa_exp is not None and self.tcpa_now <= self.tcpa_exp:
                if action == 0:
                    action = self.random.choice([1, 2])  # 강제 회피 개입
                self.in_avoidance = True  # 어떤 경우든 회피 모드 진입

        # 3. 제어 입력 계산
        self._compute_control_inputs(action, self.dt)

        # 4. 동역학 시뮬레이션 (자선 + 타선 등)
        self._simulate_dynamics(self.dt)
        self._simulate_obstacles(self.dt)

        # 5. 보상 계산 terminated 정의
        reward, terminated = self._compute_reward_and_done(action)

        # 6. truncated 정의
        truncated = self.step_count >= self.max_episode_steps

        # 7. 관측값 갱신
        obs = self._get_obs()

        self.step_count += 1

        info = {
            "enc_type": self.own_state["enc_type"],
            "tcpa_now": getattr(self, "tcpa_now", None),
            "dcpa_now": getattr(self, "dcpa_now", None),
            "in_avoidance": self.in_avoidance,
        }

        return obs, reward, terminated, truncated, info

    def _estimate_env_bias(self, dt):
        if self.prev_state is None:
            return 0.0
        
        u = self.prev_state['u']
        psi = self.prev_state['psi']
        r = self.prev_state['r']
        psi_pred = psi + r * dt
        dx_pred = u * math.cos(psi_pred) * dt
        dy_pred = u * math.sin(psi_pred) * dt
        x_expected = self.prev_state['x'] + dx_pred
        y_expected = self.prev_state['y'] + dy_pred
        dx_err = self.own_state['x'] - x_expected
        dy_err = self.own_state['y'] - y_expected
        return np.linalg.norm([dx_err, dy_err])

    def _compute_reward_and_done(self, action):
        x, y, psi, enc_type = self.own_state['x'], self.own_state['y'], self.own_state['psi'], self.own_state["enc_type"]
        terminated = False
        reward = 0.0
        
        if action == 0:
            e_cross = np.abs(y)
            reward += compute_path_reward(e_cross, krpath=-1.0)
        elif self.avoid_obs is not None:
            dx = x - self.avoid_obs['x']
            dy = y - self.avoid_obs['y']
            dist = math.hypot(dx, dy)

            if dist < 3.0:
                return -1.0, True  # 충돌 → 종료

            dcpa, tcpa = compute_cpa(self.own_state, self.avoid_obs)
            is_static = not self.avoid_obs.get("dynamic", True)

            if is_risk(dcpa, tcpa, dcpa_thresh=30.0, tcpa_thresh=60.0, is_static=is_static):
                if hasattr(self, "chi_ref_prev"):
                    # 회피변침각 = 현재 psi - 과거 경로추종 침로
                    delta_psi = np.abs(psi - self.chi_ref_prev)
                    reward += compute_avoidance_reward(enc_type, delta_psi, tcpa)
            else:
                reward -= 0.05  # 회피 실패 패널티
        else:
            reward -= 0.05  # 불필요한 회피 패널티

        return reward, terminated
    
    def _simulate_dynamics(self, dt):
        keys = ["u", "v", "w", "p", "q", "r", "x", "y", "z", "phi", "theta", "psi"]
        x0 = np.array([self.own_state[k] for k in keys])
        n_input = np.array([self.own_state["rpm1"], self.own_state["rpm2"]])

        def f(t, x):
            dx, *_ = otter(x, n_input, 25.0, np.array([0.05, 0, -0.35]), 0.3, np.deg2rad(30))
            return dx
        sol = solve_ivp(f, [0, dt], x0, method='RK45', t_eval=[dt])
        x_next = sol.y[:, -1]
        # 상태값 업데이트
        for i, key in enumerate(keys):
            self.own_state[key] = x_next[i]

    def _simulate_obstacles(self, dt):
        for target_obs in self.obs_state:
            if target_obs['dynamic']:  # 동적 장애물만 이동
                target_obs['x'] += target_obs['u'] * math.cos(target_obs['psi']) * dt  
                target_obs['y'] += target_obs['u'] * math.sin(target_obs['psi']) * dt  # 직선 운동

    def _compute_control_inputs(self, action, dt):
        y, psi, r = self.own_state['y'], self.own_state['psi'], self.own_state["r"]

        if action == 0:
            chi_inf_rad = math.radians(self.config.get("vfg_chi_inf_deg", 45.0))
            chi_path_rad = math.radians(self.config.get("vfg_chi_path_deg", 0.0))
            k = self.config.get("vfg_k", 1.0)

            psi_ref = vector_field_guidance(y, chi_path_rad, chi_inf_rad, k)
            self.chi_ref_prev = psi_ref

        elif action in [1, 2]:
            if self.avoid_obs is None:
                # 회피할 대상이 없음 → 경로추종 유도기 fallback
                chi_inf_rad = math.radians(self.config.get("vfg_chi_inf_deg", 45.0))
                chi_path_rad = math.radians(self.config.get("vfg_chi_path_deg", 0.0))
                k = self.config.get("vfg_k", 1.0)
                psi_ref = vector_field_guidance(y, chi_path_rad, chi_inf_rad, k)
            else:
                direction = 'left' if action == 1 else 'right'
                psi_ref = select_avoidance_heading(self.own_state, self.avoid_obs, direction)

        # Reference model propagation
        psi_d = psi
        psi_d, self.r_d, self.a_d = ref_model(psi_d, self.r_d, self.a_d, psi_ref, r_max, zeta_d, wn_d, dt, 1)

        tau_X, tau_N, self.z_psi = yaw_pid_controller(psi, psi_d, r, self.a_d, self.r_d, self.z_psi)

        n_c, self.n = control_allocation(tau_X, tau_N, self.n)

        self.own_state['rpm1'] = self.n[0]
        self.own_state['rpm2'] = self.n[1]

    def _spawn_headon_obstacles(self):
        targets = []
        max_trials = 1000  # 무한 루프 방지

        while len(targets) < self.config.get('n_dynamic_obs', 4) and max_trials > 0:
            max_trials -= 1

            # 장애물 무작위 위치 및 heading 설정
            x = self.np_random.uniform(40., 80.)
            y = self.np_random.uniform(-40., 40.)
            u = self.np_random.uniform(1.0, 1.5)
            psi = self.np_random.uniform(-np.pi, np.pi)

            # 자선 기준 상대 위치 (자선은 항상 (0, 0))
            dx, dy = x, y
            rel_angle = math.atan2(dy, dx)        # NED 기준: atan2(X, Y)
            rel_heading = psi                     # 자선이 heading=0이므로 그대로 사용

            heading_cond = (abs(rel_heading) > 7 * np.pi / 8)
            angle_cond = (abs(rel_angle) <= np.pi / 2)

            if heading_cond and angle_cond:
                targets.append({
                    'x': x,
                    'y': y,
                    'u': u,
                    'psi': psi,
                    'dynamic': True
                })

        targets += self._spawn_static_obstacles()
        return targets
    
    def _spawn_giveway_obstacles(self):
        targets = []
        max_trials = 1000

        while len(targets) < self.config.get('n_dynamic_obs', 4) and max_trials > 0:
            max_trials -= 1

            x = self.np_random.uniform(20., 60.)
            y = self.np_random.uniform(-70., 10.)
            u = self.np_random.uniform(1.0, 1.5)
            psi = self.np_random.uniform(-np.pi, np.pi)

            dx, dy = x, y
            rel_angle = math.atan2(dy, dx)
            rel_heading = psi

            cond1 = (-np.pi/8 <= rel_angle < 5*np.pi/8) and (-7*np.pi/8 <= rel_heading < -3*np.pi/8)
            cond2 = (abs(rel_angle) > 5 * np.pi / 8) and (-np.pi/2 <= rel_heading < -3*np.pi/8)

            if cond1 or cond2:
                targets.append({
                    'x': x,
                    'y': y,
                    'u': u,
                    'psi': psi,
                    'dynamic': True
                })

        targets += self._spawn_static_obstacles()
        return targets

    def _spawn_standon_obstacles(self):
        targets = []
        max_trials = 1000

        while len(targets) < self.config.get('n_dynamic_obs', 4) and max_trials > 0:
            max_trials -= 1

            x = self.np_random.uniform(20., 60.)
            y = self.np_random.uniform(-10., 70.)
            u = self.np_random.uniform(1.0, 1.5)
            psi = self.np_random.uniform(-np.pi, np.pi)

            dx, dy = x, y
            rel_angle = math.atan2(dy, dx)        # NED 기준: atan2(X, Y)
            rel_heading = psi

            cond1 = (-5*np.pi/8 <= rel_angle < np.pi/8) and (3*np.pi/8 <= rel_heading < 7*np.pi/8)
            cond2 = (abs(rel_angle) > 5*np.pi/8) and (3*np.pi/8 <= rel_heading < np.pi/2)

            if cond1 or cond2:
                targets.append({
                    'x': x,
                    'y': y,
                    'u': u,
                    'psi': psi,
                    'dynamic': True
                })

        targets += self._spawn_static_obstacles()
        return targets

    def _spawn_overtaking_obstacles(self):
        targets = []
        max_trials = 1000

        while len(targets) < self.config.get('n_dynamic_obs', 6) and max_trials > 0:
            max_trials -= 1

            x = self.np_random.uniform(20., 60.)   # 자선 앞쪽 (타선을 추월)
            y = self.np_random.uniform(-40., 40.)
            u = self.np_random.uniform(0.5, 1.0)    # 느리게 움직이는 타선
            psi = self.np_random.uniform(-3*np.pi/8, 3*np.pi/8)  # 자선과 거의 같은 방향 (북쪽)

            if abs(psi) <= 3 * np.pi / 8:
                targets.append({
                    'x': x,
                    'y': y,
                    'u': u,
                    'psi': psi,
                    'dynamic': True
                })

        targets += self._spawn_static_obstacles()
        return targets
    
    def _spawn_static_obstacles(self):
        statics = []
        for _ in range(self.config.get('n_static_obs', 2)):
            statics.append({
                'x': self.np_random.uniform(40., 80.),
                'y': self.np_random.uniform(-15., 15.),
                'u': 0.0,
                'psi': 0.0,
                'dynamic': False
            })
        return statics

    def render(self, mode='human'):
        import matplotlib.pyplot as plt

        # 궤적 저장 초기화
        if not hasattr(self, 'own_traj'):
            self.own_traj = []
            self.obs_traj = [[] for _ in self.obs_state]

        # 현재 자선 위치 추가
        self.own_traj.append((self.own_state["x"], self.own_state["y"]))

        # 현재 장애물 위치 추가
        for i, obs in enumerate(self.obs_state):
            if i >= len(self.obs_traj):
                self.obs_traj.append([])
            self.obs_traj[i].append((obs["x"], obs["y"]))

        plt.clf()

        # 자선 궤적 그리기
        xs, ys = zip(*self.own_traj)
        plt.plot(ys, xs, 'b-', label='Own ship trajectory')  # NED 기준
        plt.plot(ys[-1], xs[-1], 'bo', label='Own ship current')
        psi = self.own_state["psi"]
        plt.arrow(ys[-1], xs[-1], math.sin(psi), math.cos(psi), head_width=1.0, color='b')

        # 장애물 궤적 그리기
        for i, traj in enumerate(self.obs_traj):
            if len(traj) == 0:
                continue
            xs_o, ys_o = zip(*traj)
            color = 'r' if self.obs_state[i]["dynamic"] else 'k'
            marker = 'x' if self.obs_state[i]["dynamic"] else 's'
            plt.plot(ys_o, xs_o, linestyle='--', color=color, alpha=0.7)
            plt.plot(ys_o[-1], xs_o[-1], marker=marker, color=color)

        plt.xlim(-100, 100)
        plt.ylim(-100, 100)
        plt.gca().set_aspect('equal')
        plt.xlabel("East [m]")
        plt.ylabel("North [m]")
        plt.grid(True)
        plt.legend(loc='upper right')
        # 저장
        filename = f"render_log/frame_{self.render_step:04d}.png"
        plt.savefig(filename)
        self.render_step += 1
        plt.pause(0.001)

    def close(self):
        pass  # SubprocVecEnv가 호출할 수 있도록 dummy로라도 정의
