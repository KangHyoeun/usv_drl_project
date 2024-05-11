import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# PID 매개변수
Kp = 2.0
Ki = 0.1
Kd = 0.01

# 시스템의 불확실한 매개변수
a = 1.0  # nominal value
a_uncertainty = 0.2  # 20% uncertainty

# 시스템 동적 방정식 정의
def system_dynamics(x, t, u, a):
    # 불확실한 매개변수를 포함한 시스템
    dxdt = -a * x + u
    return dxdt

# PID 컨트롤러 정의
def pid_controller(t, error, errors, dt):
    # P-term
    P_out = Kp * error
    # I-term
    I_out = Ki * sum(errors) * dt
    # D-term
    D_out = Kd * (errors[-1] - errors[-2]) / dt if len(errors) > 1 else 0
    # PID 출력
    u = P_out + I_out + D_out
    return u

# 목표값
set_point = 1

# 시뮬레이션 초기 조건 및 설정
x0 = 0  # 초기 상태
dt = 0.01  # 시간 간격
T = 10  # 총 시뮬레이션 시간
t = np.linspace(0, T, int(T/dt) + 1)  # 시간 배열
x = np.zeros(len(t))  # 상태 초기화
u = np.zeros(len(t))  # 제어 입력 초기화
errors = []  # 오류 초기화

# 시뮬레이션
for i in range(1, len(t)):
    # 오류 계산
    error = set_point - x[i-1]
    errors.append(error)
    # PID 제어기 업데이트
    u[i] = pid_controller(t[i], error, errors, dt)
    # 시스템 업데이트
    a_actual = a + np.random.uniform(-a_uncertainty, a_uncertainty)
    x[i] = odeint(system_dynamics, x[i-1], [t[i-1], t[i]], args=(u[i], a_actual))[-1]

# 결과 시각화
plt.figure(figsize=(12, 5))

# 제어 입력 시각화
plt.subplot(1, 2, 1)
plt.plot(t, u, 'r-', linewidth=2, label='Control input (u)')
plt.xlabel('Time')
plt.ylabel('Control Input')
plt.title('Control Input Over Time')
plt.grid(True)

# 시스템 상태 시각화
plt.subplot(1, 2, 2)
plt.plot(t, x, 'b-', linewidth=2, label='System Output (x)')
plt.plot(t, np.ones(len(t)) * set_point, 'k--', label='Set Point')
plt.xlabel('Time')
plt.ylabel('System Output')
plt.title('System Output Over Time')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
