import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 시스템 매개변수
lambda_param = -2  # 슬라이딩 표면 기울기 매개변수
k = 5              # 스위칭 게인
b = 1              # 시스템의 제어 가능성 매개변수

# 시스템 동적 방정식 정의
def system_dynamics(x, t):
    x1, x2 = x
    s = x1 + lambda_param * x2
    f_x = 0  # 비선형 항이 없다고 가정
    d = 0    # 외란이 없다고 가정
    u = -(f_x + k * np.sign(s)) / b
    dx1dt = x2
    dx2dt = f_x + b * u + d
    return [dx1dt, dx2dt]

# 초기 조건 및 시뮬레이션 시간 설정
x0 = [0.5, 0]  # 초기 상태
t = np.linspace(0, 10, 1000)  # 시간 범위

# 시뮬레이션
solution = odeint(system_dynamics, x0, t)
x1 = solution[:, 0]
x2 = solution[:, 1]

# 결과 시각화
plt.figure(figsize=(12, 6))

# 시스템 상태 시각화
plt.subplot(1, 2, 1)
plt.plot(t, x1, label='$x_1$')
plt.plot(t, x2, label='$x_2$')
plt.xlabel('Time')
plt.ylabel('States')
plt.title('System States Over Time')
plt.legend()
plt.grid(True)

# 슬라이딩 표면 시각화
plt.subplot(1, 2, 2)
s = x1 + lambda_param * x2
plt.plot(t, s, label='Sliding Surface $s$')
plt.xlabel('Time')
plt.ylabel('$s$')
plt.title('Sliding Surface Over Time')
plt.axhline(0, color='k', linewidth=0.5, linestyle='--')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
