import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 참조 모델의 매개변수
Am = -1.0
Bm = 1.0

# 실제 시스템의 매개변수 (이 부분이 누락되었습니다)
Ap = -0.5
Bp = 1.0
Cp = 1.0

# 적응 제어기의 매개변수
Gamma_theta = 1.0  # 적응 법칙의 게인 매개변수
Gamma_k = 0.5      # 적응 법칙의 게인 매개변수

# 초기 조건
initial_state = [0.0, 0.5, 0.5]  # 초기 상태 [y, theta_hat, k_hat]

# 시뮬레이션 시간 설정
t = np.linspace(0, 10, 201)  # 시간 배열

# 참조 모델 정의
def reference_model(t):
    ym = np.ones_like(t)  # 단위 계단 함수
    return ym

# 실제 시스템과 적응 제어기를 포함한 시스템의 동적 방정식 정의
def adaptive_system(state, t):
    y, theta_hat, k_hat = state
    ym = reference_model(t)
    e = ym - y  # 추종 오차
    
    # 적응 제어기 출력
    u = theta_hat * ym - k_hat * e
    
    # 실제 시스템의 다음 상태 계산
    dydt = Ap * y + Bp * u
    # 적응 법칙에 따른 매개변수의 추정치 갱신
    dtheta_hat_dt = Gamma_theta * e * ym
    dk_hat_dt = -Gamma_k * e * y
    
    return [dydt, dtheta_hat_dt, dk_hat_dt]

# 시스템의 초기 상태에서 시뮬레이션 실행
states = odeint(adaptive_system, initial_state, t)

# 결과 시각화
plt.figure(figsize=(12, 6))

# 시스템 출력과 참조 모델 출력
plt.subplot(211)
plt.plot(t, states[:, 0], 'b-', label='System Output (y)')
plt.plot(t, reference_model(t), 'g--', label='Reference Model (ym)')
plt.xlabel('Time')
plt.ylabel('Output')
plt.title('System Output vs. Reference Model Output')
plt.legend()

# 적응 제어기 매개변수 추정치
plt.subplot(212)
plt.plot(t, states[:, 1], 'r-', label='Estimated theta_hat')
plt.plot(t, states[:, 2], 'k--', label='Estimated k_hat')
plt.xlabel('Time')
plt.ylabel('Parameter Value')
plt.title('Adaptive Controller Parameter Estimates')
plt.legend()

plt.tight_layout()
plt.show()
