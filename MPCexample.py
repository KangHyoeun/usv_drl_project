import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# 시스템 파라미터 정의
A = np.array([[1, 1], [0, 1]])
B = np.array([[0.5], [1.0]])
x0 = np.array([[0], [0]])  # 초기 상태

# MPC 파라미터
horizon = 10  # 예측 지평선
Q = np.eye(2)  # 상태 비용 행렬
R = np.array([[1]])  # 입력 비용 행렬

# 최적화 변수 정의
x = cp.Variable((2, horizon + 1))
u = cp.Variable((1, horizon))

# 비용 함수와 제약 조건 초기화
cost = 0
constraints = [x[:, 0] == x0[:, 0]]

# MPC 최적화 문제 설정
for t in range(horizon):
    cost += cp.quad_form(x[:, t], Q) + cp.quad_form(u[:, t], R)
    constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t]]

# 최종 상태에 대한 비용 추가
cost += cp.quad_form(x[:, horizon], Q)

# 최적화 문제 풀기
problem = cp.Problem(cp.Minimize(cost), constraints)
problem.solve()

# 결과 시각화
plt.figure(figsize=(12, 5))

# 제어 입력 시각화
plt.subplot(1, 2, 1)
plt.step(range(horizon), u.value[0, :], where='post')
plt.xlabel('Time')
plt.ylabel('Control Input u')
plt.title('Control Input Over Time')
plt.grid(True)

# 상태 시각화
plt.subplot(1, 2, 2)
plt.plot(range(horizon + 1), x.value[0, :], label='State x1')
plt.plot(range(horizon + 1), x.value[1, :], label='State x2')
plt.xlabel('Time')
plt.ylabel('State')
plt.title('State Trajectories Over Time')
plt.legend()
plt.grid(True)

plt.show()