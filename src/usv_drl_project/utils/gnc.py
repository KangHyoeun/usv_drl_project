import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import PchipInterpolator, PPoly, interp1d

def ssa(angle, unit=None):
    """Normalize angle to the range [-180, 180] degrees or [-π, π] radians."""
    if unit == 'deg':
        return (angle + 180) % 360 - 180
    else:
        return (angle + np.pi) % (2 * np.pi) - np.pi


def sat(x, x_max):
    """
    Saturates the input x at the specified maximum absolute value x_max.
    """
    if np.all(np.array(x_max) <= 0):
        raise ValueError("x_max must be a vector of positive elements.")

    return np.minimum(np.maximum(x, -np.array(x_max)), np.array(x_max))


def ref_model(x_d, v_d, a_d, x_ref, v_max, zeta_d, w_d, h, eulerAngle):
    """
    Position, velocity, and acceleration reference model.
    """
    if eulerAngle == 1:
        e_x = ssa(x_d - x_ref)  # Smallest signed angle
    else:
        e_x = x_d - x_ref

    a_d_dot = -w_d**3 * e_x - (2 * zeta_d + 1) * w_d**2 * v_d - (2 * zeta_d + 1) * w_d * a_d

    x_d += h * v_d
    v_d += h * a_d
    a_d += h * a_d_dot

    if abs(v_d) > v_max:
        v_d = np.sign(v_d) * v_max

    return x_d, v_d, a_d


def LOS_observer(LOSangle, LOSrate, LOScommand, h, K_f):
    T_f = 1 / (K_f + 2 * np.sqrt(K_f) + 1)
    xi = LOSangle - LOSrate

    LOSangle += h * (LOSrate + K_f * ssa(LOScommand - LOSangle))

    PHI = np.exp(-h / T_f)
    xi = (PHI * xi) + (1 - PHI) * LOSangle

    LOSrate = LOSangle - xi

    return LOSangle, LOSrate

def add_intermediate_waypoints(wpt, multiplier):

    # Ensure wpt['pos']['x'] and wpt['pos']['y'] are column vectors
    x = list(wpt['pos']['x'])
    y = list(wpt['pos']['y'])

    dense_x = []
    dense_y = []

    for i in range(len(x) - 1):
        # 현재 시작 지점 추가
        dense_x.append(x[i])
        dense_y.append(y[i])

        # 선형 보간된 중간 지점 추가
        for j in range(1, multiplier):
            t = j / multiplier
            new_x = x[i] * (1 - t) + x[i + 1] * t
            new_y = y[i] * (1 - t) + y[i + 1] * t
            dense_x.append(new_x)
            dense_y.append(new_y)

    # 마지막 waypoint 추가
    dense_x.append(x[-1])
    dense_y.append(y[-1])

    return {
        'pos': {
            'x': dense_x,
            'y': dense_y
        }
    }

def crosstrack_hermite_LOS(w_path, x_path, y_path, dx_path, dy_path, pi_h, x, y, h, Delta_h, pp_x, pp_y, idx_start, N_horizon, gamma_h=None):
    
    if not hasattr(crosstrack_hermite_LOS, 'persistent'):
        crosstrack_hermite_LOS.persistent = {}

    persistent = crosstrack_hermite_LOS.persistent

    if 'beta_hat' not in persistent:
        persistent['beta_hat'] = 0  # Initial parameter estimate

    beta_hat = persistent['beta_hat']

    idx_end = min(idx_start + N_horizon, len(w_path))
    w_horizon = w_path[idx_start:idx_end]

    x_horizon = pp_x(w_horizon)
    y_horizon = pp_y(w_horizon)

    distances = np.sqrt((x_horizon - x)**2 + (y_horizon - y)**2)
    min_distance_idx = np.argmin(distances)
    min_distance = distances[min_distance_idx]

    idx_start += min_distance_idx 

    vector_to_point = np.array([x - x_path[idx_start], y - y_path[idx_start]])
    cross_prod = dx_path[idx_start] * vector_to_point[1] - dy_path[idx_start] * vector_to_point[0]
    y_e = np.sign(cross_prod) * min_distance

    if gamma_h == None:  # LOS guidance law for course control
        LOS_angle = pi_h[idx_start] - np.arctan2(y_e, Delta_h)

    else:  # ALOS guidance law for heading control
        LOS_angle = pi_h[idx_start] - beta_hat - np.arctan2(y_e, Delta_h)
        beta_hat += h * gamma_h * Delta_h * y_e / np.sqrt(Delta_h**2 + y_e**2)
        persistent['beta_hat'] = beta_hat

    return LOS_angle, idx_start, y_e 

## Calculate the derivative of a piecewise polynomial with smoothing
def pp_derivative(pp: PPoly) -> PPoly:
    # Extract the pieces of the piecewise polynomial
    breaks = pp.x
    coefs = pp.c

    deg_plus_1, n_intervals = coefs.shape
    k = deg_plus_1

    powers = np.arange(k - 1, -1, -1)
    dcoefs = coefs[:-1,:] * powers[:k - 1,np.newaxis]

    dpp = PPoly(dcoefs, breaks)

    num_points = 1000
    xq = np.linspace(breaks[0], breaks[-1], num_points)
    yq = pp(xq)
    dyq = dpp(xq)
    
    window_size = 100
    smoothed_dyq = uniform_filter1d(dyq, size=window_size)
    
    dpp = interp1d(xq, smoothed_dyq, kind='linear', fill_value="extrapolate")
    return dpp

def hermite_spline(wpt, Umax, h):

    x = np.array(wpt['pos']['x'])
    y = np.array(wpt['pos']['y'])

    path_length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))

    time = path_length / Umax

    N_interval = int(np.floor(time / h)) + 1 
    delta_path = path_length / N_interval
    N_horizon = int(np.round(Umax / delta_path))

    w_path = np.linspace(0, N_interval, N_interval + 1)
    wpt_idx = np.linspace(0, N_interval, len(x))

    pp_x = PchipInterpolator(wpt_idx, x)
    pp_y = PchipInterpolator(wpt_idx, y)

    pp_dx = pp_derivative(pp_x)
    pp_dy = pp_derivative(pp_y)

    dx_path = pp_dx(w_path)
    dy_path = pp_dy(w_path)
    pi_h = np.arctan2(dy_path, dx_path)

    x_path = pp_x(w_path)
    y_path = pp_y(w_path)

    return w_path, x_path, y_path, dx_path, dy_path, pi_h, pp_x, pp_y, N_horizon


def LOS_chi(x, y, Delta_h, R_switch, wpt):
    """
    Compute the desired course angle (chi_ref) and cross-track error (y_e).
    """
    if not hasattr(LOS_chi, 'persistent'):
        LOS_chi.persistent = {}

    persistent = LOS_chi.persistent

    if 'k' not in persistent:
        dist_between_wpts = np.sqrt(
            np.diff(wpt['pos']['x'])**2 + np.diff(wpt['pos']['y'])**2)
        if R_switch > min(dist_between_wpts):
            raise ValueError(
                "The distances between the waypoints must be larger than R_switch"
            )

        if R_switch < 0:
            raise ValueError("R_switch must be larger than zero")

        if Delta_h < 0:
            raise ValueError("Delta_h must be larger than zero")

        persistent['k'] = 0
        persistent['xk'] = wpt['pos']['x'][0]
        persistent['yk'] = wpt['pos']['y'][0]

    k = persistent['k']
    xk = persistent['xk']
    yk = persistent['yk']

    n = len(wpt['pos']['x'])

    if k < n - 1:
        xk_next = wpt['pos']['x'][k + 1]
        yk_next = wpt['pos']['y'][k + 1]
    else:
        bearing = np.arctan2(wpt['pos']['y'][-1] - wpt['pos']['y'][-2],
                             wpt['pos']['x'][-1] - wpt['pos']['x'][-2])
        R = 1e10
        xk_next = wpt['pos']['x'][-1] + R * np.cos(bearing)
        yk_next = wpt['pos']['y'][-1] + R * np.sin(bearing)

    pi_h = np.arctan2(yk_next - yk, xk_next - xk)

    x_e = (x - xk) * np.cos(pi_h) + (y - yk) * np.sin(pi_h)
    y_e = -(x - xk) * np.sin(pi_h) + (y - yk) * np.cos(pi_h)

    d = np.sqrt((xk_next - xk)**2 + (yk_next - yk)**2)

    if d - x_e < R_switch and k < n - 1:
        persistent['k'] += 1
        persistent['xk'] = xk_next
        persistent['yk'] = yk_next

    chi_ref = pi_h - np.arctan(y_e / Delta_h)

    return chi_ref, y_e


def EKF_5states(position1,
                position2,
                h,
                Z,
                frame,
                Qd,
                Rd,
                alpha_1=0.01,
                alpha_2=0.1,
                x_prd_init=None):
    if not hasattr(EKF_5states, 'persistent'):
        EKF_5states.persistent = {}

    persistent = EKF_5states.persistent

    # WGS-84 데이터
    a = 6378137  # 반장축 (적도 반지름)
    f = 1 / 298.257223563  # 평평도
    e = np.sqrt(2 * f - f**2)  # 지구 이심률

    # 초기 상태 설정
    I5 = np.eye(5)

    if 'x_prd' not in persistent:
        if x_prd_init is None:
            print(
                f"Using default initial EKF states: x_prd = [{position1}, {position2}, 0, 0, 0]"
            )
            persistent['x_prd'] = np.array([position1, position2, 0, 0, 0])
        else:
            print(
                f"Using user specified initial EKF states: x_prd = {x_prd_init}"
            )
            persistent['x_prd'] = np.array(x_prd_init)
        persistent['P_prd'] = I5
        persistent['count'] = 1

    x_prd = persistent['x_prd']
    P_prd = persistent['P_prd']
    count = persistent['count']

    Cd = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
    Ed = h * np.array([[0, 0], [0, 0], [1, 0], [0, 0], [0, 1]])

    if count == 1:
        y = np.array([position1, position2])
        K = P_prd @ Cd.T @ np.linalg.inv(Cd @ P_prd @ Cd.T + Rd)
        IKC = I5 - K @ Cd
        P_hat = IKC @ P_prd @ IKC.T + K @ Rd @ K.T
        eps = y - Cd @ x_prd

        if frame == 'LL':
            eps = eps # (2 * np.pi) - np.pi  # Smallest signed angle

        x_hat = x_prd + K @ eps
        count = Z
    else:
        x_hat = x_prd
        P_hat = P_prd
        count -= 1

    if frame == 'NED':
        f = np.array([
            x_hat[2] * np.cos(x_hat[3]), x_hat[2] * np.sin(x_hat[3]),
            -alpha_1 * x_hat[2], x_hat[4], -alpha_2 * x_hat[4]
        ])

        Ad = I5 + h * np.array(
            [[0, 0, np.cos(x_hat[3]), -x_hat[2] * np.sin(x_hat[3]), 0],
             [0, 0, np.sin(x_hat[3]), x_hat[2] * np.cos(x_hat[3]), 0],
             [0, 0, -alpha_1, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, -alpha_2]])

    elif frame == 'LL':
        Rn = a / np.sqrt(1 - e**2 * np.sin(x_hat[0])**2)
        Rm = Rn * ((1 - e**2) / (1 - e**2 * np.sin(x_hat[0])**2))

        f = np.array([
            (1 / Rm) * x_hat[2] * np.cos(x_hat[3]),
            (1 / (Rn * np.cos(x_hat[0]))) * x_hat[2] * np.sin(x_hat[3]),
            -alpha_1 * x_hat[2], x_hat[4], -alpha_2 * x_hat[4]
        ])

        Ad = I5 + h * np.array([
            [
                0, 0, (1 / Rm) * np.cos(x_hat[3]),
                -(1 / Rm) * x_hat[2] * np.sin(x_hat[3]), 0
            ],
            [
                np.tan(x_hat[0]) /
                (Rn * np.cos(x_hat[0])) * x_hat[2] * np.sin(x_hat[3]), 0,
                (1 / (Rn * np.cos(x_hat[0]))) * np.sin(x_hat[3]),
                (1 / (Rn * np.cos(x_hat[0]))) * x_hat[2] * np.cos(x_hat[3]), 0
            ], [0, 0, -alpha_1, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, -alpha_2]
        ])

    x_prd = x_hat + h * f
    P_prd = Ad @ P_hat @ Ad.T + Ed @ Qd @ Ed.T

    persistent['x_prd'] = x_prd
    persistent['P_prd'] = P_prd
    persistent['count'] = count

    return x_hat

def ALOS_psi(x,y,Delta_h,gamma_h,h,R_switch,wpt):

    if not hasattr(ALOS_psi, 'persistent'):
        ALOS_psi.persistent = {}

    persistent = ALOS_psi.persistent

    if 'k' not in persistent:
        dist_between_wpts = np.sqrt(np.diff(wpt['pos']['x'])**2 + np.diff(wpt['pos']['y'])**2)

        if R_switch > min(dist_between_wpts):
            raise ValueError("The distances between the waypoints must be larger than R_switch")

        # Check input parameters
        if R_switch < 0: raise ValueError("R_switch must be larger than zero") 
        if Delta_h < 0: raise ValueError("Delta must be larger than zero") 

        persistent['beta_hat'] = 0
        persistent['k'] = 0
        persistent['xk'] = wpt['pos']['x'][0]
        persistent['yk'] = wpt['pos']['y'][0]

    k = persistent['k']
    xk = persistent['xk']
    yk = persistent['yk']
    beta_hat = persistent['beta_hat']

    n = len(wpt['pos']['x'])

    if k < n - 1:
        xk_next = wpt['pos']['x'][k + 1]  
        yk_next = wpt['pos']['y'][k + 1]    
    else:
        bearing = np.arctan2(wpt['pos']['y'][-1] - wpt['pos']['y'][-2],
                             wpt['pos']['x'][-1] - wpt['pos']['x'][-2])
        R = 1e10
        xk_next = wpt['pos']['x'][n] + R * np.cos(bearing)
        yk_next = wpt['pos']['y'][n] + R * np.sin(bearing) 

    ## Compute the path-tangnetial angle w.r.t. North
    pi_h = np.arctan2(yk_next - yk, xk_next - xk) 

    # Along-track and cross-track errors (x_e, y_e) 
    x_e =  (x - xk) * np.cos(pi_h) + (y - yk) * np.sin(pi_h)
    y_e = -(x - xk) * np.sin(pi_h) + (y - yk) * np.cos(pi_h)

    # If the next waypoint satisfy the switching criterion, k = k + 1
    d = np.sqrt((xk_next-xk)**2 + (yk_next-yk)**2)
    if (d - x_e < R_switch and k < n - 1):
        persistent['k'] += 1
        persistent['xk'] = xk_next
        persistent['yk'] = yk_next

    # ALOS guidance law
    psi_ref = pi_h - beta_hat - np.arctan(y_e/Delta_h) 

    # Propagation of states to time k+1
    beta_hat += h * gamma_h * Delta_h * y_e / np.sqrt(Delta_h**2 + y_e**2)

    persistent['beta_hat'] = beta_hat

    return psi_ref, y_e

def ILOS_psi(x,y,Delta_h,kappa,h,R_switch,wpt):

    if not hasattr(ILOS_psi, 'persistent'):
        ILOS_psi.persistent = {}

    persistent = ILOS_psi.persistent

    if 'k' not in persistent:
        dist_between_wpts = np.sqrt(np.diff(wpt['pos']['x'])**2 + np.diff(wpt['pos']['y'])**2)

        if R_switch > min(dist_between_wpts):
            raise ValueError("The distances between the waypoints must be larger than R_switch")

        # Check input parameters
        if R_switch < 0: raise ValueError("R_switch must be larger than zero") 
        if Delta_h < 0: raise ValueError("Delta must be larger than zero") 

        persistent['y_int'] = 0              # initial states
        persistent['k'] = 0
        persistent['xk'] = wpt['pos']['x'][0]
        persistent['yk'] = wpt['pos']['y'][0]

    k = persistent['k']
    xk = persistent['xk']
    yk = persistent['yk']
    y_int = persistent['y_int']

    n = len(wpt['pos']['x'])

    if k < n - 1:
        xk_next = wpt['pos']['x'][k + 1]  
        yk_next = wpt['pos']['y'][k + 1]    
    else:
        bearing = np.arctan2(wpt['pos']['y'][-1] - wpt['pos']['y'][-2],
                             wpt['pos']['x'][-1] - wpt['pos']['x'][-2])
        R = 1e10
        xk_next = wpt['pos']['x'][n] + R * np.cos(bearing)
        yk_next = wpt['pos']['y'][n] + R * np.sin(bearing) 

    ## Compute the path-tangnetial angle w.r.t. North
    pi_h = np.arctan2(yk_next - yk, xk_next - xk) 

    # Along-track and cross-track errors (x_e, y_e) 
    x_e =  (x - xk) * np.cos(pi_h) + (y - yk) * np.sin(pi_h)
    y_e = -(x - xk) * np.sin(pi_h) + (y - yk) * np.cos(pi_h)

    # If the next waypoint satisfy the switching criterion, k = k + 1
    d = np.sqrt((xk_next-xk)**2 + (yk_next-yk)**2)
    if (d - x_e < R_switch and k < n - 1):
        persistent['k'] += 1
        persistent['xk'] = xk_next
        persistent['yk'] = yk_next

    # ILOS guidance law
    Kp = 1 / Delta_h
    psi_ref = pi_h - np.arctan(Kp * (y_e + kappa * y_int))     

    # Propagation of states to time k+1
    y_int +=  h * Delta_h * y_e / (Delta_h**2 + (y_e + kappa * y_int)**2)

    persistent['y_int'] = y_int

    return psi_ref, y_e
