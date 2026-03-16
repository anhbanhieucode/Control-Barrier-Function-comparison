import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cvxpy as cp

class DoubleIntegratorCBF:
    def __init__(self, start_pos, goal_pos, r_robot, a_max, v_max):
        self.p = np.array(start_pos, dtype=float)
        self.v = np.array([0.0, 0.0], dtype=float) # Vận tốc ban đầu
        self.goal = np.array(goal_pos, dtype=float)
        self.r_robot = r_robot
        
        self.a_max = a_max # Gia tốc tối đa
        self.v_max = v_max # Vận tốc tối đa (để kẹp lại trong nominal controller)
        self.path = [self.p.copy()]

    def nominal_controller(self):
        """
        Bộ điều khiển PD (Proportional-Derivative) để tracking vị trí.
        Output là gia tốc mong muốn (u_nom).
        """
        kp = 20000.0  # Gain vị trí
        kd = 1.5  # Gain vận tốc (Damping)
        
        error_pos = self.goal - self.p
        error_vel = np.array([0.0, 0.0]) - self.v # Muốn v = 0 tại đích
        
        u_nom = kp * error_pos + kd * error_vel
        
        # Clip gia tốc theo a_max (đơn giản hoá, chuẩn ra phải dùng box constraint trong QP)
        norm = np.linalg.norm(u_nom)
        if norm > self.a_max:
            u_nom = (u_nom / norm) * self.a_max
        return u_nom

    def solve_cbf_qp(self, u_nom, obstacles):
        """
        Giải QP cho hệ bậc 2 (Double Integrator).
        Biến tối ưu: u (gia tốc)
        """
        u = cp.Variable(2)
        constraints = []
        
        # Hệ số CBF bậc 2 (Tương tự như PD cho hàng rào an toàn)
        # Cần tune khéo: Kp lớn giúp robot dũng cảm, Kv lớn giúp robot phanh sớm
        K_p = 5.0 
        K_v = 10.0

        for obs in obstacles:
            # 1. Tính toán các đại lượng trạng thái tương đối
            p_rel = self.p - obs['pos']
            v_rel = self.v - obs['vel']
            
            # Giả sử vật cản chuyển động với gia tốc = 0 (Constant Velocity Model)
            # Nếu vật cản có gia tốc, cần cộng vào đây.
            a_obs = np.array([0.0, 0.0]) 

            dist_sq = np.sum(p_rel**2)
            min_dist = self.r_robot + obs['radius']
            
            # 2. Tính h(x) và h_dot(x)
            h = dist_sq - min_dist**2
            h_dot = 2 * np.dot(p_rel, v_rel)
            
            # 3. Xây dựng ràng buộc High Order CBF
            # Bất đẳng thức: 2*p_rel*u >= -2*v_rel^2 + 2*p_rel*a_obs - Kv*h_dot - Kp*h
            
            Lgh = 2 * p_rel  # Hệ số nhân với u
            
            # Phần bên phải bất đẳng thức (Lfh + ...)
            term_dyn = -2 * np.dot(v_rel, v_rel) + 2 * np.dot(p_rel, a_obs)
            term_barrier = -K_v * h_dot - K_p * h
            
            rhs = term_dyn + term_barrier
            
            # Đưa vào CVXPY: -Lgh * u <= -rhs (Đổi dấu >= thành <=)
            constraints.append(-Lgh @ u <= -rhs)

        # Ràng buộc vật lý: Gia tốc không quá a_max
        constraints += [cp.norm(u) <= self.a_max]
        
        # (Optional) Ràng buộc vận tốc tối đa: v_next = v + u*dt <= v_max
        # Cái này xấp xỉ tuyến tính để giữ bài toán là QP
        # constraints += [cp.norm(self.v + u * 0.05) <= self.v_max] 

        objective = cp.Minimize(cp.sum_squares(u - u_nom))
        
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve()
            if u.value is None:
                # Infeasible: Thường xảy ra khi robot đi quá nhanh vào vùng chết
                # Fallback: Phanh tối đa ngược hướng vận tốc
                # print("Infeasible! Braking hard.")
                return -3.0 * self.v 
            return u.value
        except:
            return np.array([0.0, 0.0])

    def update(self, dt, obstacles):
        # 1. Tính gia tốc mong muốn (PD Controller)
        u_nom = self.nominal_controller()
        
        # 2. Lọc qua CBF bậc 2
        u_safe = self.solve_cbf_qp(u_nom, obstacles)
        
        # 3. Tích phân Euler (Cập nhật trạng thái)
        self.v += u_safe * dt
        self.p += self.v * dt
        
        self.path.append(self.p.copy())
        return u_safe

# --- Simulation Setup ---
def run_simulation():
    dt = 0.05
    steps = 250 # Tăng steps vì hệ có quán tính, đi sẽ mượt nhưng chậm hơn chút
    
    # Robot có quán tính: a_max=5.0, v_max=3.0
    robot = DoubleIntegratorCBF(start_pos=[0, 0], goal_pos=[10, 10], 
                                r_robot=0.5, a_max=5.0, v_max=3.0)
    
    obstacles = [
        {'pos': np.array([5.0, 5.0]), 'radius': 1.0, 'vel': np.array([0.0, 0.0])}, 
        {'pos': np.array([2.5, 8.0]), 'radius': 0.8, 'vel': np.array([1.2, -0.6])},
        {'pos': np.array([7.0, 3.0]), 'radius': 1.2, 'vel': np.array([-0.8, 0.4])},
        {'pos': np.array([6.0, 6.5]), 'radius': 0.9, 'vel': np.array([0.0, 0.06])} 
    ]

    history_robot = []
    history_obs = [[] for _ in obstacles]

    print("Simulating Double Integrator Robot...")
    for _ in range(steps):
        # Update obstacles
        for i, obs in enumerate(obstacles):
            obs['pos'] += obs['vel'] * dt
            history_obs[i].append(obs['pos'].copy())
            if obs['pos'][0] < -2 or obs['pos'][0] > 12: obs['vel'][0] *= -1
            if obs['pos'][1] < -2 or obs['pos'][1] > 12: obs['vel'][1] *= -1

        # Update Robot
        robot.update(dt, obstacles)
        history_robot.append(robot.p.copy())
        
        if np.linalg.norm(robot.p - robot.goal) < 0.1 and np.linalg.norm(robot.v) < 0.2:
            print("Đã đến đích và dừng lại!")
            break

    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-2, 12)
    ax.set_ylim(-2, 12)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title("Double Integrator CBF (Input = Acceleration)")

    robot_patch = plt.Circle(robot.p, robot.r_robot, color='blue', alpha=0.8, label='Robot')
    goal_patch = plt.Circle(robot.goal, 0.3, color='green', alpha=0.5, label='Goal')
    obs_patches = [plt.Circle(o['pos'], o['radius'], color='red', alpha=0.6) for o in obstacles]
    path_line, = ax.plot([], [], 'b-', linewidth=1, alpha=0.5)

    ax.add_patch(robot_patch)
    ax.add_patch(goal_patch)
    for p in obs_patches: ax.add_patch(p)
    ax.legend()

    def animate(i):
        if i >= len(history_robot): return
        
        robot_patch.center = history_robot[i]
        
        # Vẽ đuôi dài hơn chút để thấy độ mượt
        path_x = [p[0] for p in history_robot[:i+1]]
        path_y = [p[1] for p in history_robot[:i+1]]
        path_line.set_data(path_x, path_y)
        
        for j, patch in enumerate(obs_patches):
            patch.center = history_obs[j][i]
            
        return [robot_patch, path_line] + obs_patches

    ani = animation.FuncAnimation(fig, animate, frames=len(history_robot), interval=30, blit=True)
    plt.show()

if __name__ == "__main__":
    run_simulation()