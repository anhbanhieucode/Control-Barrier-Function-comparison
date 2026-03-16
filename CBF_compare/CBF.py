import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cvxpy as cp

class CBFRobot:
    def __init__(self, start_pos, goal_pos, r_robot, v_max):
        self.p = np.array(start_pos, dtype=float)
        self.goal = np.array(goal_pos, dtype=float)
        self.r_robot = r_robot
        self.v_max = v_max
        self.path = [self.p.copy()]

    def nominal_controller(self):
        """
        Bộ điều khiển dẫn đường cơ bản (Proportional Controller).
        Luôn muốn đi thẳng về đích.
        """
        kp = 2.0
        error = self.goal - self.p
        u_nom = kp * error
        
        # Clip vận tốc nếu vượt quá v_max
        norm = np.linalg.norm(u_nom)
        if norm > self.v_max:
            u_nom = (u_nom / norm) * self.v_max
        return u_nom

    def solve_cbf_qp(self, u_nom, obstacles, gamma=1.0):
        """
        Giải bài toán Quadratic Programming (QP):
        Minimize ||u - u_nom||^2
        Subject to: A*u <= b (Ràng buộc an toàn CBF)
        """
        u = cp.Variable(2)
        constraints = []
        
        # Với mỗi vật cản, thêm một ràng buộc (linear constraint)
        for obs in obstacles:
            p_rel = self.p - obs['pos']     # Vector vị trí tương đối
            dist_sq = np.sum(p_rel**2)      # Khoảng cách bình phương
            min_dist = self.r_robot + obs['radius'] # Khoảng cách an toàn tối thiểu
            
            # Hàm h(x)
            h = dist_sq - min_dist**2
            
            # Đạo hàm từng phần theo p (Gradient h)
            # grad_h = 2 * (p - p_obs)
            grad_h = 2 * p_rel
            
            # Tính term bù trừ chuyển động vật cản (nếu vật cản di chuyển)
            # term_obs = grad_h * v_obs
            term_obs = grad_h @ obs['vel']
            
            # Ràng buộc: grad_h * u >= -gamma * h + term_obs
            # Đổi dấu để đưa về dạng chuẩn A*u <= b của CVXPY:
            # -grad_h * u <= gamma * h - term_obs
            constraints.append(-grad_h @ u <= gamma * h - term_obs)

        # Ràng buộc vận tốc tối đa (Box constraint)
        constraints += [cp.norm(u) <= self.v_max]

        # Hàm mục tiêu: Giống u_nom nhất có thể
        objective = cp.Minimize(cp.sum_squares(u - u_nom))
        
        # Giải
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve()
            if u.value is None:
                # Nếu không tìm được nghiệm (infeasible), phanh gấp
                return np.array([0.0, 0.0])
            return u.value
        except:
            return np.array([0.0, 0.0])

    def update(self, dt, obstacles):
        # 1. Tính u_nominal (muốn đi đâu)
        u_nom = self.nominal_controller()
        
        # 2. Lọc qua CBF (có an toàn không?)
        u_safe = self.solve_cbf_qp(u_nom, obstacles)
        
        # 3. Cập nhật trạng thái
        self.p += u_safe * dt
        self.path.append(self.p.copy())
        
        return u_safe

# --- Thiết lập Môi trường ---
def run_simulation():
    dt = 0.05
    steps = 200
    
    # Khởi tạo Robot
    robot = CBFRobot(start_pos=[0, 0], goal_pos=[10, 10], r_robot=0.5, v_max=2.0)
    
    # Khởi tạo Vật cản (Vị trí, Bán kính, Vận tốc)
    # Vật cản 1 đứng yên, Vật cản 2 di chuyển cắt ngang
    obstacles = [
        {'pos': np.array([5.0, 5.0]), 'radius': 1.0, 'vel': np.array([0.0, 0.0])}, 
        {'pos': np.array([2.0, 8.0]), 'radius': 0.8, 'vel': np.array([1.5, -0.5])},
        {'pos': np.array([7.0, 2.0]), 'radius': 1.2, 'vel': np.array([-0.5, 0.5])} 
    ]

    # Lưu trữ dữ liệu để vẽ
    history_robot = []
    history_obs = [[] for _ in obstacles]

    print("Đang chạy mô phỏng...")
    for _ in range(steps):
        # Cập nhật vật cản di chuyển
        for i, obs in enumerate(obstacles):
            obs['pos'] += obs['vel'] * dt
            history_obs[i].append(obs['pos'].copy())
            
            # Simple bounce logic (để vật cản không chạy mất khỏi màn hình)
            if obs['pos'][0] < -2 or obs['pos'][0] > 12: obs['vel'][0] *= -1
            if obs['pos'][1] < -2 or obs['pos'][1] > 12: obs['vel'][1] *= -1

        # Cập nhật Robot
        robot.update(dt, obstacles)
        history_robot.append(robot.p.copy())
        
        # Kiểm tra đến đích
        if np.linalg.norm(robot.p - robot.goal) < 0.1:
            print("Đã đến đích!")
            break

    # --- Vẽ đồ thị (Animation) ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-2, 12)
    ax.set_ylim(-2, 12)
    ax.set_aspect('equal')
    ax.grid(True)

    # Các đối tượng đồ họa
    robot_patch = plt.Circle(robot.p, robot.r_robot, color='blue', alpha=0.8, label='Robot')
    goal_patch = plt.Circle(robot.goal, 0.3, color='green', alpha=0.5, label='Goal')
    obs_patches = [plt.Circle(o['pos'], o['radius'], color='red', alpha=0.6) for o in obstacles]
    path_line, = ax.plot([], [], 'b--', linewidth=1)

    ax.add_patch(robot_patch)
    ax.add_patch(goal_patch)
    for p in obs_patches: ax.add_patch(p)
    ax.legend()

    def animate(i):
        if i >= len(history_robot): return
        
        # Update Robot
        current_pos = history_robot[i]
        robot_patch.center = current_pos
        
        # Update Trail
        path_x = [p[0] for p in history_robot[:i+1]]
        path_y = [p[1] for p in history_robot[:i+1]]
        path_line.set_data(path_x, path_y)
        
        # Update Obstacles
        for j, patch in enumerate(obs_patches):
            patch.center = history_obs[j][i]
            
        return [robot_patch, path_line] + obs_patches

    ani = animation.FuncAnimation(fig, animate, frames=len(history_robot), interval=30, blit=True)
    plt.show()

if __name__ == "__main__":
    run_simulation()