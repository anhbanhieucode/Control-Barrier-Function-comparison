import numpy as np
import math
import config as cfg

class DoubleIntegratorDWA:
    def __init__(self, start_pos, goal_pos):
        self.p = np.array(start_pos, dtype=float)
        self.v = np.array([0.0, 0.0], dtype=float)
        self.goal = np.array(goal_pos, dtype=float)
        self.path = [self.p.copy()]
        
        # --- DWA PARAMETERS (HOLONOMIC CONFIG) ---
        self.v_max = cfg.MAX_VEL
        self.a_max = cfg.MAX_ACCEL
        
        # Trọng số (Weights)
        self.w_to_goal = 0.8  # Ưu tiên tiến về đích (Vector Alignment)
        self.w_speed   = 0.5  # Ưu tiên đi nhanh
        self.w_dist    = 3.0  # Ưu tiên tránh vật cản (Safety First)
        
        # Mô phỏng dự đoán (Predictive Horizon)
        self.predict_time = 1.5   # Nhìn trước 1.5 giây
        self.dt_sim       = 0.1   # Bước mô phỏng
        
        # Độ phân giải mẫu (Sampling Resolution)
        self.v_res = 0.2          # 0.2 m/s mỗi bước mẫu

    def calc_dynamic_window(self):
        """
        Tính cửa sổ vận tốc khả thi dựa trên gia tốc tối đa.
        Robot là holonomic nên Vx, Vy độc lập và đối xứng.
        """
        # Giới hạn bởi động cơ (Gia tốc)
        # Robot có thể giảm tốc hoặc tăng tốc trong khoảng [-a*dt, +a*dt]
        min_accel_limit = self.v - self.a_max * cfg.DT
        max_accel_limit = self.v + self.a_max * cfg.DT
        
        # Giới hạn bởi tốc độ tối đa tuyệt đối (Speed Limit)
        # Dynamic Window = Giao của [Accel Limits] và [Global Limits]
        
        dw_vx_min = max(-self.v_max, min_accel_limit[0])
        dw_vx_max = min( self.v_max, max_accel_limit[0])
        
        dw_vy_min = max(-self.v_max, min_accel_limit[1])
        dw_vy_max = min( self.v_max, max_accel_limit[1])
        
        return [dw_vx_min, dw_vx_max, dw_vy_min, dw_vy_max]

    def predict_trajectory(self, v_sample):
        """
        Dự đoán quỹ đạo đường thẳng (Linear Trajectory)
        x_next = x + v * t
        """
        trajectory = []
        p_curr = self.p.copy()
        steps = int(self.predict_time / self.dt_sim)
        
        for _ in range(steps):
            p_curr = p_curr + v_sample * self.dt_sim
            trajectory.append(p_curr)
            
        return np.array(trajectory)

    def calc_cost(self, trajectory, v_sample, obstacles, predictors):
        """
        Tính điểm phạt cho một cặp vận tốc (vx, vy).
        Cost càng THẤP càng tốt.
        """
        # 1. TO GOAL COST: Vector vận tốc nên hướng về đích
        # Tính vector từ vị trí hiện tại đến đích
        to_goal_vec = self.goal - self.p
        dist_total = np.linalg.norm(to_goal_vec)
        
        if dist_total < 0.01:
            cost_goal = 0
        else:
            # Chuẩn hóa
            to_goal_norm = to_goal_vec / dist_total
            v_norm = np.linalg.norm(v_sample)
            
            if v_norm < 0.01:
                # Nếu đứng yên khi chưa đến đích -> Phạt nặng để tránh lười biếng
                cost_goal = 1.0 
            else:
                v_dir = v_sample / v_norm
                # Dot product: 1.0 là thẳng đích, -1.0 là đi ngược
                # Cost = 1 - cos(theta) -> 0 là tốt nhất (thẳng hàng)
                alignment = np.dot(to_goal_norm, v_dir)
                cost_goal = 1.0 - alignment 

        # 2. SPEED COST: Ưu tiên vận tốc lớn (để đến đích nhanh)
        # Cost = (V_max - V_current) / V_max
        speed = np.linalg.norm(v_sample)
        cost_speed = (self.v_max - speed) / self.v_max

        # 3. OBSTACLE COST (Sử dụng LP Prediction)
        min_dist = float('inf')
        
        for k, p_rob in enumerate(trajectory):
            # Thời gian tương lai
            t_future = (k + 1) * self.dt_sim
            
            # Chỉ số tương ứng trong bộ dự đoán LP (dt của LP thường là cfg.DT)
            lp_step = int(t_future / cfg.DT)
            
            for i, obs in enumerate(obstacles):
                # Lấy vị trí vật cản dự đoán
                if predictors is not None:
                    # Lấy chuỗi dự đoán (robust boxes/circles)
                    # Giả sử hàm get_robust_obstacle hỗ trợ trả về list hoặc ta lấy dự đoán thô
                    # Ở đây ta gọi trực tiếp predict_future để lấy box chính xác
                    
                    # Hack: Để nhanh, ta dùng hàm get_robust_obstacle của bạn
                    # nhưng cần đảm bảo nó trả về list nếu ta muốn chính xác từng step.
                    # Nếu không, ta dùng logic đơn giản hóa: Lấy mẫu tại thời điểm tương lai.
                    
                    obs_state = [obs['pos'][0], obs['pos'][1], obs['vel'][0], obs['vel'][1]]
                    # Dự đoán 1 điểm tại đúng thời gian t_future
                    # (Dùng hàm dự đoán 1 bước dài)
                    center_pred, r_pred = predictors[i].get_robust_obstacle(
                        obs_state, obs['radius'], cfg.DT, lookahead_steps=lp_step
                    )
                    obs_pos = center_pred
                    obs_r = r_pred
                else:
                    # Fallback nếu không có predictor
                    obs_pos = obs['pos']
                    obs_r = obs['radius']

                # Tính khoảng cách an toàn
                dist = np.linalg.norm(p_rob - obs_pos) - (cfg.ROBOT_RADIUS + obs_r)
                
                if dist <= 0: 
                    return float('inf') # Va chạm chắc chắn -> Loại bỏ ngay
                
                if dist < min_dist:
                    min_dist = dist

        # Hàm cost nghịch đảo: Gần vật cản -> Cost cực lớn
        # Giới hạn khoảng cách quan tâm (ví dụ 3m) để không bị nhiễu bởi vật ở xa
        if min_dist > 3.0:
            cost_obs = 0
        else:
            cost_obs = 1.0 / (min_dist + 0.1)

        # Tổng hợp Cost
        total_cost = (self.w_to_goal * cost_goal + 
                      self.w_speed   * cost_speed + 
                      self.w_dist    * cost_obs)
        
        return total_cost

    def update(self, dt, obstacles, predictors=None):
        # 1. Tính cửa sổ tìm kiếm (Dynamic Window)
        dw = self.calc_dynamic_window() # [vx_min, vx_max, vy_min, vy_max]
        
        best_u = np.array([0.0, 0.0])
        best_v = self.v.copy()
        min_cost = float('inf')
        
        # 2. Quét lưới 360 độ (Grid Search)
        # Tạo lưới các vận tốc ứng viên
        vxs = np.arange(dw[0], dw[1], self.v_res)
        vys = np.arange(dw[2], dw[3], self.v_res)
        
        # Đảm bảo luôn xét trường hợp dừng khẩn cấp (0,0) nếu nằm trong cửa sổ
        if dw[0] <= 0 <= dw[1]: vxs = np.append(vxs, 0)
        if dw[2] <= 0 <= dw[3]: vys = np.append(vys, 0)

        # 3. Đánh giá từng mẫu
        for vx in vxs:
            for vy in vys:
                v_sample = np.array([vx, vy])
                
                # Bỏ qua nếu vận tốc quá nhỏ (trừ khi đang đứng yên)
                if np.linalg.norm(v_sample) < 0.05 and np.linalg.norm(self.v) > 0.1:
                   # Đang chạy mà phanh gấp về 0 cũng là 1 option, nên vẫn xét
                   pass

                # Dự đoán và chấm điểm
                traj = self.predict_trajectory(v_sample)
                cost = self.calc_cost(traj, v_sample, obstacles, predictors)
                
                if cost < min_cost:
                    min_cost = cost
                    best_v = v_sample

        # 4. Tính gia tốc điều khiển (Control Input u)
        # u = (v_best - v_curr) / dt
        desired_accel = (best_v - self.v) / dt
        
        # Kẹp gia tốc trong giới hạn vật lý (để an toàn tuyệt đối)
        accel_norm = np.linalg.norm(desired_accel)
        if accel_norm > self.a_max:
            desired_accel = (desired_accel / accel_norm) * self.a_max
            
        # 5. Cập nhật trạng thái
        self.v += desired_accel * dt
        self.p += self.v * dt
        self.path.append(self.p.copy())
        
        return desired_accel