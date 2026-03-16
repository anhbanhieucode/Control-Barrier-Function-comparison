import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import config as cfg
from cbf_robot import DoubleIntegratorCBF
from obstacle_predictor import LPLearner

# --- CLASS GIẢ LẬP KHÔNG CÓ DỰ ĐOÁN ---
class DummyPredictor:
    """
    Giả lập một Predictor 'mù'. 
    Nó luôn trả về vị trí hiện tại và bán kính gốc, 
    coi như không biết vật cản sẽ di chuyển đi đâu (Standard CBF).
    """
    def update(self, ax, ay): pass # Không cần học gì cả
    
    def predict_future(self, current_state, dt, steps):
        return [] # Không vẽ hộp dự đoán
        
    def get_robust_obstacle(self, obs_state, obs_radius, dt, lookahead_steps=5):
        # Chỉ trả về đúng vị trí hiện tại (x, y) và bán kính thật
        return np.array([obs_state[0], obs_state[1]]), obs_radius

def run_comparison():
    # --- SETUP CHUNG ---
    # Hai robot giống hệt nhau về động lực học
    robot_std = DoubleIntegratorCBF(start_pos=[0, 0], goal_pos=[8, 8])   # Robot không dùng dự đoán
    robot_robust = DoubleIntegratorCBF(start_pos=[0, 0], goal_pos=[8, 8]) # Robot dùng LP dự đoán
    
    # 5 Obstacles (HARD MODE)
    obstacles = [
        {'pos': np.array([5.0, 5.0]), 'radius': 0.8, 'vel': np.array([2.2, 1.5])}, 
        {'pos': np.array([3.0, 9.0]), 'radius': 0.7, 'vel': np.array([2.2, -2.5])},
        {'pos': np.array([9.0, 4.0]), 'radius': 0.9, 'vel': np.array([-1.2, 1.2])},
        {'pos': np.array([8.0, 8.0]), 'radius': 0.6, 'vel': np.array([-1.8, -1.8])},
        {'pos': np.array([2.0, 2.0]), 'radius': 0.7, 'vel': np.array([1.0, 1.0])},
    ]
    
    # --- SETUP PREDICTORS ---
    # 1. Predictor thật cho Robot Robust
    lp_predictors = [LPLearner(history_len=15) for _ in obstacles]
    
    # 2. Predictor giả cho Robot Standard (chỉ trả về current state)
    dummy_predictors = [DummyPredictor() for _ in obstacles]
    
    # History data containers
    hist_rob_std = []
    hist_rob_robust = []
    hist_obs = [[] for _ in obstacles]
    hist_preds = [[] for _ in obstacles] 

    VIS_LOOKAHEAD_STEPS = 6
    print(f"Comparing: Standard CBF (Left) vs Robust LP-CBF (Right)")

    # --- SIMULATION LOOP ---
    for step in range(cfg.SIM_STEPS):
        # 1. Update Obstacles
        for i, obs in enumerate(obstacles):
            # Di chuyển khó lường
            ax = 1.5 * np.sin(step * 0.2 + i) 
            ay = 1.5 * np.cos(step * 0.25 + i*2)
            
            # Cập nhật LP Predictor thật
            lp_predictors[i].update(ax, ay)
            
            # Vật lý Obstacle
            obs['vel'] += np.array([ax, ay]) * cfg.DT
            obs['pos'] += obs['vel'] * cfg.DT
            
            # Va chạm tường
            if obs['pos'][0] < -2 or obs['pos'][0] > 12: obs['vel'][0] *= -1
            if obs['pos'][1] < -2 or obs['pos'][1] > 12: obs['vel'][1] *= -1
            
            hist_obs[i].append(obs['pos'].copy())
            
            # Lưu dự đoán (chỉ dùng cho bên Robust visualization)
            pred_seq = lp_predictors[i].predict_future(
                [obs['pos'][0], obs['pos'][1], obs['vel'][0], obs['vel'][1]], 
                cfg.DT, steps=VIS_LOOKAHEAD_STEPS
            )
            hist_preds[i].append(pred_seq)

        # 2. Update Hai Robot độc lập
        # Robot Standard: Dùng dummy_predictors (chỉ nhìn thấy hiện tại)
        robot_std.update(cfg.DT, obstacles, dummy_predictors)
        hist_rob_std.append(robot_std.p.copy())
        
        # Robot Robust: Dùng lp_predictors (nhìn thấy tương lai mở rộng)
        robot_robust.update(cfg.DT, obstacles, lp_predictors)
        hist_rob_robust.append(robot_robust.p.copy())
        
        # Điều kiện dừng (nếu cả 2 cùng đến đích thì tốt, không thì chạy hết time)
        dist1 = np.linalg.norm(robot_std.p - robot_std.goal)
        dist2 = np.linalg.norm(robot_robust.p - robot_robust.goal)
        if dist1 < 0.2 and dist2 < 0.2:
            print("Both Reached Goal!")
            break

    # --- VISUALIZATION (2 Subplots) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # == CẤU HÌNH AX1 (STANDARD) ==
    ax1.set_title("Standard CBF (No Prediction)\nRobot reacts only when close")
    ax1.set_xlim(-1, 13); ax1.set_ylim(-3, 13); ax1.set_aspect('equal'); ax1.grid(True)

    # --- CHÈN CODE VẼ ĐƯỜNG ĐỎ CHO AX1 Ở ĐÂY ---
    ax1.plot([0, 8], [0, 8], 'r--', linewidth=1.5, alpha=0.5, label='Ref Line', zorder=0) # <--- THÊM DÒNG NÀY
    # -------------------------------------------

    rob1_patch = plt.Circle(robot_std.p, cfg.ROBOT_RADIUS, color='gray', label='Std Robot', zorder=5)
    goal1_patch = plt.Circle(robot_std.goal, 0.3, color='green', alpha=0.5, zorder=1)
    obs_patches_1 = [plt.Circle(o['pos'], o['radius'], color='red', alpha=0.6, zorder=2) for o in obstacles]
    path1_line, = ax1.plot([], [], 'k--', linewidth=1) # Màu đen nét đứt
    
    ax1.add_patch(rob1_patch); ax1.add_patch(goal1_patch)
    for p in obs_patches_1: ax1.add_patch(p)

    # == CẤU HÌNH AX2 (ROBUST LP) ==
    ax2.set_title("Robust CBF (LP Prediction)\nRobot avoids green boxes early")
    ax2.set_xlim(-1, 13); ax2.set_ylim(-3, 13); ax2.set_aspect('equal'); ax2.grid(True)

    # --- CHÈN CODE VẼ ĐƯỜNG ĐỎ CHO AX2 Ở ĐÂY ---
    ax2.plot([0, 8], [0, 8], 'r--', linewidth=1.5, alpha=0.5, label='Ref Line', zorder=0) # <--- THÊM DÒNG NÀY
    # -------------------------------------------

    rob2_patch = plt.Circle(robot_robust.p, cfg.ROBOT_RADIUS, color='blue', label='Robust Robot', zorder=5)
    goal2_patch = plt.Circle(robot_robust.goal, 0.3, color='green', alpha=0.5, zorder=1)
    obs_patches_2 = [plt.Circle(o['pos'], o['radius'], color='red', alpha=0.6, zorder=2) for o in obstacles]
    path2_line, = ax2.plot([], [], 'b-', linewidth=1) # Màu xanh nét liền
    
    ax2.add_patch(rob2_patch); ax2.add_patch(goal2_patch)
    for p in obs_patches_2: ax2.add_patch(p)

    # Containers cho Visual dự đoán (Chỉ dùng cho AX2)
    all_pred_boxes = [] 
    all_pred_circles = []
    
    for idx, obs_data in enumerate(obstacles):
        box_seq, circ_seq = [], []
        r = obs_data['radius']
        for k in range(VIS_LOOKAHEAD_STEPS):
            alpha = max(0.2, 1.0 - k*0.12)
            c = (0.0, 0.8, 0.0, alpha)
            
            # Box
            rect = plt.Rectangle((0,0), 0, 0, linewidth=1.5, edgecolor=c, facecolor='none', linestyle='--', zorder=3)
            ax2.add_patch(rect)
            box_seq.append(rect)
            
            # Circle
            circ = plt.Circle((0,0), r, color=c, zorder=3)
            ax2.add_patch(circ)
            circ_seq.append(circ)
            
        all_pred_boxes.append(box_seq)
        all_pred_circles.append(circ_seq)

    def animate(i):
        if i >= len(hist_rob_std): return []

        # --- UPDATE AX1 (STANDARD) ---
        rob1_patch.center = hist_rob_std[i]
        path1_line.set_data([p[0] for p in hist_rob_std[:i+1]], [p[1] for p in hist_rob_std[:i+1]])
        for j, op in enumerate(obs_patches_1):
            op.center = hist_obs[j][i]
            
        # --- UPDATE AX2 (ROBUST) ---
        rob2_patch.center = hist_rob_robust[i]
        path2_line.set_data([p[0] for p in hist_rob_robust[:i+1]], [p[1] for p in hist_rob_robust[:i+1]])
        
        flat_pred_patches = []
        for j, op in enumerate(obs_patches_2):
            op.center = hist_obs[j][i]
            
            # Update prediction visuals (Boxes & Green Circles)
            preds = hist_preds[j][i]
            boxes = all_pred_boxes[j]
            circles = all_pred_circles[j]
            
            for k, dat in enumerate(preds):
                # Box
                boxes[k].set_xy((dat['x_min'], dat['y_min']))
                boxes[k].set_width(dat['x_max'] - dat['x_min'])
                boxes[k].set_height(dat['y_max'] - dat['y_min'])
                flat_pred_patches.append(boxes[k])
                
                # Green Circle
                cx = (dat['x_min'] + dat['x_max'])/2
                cy = (dat['y_min'] + dat['y_max'])/2
                circles[k].center = (cx, cy)
                flat_pred_patches.append(circles[k])

        return [rob1_patch, path1_line, rob2_patch, path2_line] + obs_patches_1 + obs_patches_2 + flat_pred_patches

    ani = animation.FuncAnimation(fig, animate, frames=len(hist_rob_std), interval=30, blit=True)
    plt.show()

if __name__ == "__main__":
    run_comparison()