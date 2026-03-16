import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import config as cfg
from matplotlib.patches import Rectangle, Circle

# Import các module robot
from cbf_robot import DoubleIntegratorCBF
from cbf_robot_static import DoubleIntegratorCBFStatic 
from obstacle_predictor import LPLearner

# --- Helper Class ---
class DummyPredictor:
    """Predictor 'mù' cho Standard CBF"""
    def update(self, ax, ay): pass
    def predict_future(self, state, dt, steps): return []
    def get_robust_obstacle(self, obs_state, r, dt, lookahead_steps=5):
        return np.array([obs_state[0], obs_state[1]]), r

def run_triple_cbf_comparison():
    # --- 1. SETUP ---
    start_point = [0, 0]
    goal_point = [8, 8]
    
    # Ba đấu thủ CBF
    robot_static = DoubleIntegratorCBFStatic(start_point, goal_point) # 1. CBF Tĩnh
    robot_std = DoubleIntegratorCBF(start_point, goal_point)          # 2. CBF Chuẩn
    robot_robust = DoubleIntegratorCBF(start_point, goal_point)       # 3. CBF Bền vững (LP)
    
    # Danh sách Obstacles
    obstacles = [
        {'pos': np.array([5.0, 5.0]), 'radius': 0.8, 'vel': np.array([2.2, 1.5])}, 
        {'pos': np.array([3.0, 9.0]), 'radius': 0.7, 'vel': np.array([2.2, -2.5])},
        {'pos': np.array([9.0, 4.0]), 'radius': 0.9, 'vel': np.array([-1.2, 1.2])},
        {'pos': np.array([8.0, 8.0]), 'radius': 0.6, 'vel': np.array([-1.8, -1.8])},
        {'pos': np.array([2.0, 2.0]), 'radius': 0.7, 'vel': np.array([1.0, 1.0])},
        {'pos': np.array([3.0, 3.0]), 'radius': 0.7, 'vel': np.array([1.0, 1.0])},
        {'pos': np.array([5.0, 2.0]), 'radius': 0.7, 'vel': np.array([3.0, 1.0])},
        {'pos': np.array([5.0, 6.0]), 'radius': 0.7, 'vel': np.array([1.0, 2.0])}
    ]
    
    # Predictors
    lp_predictors = [LPLearner(history_len=15) for _ in obstacles] 
    dummy_predictors = [DummyPredictor() for _ in obstacles]       
    
    # Data Logs
    hist_static, hist_std, hist_robust = [], [], []
    hist_obs = [[] for _ in obstacles]
    hist_preds = [[] for _ in obstacles] # <--- MỚI: Lưu lịch sử dự đoán
    
    VIS_LOOKAHEAD = 5 # Số bước vẽ dự đoán

    print("--- STARTING CBF EVOLUTION COMPARISON ---")
    print("1. Static CBF (Assumes v_obs = 0)")
    print("2. Standard CBF (Uses v_obs, No Prediction)")
    print("3. Robust CBF (Uses v_obs + LP Prediction)")

    # --- 2. SIMULATION LOOP ---
    for step in range(cfg.SIM_STEPS):
        # A. Update Obstacles & Predictors
        for i, obs in enumerate(obstacles):
            ax = 1.5 * np.sin(step * 0.2 + i) 
            ay = 1.5 * np.cos(step * 0.25 + i*2)
            
            # Update Learner
            lp_predictors[i].update(ax, ay)
            
            # Update Physics
            obs['vel'] += np.array([ax, ay]) * cfg.DT
            obs['pos'] += obs['vel'] * cfg.DT
            if obs['pos'][0] < -2 or obs['pos'][0] > 12: obs['vel'][0] *= -1
            if obs['pos'][1] < -2 or obs['pos'][1] > 12: obs['vel'][1] *= -1
            
            hist_obs[i].append(obs['pos'].copy())
            
            # <--- MỚI: Tính toán và lưu dự đoán để vẽ --->
            # Lấy trạng thái hiện tại [x, y, vx, vy]
            current_state = [obs['pos'][0], obs['pos'][1], obs['vel'][0], obs['vel'][1]]
            pred_seq = lp_predictors[i].predict_future(current_state, cfg.DT, steps=VIS_LOOKAHEAD)
            hist_preds[i].append(pred_seq)

        # B. Update Robots
        # 1. Static CBF
        robot_static.update(cfg.DT, obstacles) 
        hist_static.append(robot_static.p.copy())

        # 2. Standard CBF
        robot_std.update(cfg.DT, obstacles, dummy_predictors)
        hist_std.append(robot_std.p.copy())
        
        # 3. Robust CBF
        robot_robust.update(cfg.DT, obstacles, lp_predictors)
        hist_robust.append(robot_robust.p.copy())

        # Check về đích
        d1 = np.linalg.norm(robot_static.p - goal_point)
        d2 = np.linalg.norm(robot_std.p - goal_point)
        d3 = np.linalg.norm(robot_robust.p - goal_point)
        
        if d1 < 0.2 and d2 < 0.2 and d3 < 0.2:
            print(f"All robots reached goal at step {step}")
            break

    # --- 3. VISUALIZATION ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    axes = [ax1, ax2, ax3]

    def setup_ax(ax, title):
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlim(-2, 12); ax.set_ylim(-2, 12)
        ax.set_aspect('equal'); ax.grid(True, linestyle=':', alpha=0.6)
        ax.plot([start_point[0], goal_point[0]], [start_point[1], goal_point[1]], 
                'r--', alpha=0.4, linewidth=1, zorder=0, label='Ref')

    setup_ax(ax1, "1. Static CBF\n(Assumes obs stops - Dangerous)")
    setup_ax(ax2, "2. Standard CBF\n(Reactive - Safe but Late)")
    setup_ax(ax3, "3. Robust CBF\n(Predictive - Green Boxes show Safety Margins)") # <--- Updated Title

    # Patches cho Robot
    p_static = plt.Circle(start_point, cfg.ROBOT_RADIUS, color='orange', zorder=5, label='Static')
    p_std    = plt.Circle(start_point, cfg.ROBOT_RADIUS, color='gray', zorder=5, label='Standard')
    p_robust = plt.Circle(start_point, cfg.ROBOT_RADIUS, color='blue', zorder=5, label='Robust')
    
    patches = [p_static, p_std, p_robust]
    hists = [hist_static, hist_std, hist_robust]
    trails = []
    
    # <--- MỚI: Containers cho các hình vẽ dự đoán (Chỉ dùng cho AX3) --->
    # Cấu trúc: list[obstacle_idx][step_idx] -> Patch Object
    pred_boxes_patches = [] 
    
    # Khởi tạo Visuals cơ bản
    obs_patches_list = []
    for i, ax in enumerate(axes):
        ax.add_patch(plt.Circle(goal_point, 0.3, color='green', alpha=0.4))
        ax.add_patch(patches[i])
        
        line, = ax.plot([], [], color=patches[i].get_facecolor(), linewidth=1.5)
        trails.append(line)
        
        # Obstacles (Màu đỏ)
        obs_patches = [plt.Circle(o['pos'], o['radius'], color='red', alpha=0.6, zorder=4) for o in obstacles]
        for op in obs_patches: ax.add_patch(op)
        obs_patches_list.append(obs_patches)

        # <--- MỚI: Chỉ thêm dự đoán màu xanh vào AX3 --->
        if i == 2: # ax3 là Robust CBF
            for o_idx in range(len(obstacles)):
                obs_box_seq = []
                for k in range(VIS_LOOKAHEAD):
                    # Làm mờ dần theo thời gian (alpha giảm dần)
                    alpha_val = max(0.1, 0.6 - k * 0.03)
                    
                    # Vẽ hộp dự đoán (LP Box)
                    rect = Rectangle((0,0), 0, 0, linewidth=1, edgecolor='green', 
                                     facecolor='none', linestyle='--', alpha=alpha_val, zorder=2)
                    ax.add_patch(rect)
                    obs_box_seq.append(rect)
                pred_boxes_patches.append(obs_box_seq)

    def animate(i):
        if i >= len(hist_std): return []
        
        artists = []
        
        # 1. Update Robot & Trail & Obstacles cho cả 3 subplot
        for j, (patch, hist, trail, obs_patches) in enumerate(zip(patches, hists, trails, obs_patches_list)):
            patch.center = hist[i]
            trail.set_data([p[0] for p in hist[:i+1]], [p[1] for p in hist[:i+1]])
            artists.append(patch)
            artists.append(trail)
            
            curr_obs_pos = [hist_obs[k][i] for k in range(len(obstacles))]
            for k, op in enumerate(obs_patches):
                op.center = curr_obs_pos[k]
                artists.append(op)
        
        # 2. <--- MỚI: Update Prediction Visuals (Chỉ cho AX3) --->
        # Duyệt qua từng obstacle
        for o_idx in range(len(obstacles)):
            # Lấy chuỗi dự đoán tại thời điểm i của obstacle này
            preds_at_step = hist_preds[o_idx][i]
            
            # Duyệt qua từng bước nhìn trước (lookahead step)
            for k in range(VIS_LOOKAHEAD):
                if k < len(preds_at_step):
                    box_data = preds_at_step[k]
                    rect = pred_boxes_patches[o_idx][k]
                    
                    # Cập nhật vị trí và kích thước hình chữ nhật
                    rect.set_xy((box_data['x_min'], box_data['y_min']))
                    rect.set_width(box_data['x_max'] - box_data['x_min'])
                    rect.set_height(box_data['y_max'] - box_data['y_min'])
                    
                    artists.append(rect)

        return artists

    ani = animation.FuncAnimation(fig, animate, frames=len(hist_std), interval=30, blit=True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_triple_cbf_comparison()