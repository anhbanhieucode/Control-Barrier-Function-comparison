import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
import config as cfg

# Import các module robot (đảm bảo bạn đã có các file này)
from cbf_robot import DoubleIntegratorCBF
from cbf_robot_static import DoubleIntegratorCBFStatic 
from cbf_robot_softcon import DoubleIntegratorCBF_soft
from obstacle_predictor import LPLearner

# --- Helper Class ---
class DummyPredictor:
    """Predictor 'mù' cho Standard CBF"""
    def update(self, ax, ay): pass
    def predict_future(self, state, dt, steps): return []
    def get_robust_obstacle(self, obs_state, r, dt, lookahead_steps=5):
        return np.array([obs_state[0], obs_state[1]]), r

def run_quad_comparison():
    # --- 1. SETUP ---
    start_point = [0, 0]
    goal_point = [8, 8]
    
    # Bốn đấu thủ
    robot_static = DoubleIntegratorCBFStatic(start_point, goal_point) # 1. Static
    robot_std    = DoubleIntegratorCBF(start_point, goal_point)       # 2. Standard
    robot_robust = DoubleIntegratorCBF(start_point, goal_point)       # 3. Robust CBF
    robot_soft    = DoubleIntegratorCBF_soft(start_point, goal_point)       # 4. Robust DWA
    
    # Danh sách Obstacles (HARD MODE)
    obstacles = [
        {'pos': np.array([5.0, 5.0]), 'radius': 0.8, 'vel': np.array([2.2, 1.5])}, 
        {'pos': np.array([3.0, 9.0]), 'radius': 0.7, 'vel': np.array([2.2, -2.5])},
        {'pos': np.array([9.0, 4.0]), 'radius': 0.9, 'vel': np.array([-1.2, 1.2])},
        {'pos': np.array([8.0, 8.0]), 'radius': 0.6, 'vel': np.array([-1.8, -1.8])},
        {'pos': np.array([2.0, 2.0]), 'radius': 0.7, 'vel': np.array([1.0, 1.0])},
    ]
    
    # Predictors
    lp_predictors = [LPLearner(history_len=15) for _ in obstacles] 
    dummy_predictors = [DummyPredictor() for _ in obstacles]       
    
    # Data Logs
    hist_static, hist_std, hist_robust, hist_dwa = [], [], [], []
    hist_obs = [[] for _ in obstacles]
    hist_preds = [[] for _ in obstacles]
    
    VIS_LOOKAHEAD = 18

    print("--- STARTING QUAD COMPARISON ---")
    print("1. Static CBF | 2. Standard CBF | 3. Robust CBF | 4. Robust DWA")

    # --- 2. SIMULATION LOOP ---
    for step in range(cfg.SIM_STEPS):
        # A. Update Obstacles
        for i, obs in enumerate(obstacles):
            ax = 1.5 * np.sin(step * 0.2 + i) 
            ay = 1.5 * np.cos(step * 0.25 + i*2)
            
            lp_predictors[i].update(ax, ay)
            
            obs['vel'] += np.array([ax, ay]) * cfg.DT
            obs['pos'] += obs['vel'] * cfg.DT
            if obs['pos'][0] < -2 or obs['pos'][0] > 12: obs['vel'][0] *= -1
            if obs['pos'][1] < -2 or obs['pos'][1] > 12: obs['vel'][1] *= -1
            
            hist_obs[i].append(obs['pos'].copy())
            
            # Lưu dự đoán để vẽ
            curr_state = [obs['pos'][0], obs['pos'][1], obs['vel'][0], obs['vel'][1]]
            pred_seq = lp_predictors[i].predict_future(curr_state, cfg.DT, steps=VIS_LOOKAHEAD)
            hist_preds[i].append(pred_seq)

        # B. Update Robots
        # 1. Static
        robot_static.update(cfg.DT, obstacles) 
        hist_static.append(robot_static.p.copy())

        # 2. Standard
        robot_std.update(cfg.DT, obstacles, dummy_predictors)
        hist_std.append(robot_std.p.copy())
        
        # 3. Robust CBF
        robot_robust.update(cfg.DT, obstacles, lp_predictors)
        hist_robust.append(robot_robust.p.copy())
        
        # 4. Robust DWA
        robot_soft.update(cfg.DT, obstacles, lp_predictors)
        hist_dwa.append(robot_soft.p.copy())

        # Check về đích
        dists = [np.linalg.norm(r.p - goal_point) for r in [robot_static, robot_std, robot_robust, robot_soft]]
        if all(d < 0.2 for d in dists):
            print(f"All robots reached goal at step {step}")
            break

    # --- 3. VISUALIZATION ---
    # Bố cục 2x2
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 18))
    axes = [ax1, ax2, ax3, ax4]

    def setup_ax(ax, title):
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlim(-2, 16); ax.set_ylim(-2, 16)
        ax.set_aspect('equal'); ax.grid(True, linestyle=':', alpha=0.6)
        ax.plot([start_point[0], goal_point[0]], [start_point[1], goal_point[1]], 
                'r--', alpha=0.4, linewidth=1, zorder=0, label='Ref')

    setup_ax(ax1, "1. Static CBF\n(Assumes obs stops - Dangerous)")
    setup_ax(ax2, "2. Standard CBF\n(Reactive - Safe but Late)")
    setup_ax(ax3, "3. Robust CBF\n(Predictive Filter - Green Boxes)")
    setup_ax(ax4, "4. Robust soft_CBF\n(Predictive Planner - IDKman_I'm tired)")

    # Robot Patches
    patches = [
        plt.Circle(start_point, cfg.ROBOT_RADIUS, color='orange', zorder=5, label='Static'),
        plt.Circle(start_point, cfg.ROBOT_RADIUS, color='gray', zorder=5, label='Standard'),
        plt.Circle(start_point, cfg.ROBOT_RADIUS, color='blue', zorder=5, label='Robust'),
        plt.Circle(start_point, cfg.ROBOT_RADIUS, color='purple', zorder=5, label='soft_constraints')
    ]
    hists = [hist_static, hist_std, hist_robust, hist_dwa]
    trails = []
    
    # Containers cho Visual dự đoán (Cho ax3 và ax4)
    # pred_boxes_patches_3[obstacle_idx][step_idx]
    pred_boxes_patches_3 = [] 
    pred_boxes_patches_4 = []

    # Khởi tạo Visuals cơ bản
    obs_patches_list = []
    
    for i, ax in enumerate(axes):
        ax.add_patch(plt.Circle(goal_point, 0.3, color='green', alpha=0.4))
        ax.add_patch(patches[i])
        
        line, = ax.plot([], [], color=patches[i].get_facecolor(), linewidth=1.5)
        trails.append(line)
        
        # Obstacles (Đỏ)
        obs_patches = [plt.Circle(o['pos'], o['radius'], color='red', alpha=0.6, zorder=4) for o in obstacles]
        for op in obs_patches: ax.add_patch(op)
        obs_patches_list.append(obs_patches)

        # Tạo Green Boxes cho AX3 (Robust CBF) và AX4 (Robust DWA)
        if i in [2, 3]: 
            current_box_container = []
            for o_idx in range(len(obstacles)):
                obs_box_seq = []
                for k in range(VIS_LOOKAHEAD):
                    alpha_val = max(0.05, 0.4 - k * 0.02) # Mờ dần
                    rect = Rectangle((0,0), 0, 0, linewidth=1, edgecolor='green', 
                                     facecolor='none', linestyle='--', alpha=alpha_val, zorder=2)
                    ax.add_patch(rect)
                    obs_box_seq.append(rect)
                current_box_container.append(obs_box_seq)
            
            if i == 2: pred_boxes_patches_3 = current_box_container
            if i == 3: pred_boxes_patches_4 = current_box_container

    def animate(i):
        if i >= len(hist_std): return []
        
        artists = []
        
        # 1. Update Robot, Trail, Obstacles
        for j, (patch, hist, trail, obs_patches) in enumerate(zip(patches, hists, trails, obs_patches_list)):
            patch.center = hist[i]
            trail.set_data([p[0] for p in hist[:i+1]], [p[1] for p in hist[:i+1]])
            artists.append(patch)
            artists.append(trail)
            
            curr_obs_pos = [hist_obs[k][i] for k in range(len(obstacles))]
            for k, op in enumerate(obs_patches):
                op.center = curr_obs_pos[k]
                artists.append(op)
        
        # 2. Update Prediction Boxes (Cho cả AX3 và AX4)
        # Hàm helper để update 1 bộ boxes
        def update_boxes(box_container):
            for o_idx in range(len(obstacles)):
                preds_at_step = hist_preds[o_idx][i]
                for k in range(VIS_LOOKAHEAD):
                    if k < len(preds_at_step):
                        box_data = preds_at_step[k]
                        rect = box_container[o_idx][k]
                        rect.set_xy((box_data['x_min'], box_data['y_min']))
                        rect.set_width(box_data['x_max'] - box_data['x_min'])
                        rect.set_height(box_data['y_max'] - box_data['y_min'])
                        artists.append(rect)

        update_boxes(pred_boxes_patches_3)
        update_boxes(pred_boxes_patches_4)

        return artists

    ani = animation.FuncAnimation(fig, animate, frames=len(hist_std), interval=30, blit=True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_quad_comparison()