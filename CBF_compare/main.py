import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import config as cfg
from cbf_robot import DoubleIntegratorCBF
from obstacle_predictor import LPLearner

def run_simulation():
    # Setup
    robot = DoubleIntegratorCBF(start_pos=[0, 0], goal_pos=[10, 10])
    
    # Obstacles with erratic movement
# --- TĂNG ĐỘ KHÓ: THÊM OBSTACLE & TĂNG TỐC ĐỘ ---
    obstacles = [
        # Obstacle 1: Giữ đường chéo
        {'pos': np.array([5.0, 5.0]), 'radius': 0.8, 'vel': np.array([1.5, 0.7])}, 
        # Obstacle 2: Lao từ trên xuống nhanh
        {'pos': np.array([3.0, 9.0]), 'radius': 0.7, 'vel': np.array([0.5, -1.7])},
        # Obstacle 3: Lao từ phải sang trái
        {'pos': np.array([9.0, 4.0]), 'radius': 0.9, 'vel': np.array([-1.6, 0.4])},
        # Obstacle 4: Chặn gần đích
        {'pos': np.array([8.0, 8.0]), 'radius': 0.6, 'vel': np.array([-0.8, -0.9])},
        # Obstacle 5: Quấy rối ngay lúc xuất phát
        {'pos': np.array([2.0, 2.0]), 'radius': 0.7, 'vel': np.array([1.3, 1.2])},
    ]
    
    # Initialize Predictors (One per obstacle)
    predictors = [LPLearner(history_len=15) for _ in obstacles]
    
    history_robot = []
    history_obs = [[] for _ in obstacles]
    history_preds = [[] for _ in obstacles] 

    # How many steps ahead to visualize
    VIS_LOOKAHEAD_STEPS = 8

    print(f"Starting Robust CBF Simulation...")

    for step in range(cfg.SIM_STEPS):
        # 1. Update Obstacles (Wandering Logic)
        for i, obs in enumerate(obstacles):
            # Calculate Acceleration (simulated random movement)
            # Sine wave acceleration to create changing bounds
            ax = 0.6 * np.sin(step * 0.15) 
            ay = 0.6 * np.cos(step * 0.18)
            
            # Update Predictor with NEW acceleration data
            predictors[i].update(ax, ay)
            
            # Physics update for obstacle
            obs['vel'] += np.array([ax, ay]) * cfg.DT
            obs['pos'] += obs['vel'] * cfg.DT
            
            # Bounce bounds
            if obs['pos'][0] < -2 or obs['pos'][0] > 12: obs['vel'][0] *= -1
            if obs['pos'][1] < -2 or obs['pos'][1] > 12: obs['vel'][1] *= -1
            
            history_obs[i].append(obs['pos'].copy())
            
            # --- NEW: Save the entire sequence of future predictions ---
            pred_sequence = predictors[i].predict_future(
                [obs['pos'][0], obs['pos'][1], obs['vel'][0], obs['vel'][1]], 
                cfg.DT, steps=VIS_LOOKAHEAD_STEPS
            )
            history_preds[i].append(pred_sequence)

        # 2. Update Robot (Using Predictors for CBF)
        # Note: The CBF might use a different lookahead (e.g., 5 steps defined in obstacle_predictor.py)
        # than the visualization (VIS_LOOKAHEAD_STEPS=8). This is fine.
        robot.update(cfg.DT, obstacles, predictors)
        history_robot.append(robot.p.copy())
        
        if np.linalg.norm(robot.p - robot.goal) < 0.2:
            print("Goal Reached!")
            break

# --- Visualization with Prediction Trails ---
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_xlim(-3, 13)
    ax.set_ylim(-3, 13)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title("Robust CBF with Future Prediction Visualization")

    # 1. Vẽ Robot, Goal và Obstacles hiện tại
    # zorder cao hơn để robot luôn nổi lên trên
    robot_patch = plt.Circle(robot.p, cfg.ROBOT_RADIUS, color='blue', label='Robot', zorder=5)
    goal_patch = plt.Circle(robot.goal, 0.3, color='green', alpha=0.5, label='Goal', zorder=1)
    # zorder=2 cho obstacle hiện tại (màu đỏ)
    obs_patches = [plt.Circle(o['pos'], o['radius'], color='red', alpha=0.6, zorder=2) for o in obstacles]
    path_line, = ax.plot([], [], 'b-', linewidth=1, zorder=1)
    
    ax.add_patch(robot_patch)
    ax.add_patch(goal_patch)
    for p in obs_patches: ax.add_patch(p)

    # --- NEW: Create multiple prediction BOXES AND CIRCLES per obstacle ---
    # Chúng ta cần 2 list riêng biệt để quản lý hộp và hình tròn dự đoán
    all_pred_boxes = [] 
    all_pred_circles = [] # List mới để chứa các hình tròn dự đoán

    # Cần dùng enumerate để lấy thông tin radius của từng obstacle
    for idx, obs_data in enumerate(obstacles): 
        obs_box_sequence = []
        obs_circle_sequence = []
        
        obs_radius = obs_data['radius'] # Lấy bán kính của obstacle này

        for k in range(VIS_LOOKAHEAD_STEPS):
            # Tính toán màu sắc (Xanh lá cây nhạt dần)
            alpha_box = max(0.2, 1.0 - (k * 0.12)) 
            color_box = (0.0, 0.8, 0.0, alpha_box)
            
            # Alpha cho hình tròn nhỏ hơn một chút cho đỡ rối mắt
            alpha_circle = max(0.1, 0.6 - (k * 0.1))
            color_circle = (0.0, 0.8, 0.0, alpha_circle)

            # 1. Tạo hình hộp chữ nhật (Rectangle) - zorder=3
            rect = plt.Rectangle((0,0), 0, 0, linewidth=1.5, 
                                 edgecolor=color_box, facecolor='none', linestyle='--', 
                                 zorder=3)
            ax.add_patch(rect)
            obs_box_sequence.append(rect)

            # 2. Tạo hình tròn dự đoán (Circle) - zorder=3 (cùng lớp với hộp)
            # Khởi tạo tạm ở vị trí (0,0), sẽ cập nhật trong animate
            circle_pred = plt.Circle((0,0), obs_radius, color=color_circle, zorder=3)
            ax.add_patch(circle_pred)
            obs_circle_sequence.append(circle_pred)

        all_pred_boxes.append(obs_box_sequence)
        all_pred_circles.append(obs_circle_sequence)


    def animate(i):
        if i >= len(history_robot): return []
        
        # Cập nhật Robot và đường đi
        robot_patch.center = history_robot[i]
        path_line.set_data([p[0] for p in history_robot[:i+1]], [p[1] for p in history_robot[:i+1]])
        
        flat_patches_to_draw = [] # Danh sách gom tất cả các patch cần vẽ lại

        for j, obs_patch in enumerate(obs_patches):
            # 1. Update vị trí Obstacle hiện tại (màu đỏ)
            obs_patch.center = history_obs[j][i]
            
            # 2. Update chuỗi dự đoán tương lai
            prediction_data_sequence = history_preds[j][i] # Dữ liệu tính toán
            boxes_sequence_patches = all_pred_boxes[j]     # Các object hình hộp
            circles_sequence_patches = all_pred_circles[j] # Các object hình tròn
            
            for k, box_data in enumerate(prediction_data_sequence):
                # A. Cập nhật hình hộp (Rectangle)
                rect_patch = boxes_sequence_patches[k]
                width = box_data['x_max'] - box_data['x_min']
                height = box_data['y_max'] - box_data['y_min']
                rect_patch.set_xy((box_data['x_min'], box_data['y_min']))
                rect_patch.set_width(width)
                rect_patch.set_height(height)
                flat_patches_to_draw.append(rect_patch)

                # B. Cập nhật hình tròn dự đoán (Circle) ở tâm hộp
                circle_patch = circles_sequence_patches[k]
                center_x = (box_data['x_min'] + box_data['x_max']) / 2
                center_y = (box_data['y_min'] + box_data['y_max']) / 2
                circle_patch.center = (center_x, center_y)
                flat_patches_to_draw.append(circle_patch)
            
        # Trả về danh sách tất cả các yếu tố cần vẽ lại trong frame này
        return [robot_patch, path_line] + obs_patches + flat_patches_to_draw

    # blit=True rất quan trọng để animation mượt mà khi vẽ nhiều đối tượng
    ani = animation.FuncAnimation(fig, animate, frames=len(history_robot), interval=30, blit=True)
    plt.show()

if __name__ == "__main__":
    run_simulation()