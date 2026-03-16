import numpy as np
from scipy.optimize import linprog

class LPLearner:
    def __init__(self, history_len=10):
        self.ax_hist = []
        self.ay_hist = []
        self.history_len = history_len

    def update(self, ax, ay):
        """Stores acceleration history"""
        self.ax_hist.append(ax)
        self.ay_hist.append(ay)
        if len(self.ax_hist) > self.history_len:
            self.ax_hist.pop(0)
            self.ay_hist.pop(0)

    def _solve_bounds(self, data):
        """Solves LP to find Min/Max acceleration bounds"""
        if len(data) < 3: return -0.5, 0.5 # Fallback if not enough data
        
        c = [-1, 1] # Maximize difference (Max - Min)
        A_ub = []
        b_ub = []
        
        for d in data:
            # Constraints: u_min <= d AND d <= u_max
            A_ub.append([1, 0]); b_ub.append(d)   
            A_ub.append([0, -1]); b_ub.append(-d) 
            
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(None, None), method='highs')
            if res.success:
                return res.x[0], res.x[1]
        except:
            pass
        return -1.0, 1.0 # Safe fallback

    def predict_future(self, current_state, dt, steps):
        """
        Predicts future boxes.
        current_state: [x, y, vx, vy]
        """
        ux_min, ux_max = self._solve_bounds(self.ax_hist)
        uy_min, uy_max = self._solve_bounds(self.ay_hist)

        # Tune this scale to trust/distrust the prediction expansion
        ACCEL_SCALE = 3.5 

        ux_min *= ACCEL_SCALE
        ux_max *= ACCEL_SCALE
        uy_min *= ACCEL_SCALE
        uy_max *= ACCEL_SCALE
        
        preds = []
        x, y, vx, vy = current_state
        
        for k in range(1, steps + 1):
            t = k * dt
            # Kinematic expansion
            box = {
                'x_min': x + vx*t + 0.5*ux_min*t**2,
                'x_max': x + vx*t + 0.5*ux_max*t**2,
                'y_min': y + vy*t + 0.5*uy_min*t**2,
                'y_max': y + vy*t + 0.5*uy_max*t**2
            }
            preds.append(box)
            
        return preds

    def get_robust_obstacle(self, obs_state, obs_radius, dt, lookahead_steps=8):
        """
        Converts the predicted box into a Circle for the CBF.
        Returns: predicted_center, effective_radius
        """
        # Get prediction at the lookahead step
        preds = self.predict_future(obs_state, dt, lookahead_steps)
        target_box = preds[-1] # Look at the furthest prediction
        
        # 1. Calculate Center of the uncertain box
        center_x = (target_box['x_min'] + target_box['x_max']) / 2
        center_y = (target_box['y_min'] + target_box['y_max']) / 2
        pred_center = np.array([center_x, center_y])
        
        # 2. Calculate "Uncertainty Radius" (Half diagonal of the box)
        width = target_box['x_max'] - target_box['x_min']
        height = target_box['y_max'] - target_box['y_min']
        uncertainty_radius = np.sqrt(width**2 + height**2) / 2
        
        # 3. Total effective radius
        total_radius = obs_radius + uncertainty_radius 
        
        return pred_center, total_radius