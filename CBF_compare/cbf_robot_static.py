import numpy as np
import cvxpy as cp
import config as cfg

class DoubleIntegratorCBFStatic:
    def __init__(self, start_pos, goal_pos):
        self.p = np.array(start_pos, dtype=float)
        self.v = np.array([0.0, 0.0], dtype=float)
        self.goal = np.array(goal_pos, dtype=float)
        self.path = [self.p.copy()]

    def nominal_controller(self):
        error_pos = self.goal - self.p
        error_vel = np.array([0.0, 0.0]) - self.v
        u_nom = cfg.NOMINAL_KP * error_pos + cfg.NOMINAL_KD * error_vel
        norm = np.linalg.norm(u_nom)
        if norm > cfg.MAX_ACCEL:
            u_nom = (u_nom / norm) * cfg.MAX_ACCEL
        return u_nom

    # --- UPDATED FUNCTION FOR STATIC OBSTACLES ---
    def solve_cbf_qp(self, u_nom, obstacles):
        u = cp.Variable(2)
        constraints = []
        
        for obs in obstacles:
            # --- STATIC LOGIC (No Predictor needed) ---
            # Vật cản đứng yên nên p_obs cố định, không cần eff/robust radius từ predictor
            p_obs = np.array(obs['pos']) 
            r_obs = obs['radius']

            # --- CBF CALCULATION ---
            p_rel = self.p - p_obs
            
            # QUAN TRỌNG: Với vật cản tĩnh, v_obs = 0
            # Nên v_rel chính là vận tốc của robot
            v_rel = self.v 
            
            dist_sq = np.sum(p_rel**2)
            min_dist = cfg.ROBOT_RADIUS + r_obs
            
            h = dist_sq - min_dist**2
            h_dot = 2 * np.dot(p_rel, v_rel)
            
            Lgh = 2 * p_rel
            
            # Đạo hàm bậc 2 của h: 2 * (v_rel^2 + p_rel * u)
            # term_dyn đại diện cho phần 2 * v_rel^2 (chuyển vế sang phải thành trừ)
            term_dyn = -2 * np.dot(v_rel, v_rel)
            
            # Alpha functions (Class K functions)
            term_barrier = -cfg.CBF_KV * h_dot - cfg.CBF_KP * h
            
            # Constraint: Lgh * u + term_dyn >= - (terms)
            # Viết lại chuẩn form <= của cvxpy: -Lgh * u <= -(term_dyn + term_barrier)
            constraints.append(-Lgh @ u <= -(term_dyn + term_barrier))

        # Ràng buộc vật lý robot
        constraints += [cp.norm(u) <= cfg.MAX_ACCEL]
        constraints += [cp.norm(self.v + u * cfg.DT) <= cfg.MAX_VEL]
        
        objective = cp.Minimize(cp.sum_squares(u - u_nom))
        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve()
            if u.value is None: 
                # Trường hợp infeasible (quá sát vật cản), phanh gấp
                return -2.0 * self.v 
            return u.value
        except:
            return np.array([0.0, 0.0])

    def update(self, dt, obstacles):
        u_nom = self.nominal_controller()
        # Không cần truyền predictors nữa
        u_safe = self.solve_cbf_qp(u_nom, obstacles)
        self.v += u_safe * dt
        self.p += self.v * dt
        self.path.append(self.p.copy())
        return u_safe