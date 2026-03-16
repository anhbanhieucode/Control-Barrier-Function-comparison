import numpy as np
import cvxpy as cp
import config as cfg

class DoubleIntegratorCBF_soft:
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

    # --- UPDATED FUNCTION ---
    def solve_cbf_qp(self, u_nom, obstacles, predictors=None):
        u = cp.Variable(2)
        n_obs = len(obstacles)
        constraints = []

        if n_obs > 0:
            slacks = cp.Variable(n_obs, nonneg=True)
        else:
            slacks = None
        
        for i, obs in enumerate(obstacles):
            # --- ROBUST PREDICTION LOGIC ---
            if predictors is not None:
                # Extract state for predictor: [x, y, vx, vy]
                obs_state = [obs['pos'][0], obs['pos'][1], obs['vel'][0], obs['vel'][1]]
                
                # Get the robust center and radius from the LP learner
                # Look ahead 5 steps (0.25s) to be proactive
                p_obs_eff, r_obs_eff = predictors[i].get_robust_obstacle(
                    obs_state, obs['radius'], cfg.DT, lookahead_steps=10
                )
            else:
                # Fallback to standard if no predictor
                p_obs_eff = obs['pos']
                r_obs_eff = obs['radius']

            # --- STANDARD CBF WITH EFFECTIVE VALUES ---
            p_rel = self.p - p_obs_eff
            v_rel = self.v - obs['vel'] # Still use current relative velocity for damping
            
            dist_sq = np.sum(p_rel**2)
            min_dist = cfg.ROBOT_RADIUS + r_obs_eff # Use Expanded Radius
            
            h = dist_sq - min_dist**2
            h_dot = 2 * np.dot(p_rel, v_rel)
            
            Lgh = 2 * p_rel
            
            # Assume 0 accel for obstacle in the instantaneous constraint 
            # (uncertainty is handled by the radius expansion)
            term_dyn = -2 * np.dot(v_rel, v_rel) 
            term_barrier = -cfg.CBF_KV * h_dot - cfg.CBF_KP * h
            
            if slacks is not None:
                constraints.append(-Lgh @ u <= -(term_dyn + term_barrier) + slacks[i])
            else:
                constraints.append(-Lgh @ u <= -(term_dyn + term_barrier))

        constraints += [cp.norm(u) <= cfg.MAX_ACCEL]
        constraints += [cp.norm(self.v + u * cfg.DT) <= cfg.MAX_VEL]

        # 3. Define Objective with Penalty
        # Minimize (Control Deviation) + (Safety Violation Penalty)
        # Using a huge weight (e.g., 1e5 or 1e6) ensures "Exact Penalty" behavior
        penalty_weight = 1000000
        
        cost = cp.sum_squares( u- u_nom)

        if slacks is not None:
            cost+= penalty_weight * cp.sum(slacks)
        
        objective = cp.Minimize(cost)
        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve()
            if u.value is None: return -2.0 * self.v 
            return u.value
        except:
            return np.array([0.0, 0.0])
        

    def update(self, dt, obstacles, predictors=None):
        u_nom = self.nominal_controller()
        # Pass predictors to the solver
        u_safe = self.solve_cbf_qp(u_nom, obstacles, predictors)
        self.v += u_safe * dt
        self.p += self.v * dt
        self.path.append(self.p.copy())
        return u_safe