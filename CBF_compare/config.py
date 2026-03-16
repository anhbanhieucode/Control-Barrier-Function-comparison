# config.py

# --- Physical Constraints ---
ROBOT_RADIUS = 0.1
MAX_ACCEL = 30.0      # m/s^2
MAX_VEL = 300.0        # m/s

# --- Nominal Controller (PD) Gains ---
# Controls how aggressively the robot moves toward the goal
NOMINAL_KP = 2.0     # Position Gain
NOMINAL_KD = 1.5     # Velocity Gain (Damping)

# --- Control Barrier Function (CBF) Gains ---
# Controls how "safe" vs "aggressive" the avoidance is
# Higher values = allows getting closer to obstacles before reacting
CBF_KP = 20.0 
CBF_KV = 23.0

# --- Simulation Settings ---
DT = 0.05            # Time step (seconds)
SIM_STEPS = 600      # Duration of simulation