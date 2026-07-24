"""Dynamic Window Approach (DWA): a real-time local path planner.

At every control step it searches a short window of feasible (v, w)
velocity commands, simulates where each would take the robot over a short
horizon, scores each rollout against the goal / obstacles / speed, and
executes only the best one's first step. Because the whole search is redone
every step from the latest LiDAR scan, the robot naturally reacts to
obstacles it has just seen.

Reference:
    Fox, D., Burgard, W., & Thrun, S. (1997). The dynamic window approach
    to collision avoidance. IEEE Robotics & Automation Magazine, 4(1), 23-33.

Coordinate System:
    - Robot pose is (x, y, theta) in meters and radians
    - Robot velocity is (v, w): linear speed [m/s] and angular rate [rad/s]
    - Positive w is counterclockwise rotation

Units:
    - Distances: meters (m)
    - Velocities: meters/second (m/s) for linear, rad/s for angular
    - Time: seconds (s)

Example:
    >>> import numpy as np
    >>> from src.planning import DWAConfig, plan
    >>> config = DWAConfig()
    >>> pose = np.array([0.0, 0.0, 0.0])
    >>> goal = np.array([5.0, 5.0])
    >>> obstacles = np.array([[3, 3], [4, 2]])
    >>> v, w, traj = plan(pose, 0.0, 0.0, goal, obstacles, config)
"""
import numpy as np


class DWAConfig:
    """Configuration parameters for the Dynamic Window Approach planner.
    
    Attributes:
        max_speed: Maximum forward velocity in m/s.
        min_speed: Minimum velocity (typically 0 to prevent reversing).
        max_yaw_rate: Maximum angular velocity in rad/s.
        max_accel: Maximum linear acceleration in m/s^2.
        max_delta_yaw_rate: Maximum angular acceleration in rad/s^2.
        v_resolution: Linear velocity sampling resolution in m/s.
        yaw_rate_resolution: Angular velocity sampling resolution in rad/s.
        dt: Control step time interval in seconds.
        predict_time: Trajectory prediction horizon in seconds.
        to_goal_cost_gain: Weight for the distance-to-goal cost.
        speed_cost_gain: Weight for the speed cost (prefer higher speed).
        obstacle_cost_gain: Weight for the obstacle proximity cost.
        robot_radius: Robot radius for collision checking in meters.
    """
    
    def __init__(self) -> None:
        """Initialize DWA configuration with default values."""
        self.max_speed = 1.0          # [m/s]
        self.min_speed = 0.0          # [m/s] (no reversing)
        self.max_yaw_rate = 90.0 * np.pi / 180.0    # [rad/s]
        self.max_accel = 1.0          # [m/s^2]
        self.max_delta_yaw_rate = 90.0 * np.pi / 180.0  # [rad/s^2]

        self.v_resolution = 0.05       # [m/s]
        self.yaw_rate_resolution = 5.0 * np.pi / 180.0  # [rad/s]
        self.dt = 0.1                  # [s] control / simulation step
        self.predict_time = 2.0        # [s] rollout horizon

        self.to_goal_cost_gain = 1.0
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 1.0

        self.robot_radius = 0.3        # [m] used for collision checking


def motion(pose: np.ndarray, v: float, w: float, dt: float) -> np.ndarray:
    """Compute new robot pose after applying velocity for dt seconds.
    
    Uses the unicycle motion model:
        x' = x + v * cos(theta) * dt
        y' = y + v * sin(theta) * dt
        theta' = theta + w * dt
    
    Args:
        pose: Current pose [x, y, theta] in meters and radians.
        v: Linear velocity in m/s.
        w: Angular velocity in rad/s.
        dt: Time step in seconds.
    
    Returns:
        New pose [x', y', theta'] after applying velocities.
    """
    x, y, theta = pose
    theta = theta + w * dt
    x = x + v * np.cos(theta) * dt
    y = y + v * np.sin(theta) * dt
    return np.array([x, y, theta])


def dynamic_window(v: float, w: float, config: DWAConfig) -> tuple:
    """Compute the dynamic window of reachable velocities.
    
    The dynamic window is the intersection of:
    1. Velocities reachable given acceleration limits
    2. Hard velocity limits
    
    Args:
        v: Current linear velocity in m/s.
        w: Current angular velocity in rad/s.
        config: DWA configuration object.
    
    Returns:
        Tuple (v_min, v_max, w_min, w_max) defining the search window.
    """
    v_min = max(config.min_speed, v - config.max_accel * config.dt)
    v_max = min(config.max_speed, v + config.max_accel * config.dt)
    w_min = max(-config.max_yaw_rate, w - config.max_delta_yaw_rate * config.dt)
    w_max = min(config.max_yaw_rate, w + config.max_delta_yaw_rate * config.dt)
    return v_min, v_max, w_min, w_max


def predict_trajectory(
    pose: np.ndarray,
    v: float,
    w: float,
    config: DWAConfig
) -> np.ndarray:
    """Predict robot trajectory over the planning horizon.
    
    Args:
        pose: Starting pose [x, y, theta] in meters and radians.
        v: Constant linear velocity in m/s.
        w: Constant angular velocity in rad/s.
        config: DWA configuration object.
    
    Returns:
        Array of shape (n_steps, 3) containing pose at each time step,
        including the starting pose.
    """
    traj = [pose]
    p = pose
    t = 0.0
    while t <= config.predict_time:
        p = motion(p, v, w, config.dt)
        traj.append(p)
        t += config.dt
    return np.array(traj)


def calc_obstacle_cost(
    traj: np.ndarray,
    obstacle_points: np.ndarray,
    config: DWAConfig
) -> float:
    """Calculate cost based on proximity to obstacle points.
    
    Trajectories that collide (min distance <= robot_radius) receive
    infinite cost and are rejected.
    
    Args:
        traj: Predicted trajectory as array (n_points, 3).
        obstacle_points: Array (n_obstacles, 2) of obstacle coordinates.
        config: DWA configuration object.
    
    Returns:
        Cost value: 1/min_distance for valid trajectories, inf for collisions.
    """
    if len(obstacle_points) == 0:
        return 0.0
    xy = traj[:, :2]
    diff = xy[:, None, :] - obstacle_points[None, :, :]
    dist = np.hypot(diff[..., 0], diff[..., 1])
    min_dist = dist.min()
    if min_dist <= config.robot_radius:
        return np.inf  # collision -> reject this trajectory
    return 1.0 / min_dist


def calc_to_goal_cost(traj: np.ndarray, goal: np.ndarray) -> float:
    """Calculate cost based on distance to goal.
    
    Uses Euclidean distance from trajectory endpoint to goal.
    This rewards trajectories that bring the robot closer to the goal,
    even if not directly facing it.
    
    Args:
        traj: Predicted trajectory as array (n_points, 3).
        goal: Goal position [x, y] in meters.
    
    Returns:
        Euclidean distance to goal from trajectory endpoint.
    """
    dx = goal[0] - traj[-1, 0]
    dy = goal[1] - traj[-1, 1]
    return np.hypot(dx, dy)


def plan(
    pose: np.ndarray,
    v: float,
    w: float,
    goal: np.ndarray,
    obstacle_points: np.ndarray,
    config: DWAConfig
) -> tuple:
    """Compute optimal velocity command using Dynamic Window Approach.
    
    Searches the dynamic window of feasible velocities, simulates each
    candidate trajectory, scores by (goal + speed + obstacle) costs,
    and returns the best command.
    
    Args:
        pose: Current robot pose [x, y, theta] in meters and radians.
        v: Current linear velocity in m/s.
        w: Current angular velocity in rad/s.
        goal: Target position [x, y] in meters.
        obstacle_points: Array (n, 2) of obstacle coordinates from LiDAR.
        config: DWA configuration object.
    
    Returns:
        Tuple (best_v, best_w, best_trajectory) where:
        - best_v: Optimal linear velocity in m/s.
        - best_w: Optimal angular velocity in rad/s.
        - best_trajectory: Array (n_steps, 3) of predicted poses.
    """
    v_min, v_max, w_min, w_max = dynamic_window(v, w, config)

    best_cost = np.inf
    best_u = (0.0, 0.0)
    best_traj = np.array([pose])

    for v_cmd in np.arange(v_min, v_max + 1e-9, config.v_resolution):
        for w_cmd in np.arange(w_min, w_max + 1e-9, config.yaw_rate_resolution):
            traj = predict_trajectory(pose, v_cmd, w_cmd, config)

            cost = (
                config.to_goal_cost_gain * calc_to_goal_cost(traj, goal)
                + config.speed_cost_gain * (config.max_speed - v_cmd)
                + config.obstacle_cost_gain * calc_obstacle_cost(traj, obstacle_points, config)
            )

            if cost < best_cost:
                best_cost = cost
                best_u = (v_cmd, w_cmd)
                best_traj = traj

    return best_u[0], best_u[1], best_traj
