"""Dynamic Window Approach (DWA): a real-time local path planner.

At every control step it searches a short window of feasible (v, w)
velocity commands, simulates where each would take the robot over a short
horizon, scores each rollout against the goal / obstacles / speed, and
executes only the best one's first step. Because the whole search is redone
every step from the latest LiDAR scan, the robot naturally reacts to
obstacles it has just seen.
"""
import numpy as np


class DWAConfig:
    max_speed = 1.0          # [m/s]
    min_speed = 0.0          # [m/s]  (no reversing)
    max_yaw_rate = 90.0 * np.pi / 180.0    # [rad/s]
    max_accel = 1.0          # [m/s^2]
    max_delta_yaw_rate = 90.0 * np.pi / 180.0  # [rad/s^2]

    v_resolution = 0.05       # [m/s]
    yaw_rate_resolution = 5.0 * np.pi / 180.0  # [rad/s]
    dt = 0.1                  # [s] control / simulation step
    predict_time = 2.0        # [s] rollout horizon

    to_goal_cost_gain = 1.0
    speed_cost_gain = 1.0
    obstacle_cost_gain = 1.0

    robot_radius = 0.3        # [m] used for collision checking


def motion(pose, v, w, dt):
    x, y, theta = pose
    theta = theta + w * dt
    x = x + v * np.cos(theta) * dt
    y = y + v * np.sin(theta) * dt
    return np.array([x, y, theta])


def dynamic_window(v, w, config):
    # velocities reachable in one control step, intersected with hard limits
    v_min = max(config.min_speed, v - config.max_accel * config.dt)
    v_max = min(config.max_speed, v + config.max_accel * config.dt)
    w_min = max(-config.max_yaw_rate, w - config.max_delta_yaw_rate * config.dt)
    w_max = min(config.max_yaw_rate, w + config.max_delta_yaw_rate * config.dt)
    return v_min, v_max, w_min, w_max


def predict_trajectory(pose, v, w, config):
    traj = [pose]
    p = pose
    t = 0.0
    while t <= config.predict_time:
        p = motion(p, v, w, config.dt)
        traj.append(p)
        t += config.dt
    return np.array(traj)


def calc_obstacle_cost(traj, obstacle_points, config):
    if len(obstacle_points) == 0:
        return 0.0
    xy = traj[:, :2]
    # distance from every trajectory point to every obstacle point
    diff = xy[:, None, :] - obstacle_points[None, :, :]
    dist = np.hypot(diff[..., 0], diff[..., 1])
    min_dist = dist.min()
    if min_dist <= config.robot_radius:
        return np.inf  # collision -> reject this trajectory
    return 1.0 / min_dist


def calc_to_goal_cost(traj, goal):
    # Distance remaining to the goal, not just final heading - this is what
    # actually rewards routing *around* an obstacle instead of just slowing
    # to a stop while still facing it (a heading-only cost treats "stopped,
    # pointed at the goal" as free, which is a bad local minimum).
    dx = goal[0] - traj[-1, 0]
    dy = goal[1] - traj[-1, 1]
    return np.hypot(dx, dy)


def plan(pose, v, w, goal, obstacle_points, config):
    """Return (best_v, best_w, best_trajectory) for this control step."""
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
