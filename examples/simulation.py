"""Real-time LiDAR-based path planning simulation.

A robot with a simulated 360-degree LiDAR navigates toward a goal through a
field of obstacles. Every animation frame it: takes a LiDAR scan, feeds the
detected points into a Dynamic Window Approach (DWA) planner, and executes
the best resulting velocity command. There is no pre-computed global path -
the robot only ever reacts to what it currently "sees", which is what makes
this real-time rather than an offline planner.

Run:
    python -m examples.simulation

Or from the repository root:
    PYTHONPATH=. python examples/simulation.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from src.lidar import Lidar
from src.planning import DWAConfig, plan

# ---------------------------------------------------------------------------
# World setup
# ---------------------------------------------------------------------------
OBSTACLES = [
    (2.0, 2.5), (4.0, 4.5), (6.0, 3.0), (5.0, 6.5),
    (8.0, 5.0), (3.0, 7.5), (7.5, 8.0), (1.5, 5.5),
]
OBSTACLE_RADIUS = 0.5

START_POSE = np.array([0.0, 0.0, np.pi / 4])  # x, y, theta
GOAL = np.array([10, 10])
GOAL_TOLERANCE = 0.4

config = DWAConfig()
lidar = Lidar(max_range=5.0, num_beams=90, noise_std=0.01)

# ---------------------------------------------------------------------------
# Simulation state
# ---------------------------------------------------------------------------
pose = START_POSE.copy()
v, w = 0.0, 0.0
trail = [pose[:2].copy()]

# ---------------------------------------------------------------------------
# Plot setup
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(-1, 11)
ax.set_ylim(-1, 11)
ax.set_aspect("equal")
ax.set_title("Real-time LiDAR path planning (DWA)")

for (ox, oy) in OBSTACLES:
    ax.add_patch(plt.Circle((ox, oy), OBSTACLE_RADIUS, color="dimgray"))

ax.plot(*GOAL, "g*", markersize=18, label="goal")

lidar_scatter = ax.scatter([], [], s=8, c="red", alpha=0.6, label="LiDAR hits")
robot_dot, = ax.plot([], [], "bo", markersize=10, label="robot")
heading_line, = ax.plot([], [], "b-", linewidth=2)
traj_line, = ax.plot([], [], "c-", linewidth=1, label="DWA rollout")
trail_line, = ax.plot([], [], "b:", linewidth=1, alpha=0.5, label="path so far")
ax.legend(loc="upper left", fontsize=8)


def step():
    """Advance the simulation by one control step. Returns False when done."""
    global pose, v, w

    ranges, points = lidar.scan(pose, OBSTACLES, OBSTACLE_RADIUS)
    obstacle_points = lidar.hits_only(ranges, points)

    v, w, best_traj = plan(pose, v, w, GOAL, obstacle_points, config)
    pose = best_traj[1]  # execute one dt of the chosen command
    trail.append(pose[:2].copy())

    return obstacle_points, best_traj


def update(_frame):
    if np.hypot(GOAL[0] - pose[0], GOAL[1] - pose[1]) < GOAL_TOLERANCE:
        ani.event_source.stop()
        ax.set_title("Goal reached!")
        return lidar_scatter, robot_dot, heading_line, traj_line, trail_line

    obstacle_points, best_traj = step()

    if len(obstacle_points) > 0:
        lidar_scatter.set_offsets(obstacle_points)
    else:
        lidar_scatter.set_offsets(np.empty((0, 2)))

    robot_dot.set_data([pose[0]], [pose[1]])
    heading_line.set_data(
        [pose[0], pose[0] + 0.6 * np.cos(pose[2])],
        [pose[1], pose[1] + 0.6 * np.sin(pose[2])],
    )
    traj_line.set_data(best_traj[:, 0], best_traj[:, 1])

    trail_arr = np.array(trail)
    trail_line.set_data(trail_arr[:, 0], trail_arr[:, 1])

    return lidar_scatter, robot_dot, heading_line, traj_line, trail_line


ani = FuncAnimation(fig, update, interval=config.dt * 1000, blit=False)

if __name__ == "__main__":
    plt.show()
