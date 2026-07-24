# Real-Time LiDAR Path Planning (Dynamic Window Approach)

A small, self-contained demo of a mobile robot navigating toward a goal
through obstacles using a **simulated 2D LiDAR** and the **Dynamic Window
Approach (DWA)** — a classic real-time local planner used on real robots
(TurtleBot, ROS `move_base`, etc.).

There is no pre-computed global path. Every simulation step the robot:

1. Takes a 360° LiDAR scan (raycasting against the obstacles).
2. Feeds the detected hit points into the DWA planner, which searches a
   short window of feasible `(v, w)` velocity commands (linear + angular
   speed), simulates a ~2s rollout for each, and scores them on:
   - progress toward the goal
   - forward speed (prefer not crawling)
   - clearance from the nearest LiDAR-detected obstacle point
3. Executes only the first step of the best-scoring command, then repeats.

Because planning is redone from scratch every step using only the latest
scan, the robot reacts to obstacles the moment it "sees" them — this is
what makes it real-time / reactive rather than an offline path planner.

## Files

| File        | Purpose                                              |
|-------------|-------------------------------------------------------|
| `lidar.py`  | `Lidar` class — raycasting against circular obstacles |
| `dwa.py`    | Dynamic Window Approach planner                       |
| `main.py`   | World setup + matplotlib animation loop               |

## Run it

```bash
pip install -r requirements.txt
python main.py
```

A window opens showing:
- gray circles = obstacles
- red dots = current LiDAR hits
- cyan line = the DWA rollout the planner picked this step
- blue dotted line = the robot's traveled path
- green star = goal

## How the pieces work

**LiDAR simulation (`lidar.py`)** — casts `num_beams` rays evenly spaced
around the robot, each up to `max_range`. For each ray it solves the
ray-circle intersection against every obstacle analytically and keeps the
closest hit. Optional Gaussian noise can be added to mimic real sensor
error.

**DWA (`dwa.py`)** — at each step it:
1. Builds a "dynamic window": the range of `(v, w)` reachable in one
   timestep given the robot's current velocity and acceleration limits.
2. Samples that window on a grid and forward-simulates each candidate for
   `predict_time` seconds using a simple unicycle motion model.
3. Scores each rollout: trajectories that end pointing at the goal, keep
   speed up, and stay clear of LiDAR obstacle points score best.
   Trajectories that would collide (closest approach inside
   `robot_radius`) are rejected outright.
4. Returns the winning `(v, w)` — only its *first* step is executed before
   replanning, so the robot constantly re-reacts to new sensor data.

## Ideas to extend this

- Swap the hand-placed circular obstacles for an occupancy grid built live
  from LiDAR returns.
- Add a global planner (A*/RRT*) over a coarse map and use DWA only for
  local obstacle avoidance around the global path.
- Replace the simulated `Lidar` class with real scan data from a ROS
  `sensor_msgs/LaserScan` topic.
- Add moving obstacles to test dynamic replanning.
