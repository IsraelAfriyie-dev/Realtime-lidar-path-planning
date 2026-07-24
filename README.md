# Real-Time LiDAR Path Planning

A self-contained Python project demonstrating real-time mobile robot navigation using a simulated 2D LiDAR sensor and the Dynamic Window Approach (DWA) local planner.

**Author:** Israel Afriyie  
**License:** MIT

## Overview

This project implements a robot that navigates autonomously toward a target position through a field of obstacles using only its onboard LiDAR sensor. Unlike global path planners that compute a complete route upfront, DWA re-plans at every step based on the latest sensor readings, enabling reactive collision avoidance in dynamic or partially-known environments.

## Problem Addressed

How can a mobile robot navigate safely through obstacles when it only has local sensing (LiDAR) and no prior map? The robot must:
1. Detect nearby obstacles in real-time
2. Generate smooth, collision-free motion commands
3. Reach the goal while avoiding obstacles

## Key Features

- **Simulated 2D LiDAR**: Raycasting against circular obstacles with configurable noise
- **Dynamic Window Approach**: Real-time local planner based on velocity space search
- **Reactive Navigation**: Re-plans at every control step using latest sensor data
- **Visualization**: Real-time animation showing LiDAR hits, planned trajectories, and robot path
- **Configurable**: All parameters (velocities, costs, sensor properties) easily adjustable

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Real-Time Control Loop                       │
│                                                                   │
│   ┌──────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│   │  LiDAR   │───▶│  Obstacle    │───▶│   DWA Planner        │  │
│   │  Sensor  │    │  Detection   │    │   (velocity search)  │  │
│   └──────────┘    └──────────────┘    └──────────────────────┘  │
│                                                    │              │
│                                                    ▼              │
│                                            ┌──────────────┐     │
│                                            │  Best (v,w)  │     │
│                                            └──────────────┘     │
│                                                    │              │
│                                                    ▼              │
│                                            ┌──────────────┐     │
│                                            │   Robot      │     │
│                                            │   Motion     │     │
│                                            └──────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

### Processing Pipeline

1. **LiDAR Scan**: Cast rays around robot, compute ray-circle intersections
2. **Point Cloud Processing**: Extract obstacle hit points from ranges
3. **Trajectory Prediction**: Simulate candidate velocity commands
4. **Cost Evaluation**: Score trajectories by goal progress, speed, and clearance
5. **Command Execution**: Apply best velocity, repeat at next timestep

## LiDAR Input

### Format
The simulated LiDAR produces:
- `ranges`: Array of distances `[r₁, r₂, ..., rₙ]` where `n = num_beams`
- `points`: Array of 2D hit coordinates `[[x₁,y₁], [x₂,y₂], ...]`

### Sensor Model
- **Type**: 2D raycasting (no vertical dimension)
- **Beams**: Evenly spaced around 360° (configurable count)
- **Range**: Maximum sensing distance (configurable)
- **Noise**: Optional Gaussian noise on range measurements

### Units
- Distances: meters (m)
- Angles: radians (rad)
- Update rate: determined by control loop (default `dt = 0.1s`)

## Point-Cloud Processing

The `Lidar.hits_only()` method filters raw scan data:
```python
ranges, points = lidar.scan(pose, obstacles, obstacle_radius)
obstacle_points = lidar.hits_only(ranges, points)  # Filter max-range beams
```

## Obstacle Detection

Obstacles are represented as circles with:
- Center position `(x, y)` in world coordinates
- Uniform radius `r` (all obstacles have same size)

The raycaster computes analytical ray-circle intersections to find the closest hit per beam.

## Environmental Mapping

This implementation uses **direct obstacle representation**:
- No grid-based map is built
- Obstacle points from LiDAR are used directly in planning
- Simple and efficient for small obstacle counts

For larger environments, consider:
- Occupancy grid maps
- Octomap for 3D
- Point cloud clustering

## Path Planning Algorithm

The Dynamic Window Approach (DWA) planner:

1. **Dynamic Window**: Compute velocity bounds reachable given current (v, w) and acceleration limits
2. **Trajectory Prediction**: For each (v, w) candidate, predict pose over `predict_time` seconds
3. **Cost Function**:
   - `to_goal_cost`: Euclidean distance from trajectory end to goal
   - `speed_cost`: Prefer higher forward velocity
   - `obstacle_cost`: Penalize proximity to obstacles
4. **Selection**: Choose (v, w) with lowest weighted sum cost
5. **Execution**: Apply only the first step, then re-plan

### Reference
Fox, D., Burgard, W., & Thrun, S. (1997). The dynamic window approach to collision avoidance. IEEE Robotics & Automation Magazine, 4(1), 23-33.

## Collision Checking

Collision checking uses distance between trajectory points and obstacle points:

```python
min_dist = min(||traj_point - obstacle_point|| for all points)
if min_dist <= robot_radius:
    reject trajectory  # Infinite cost
```

## Repository Structure

```
Realtime-lidar-path-planning/
├── README.md                 # This file
├── LICENSE                  # MIT License
├── requirements.txt         # Python dependencies
├── pyproject.toml          # Project metadata and build config
├── .gitignore              # Git ignore patterns
├── configs/
│   └── default.yaml         # Default configuration
├── src/
│   ├── __init__.py
│   ├── lidar/
│   │   ├── __init__.py
│   │   └── sensor.py        # LiDAR simulation
│   └── planning/
│       ├── __init__.py
│       └── dwa.py           # DWA planner
├── examples/
│   └── simulation.py        # Main simulation demo
├── tests/
│   ├── __init__.py
│   ├── test_lidar.py        # LiDAR unit tests
│   ├── test_dwa.py          # DWA unit tests
│   └── test_integration.py  # Integration tests
└── data/
    └── sample/              # Sample data (if any)
```

## Requirements

### Hardware
- Any system capable of running Python 3.8+

### Software
- Python 3.8 or higher
- NumPy >= 1.24
- Matplotlib >= 3.7

## Installation

### Option 1: Clone and install
```bash
git clone https://github.com/IsraelAfriyie-dev/Realtime-lidar-path-planning.git
cd Realtime-lidar-path-planning
pip install -r requirements.txt
```

### Option 2: Install as package
```bash
pip install -e .
```

## Running the Project

### Quick Start
```bash
# Run the visualization demo
python examples/simulation.py
```

### Using PYTHONPATH
```bash
PYTHONPATH=. python examples/simulation.py
```

### Module Import
```python
from src.lidar import Lidar
from src.planning import DWAConfig, plan
```

## Configuration

All parameters are configurable via `DWAConfig`:

```python
from src.planning import DWAConfig

config = DWAConfig()
config.max_speed = 1.5        # Increase max velocity
config.robot_radius = 0.4     # Larger robot
config.dt = 0.05              # Faster control loop
config.predict_time = 3.0     # Longer horizon
```

Or load from YAML:
```yaml
# configs/custom.yaml
planner:
  max_speed: 1.5
  robot_radius: 0.4
  dt: 0.05
lidar:
  max_range: 8.0
  num_beams: 180
```

## Expected Inputs and Outputs

### Inputs
- **Obstacles**: List of `(x, y, radius)` tuples or circles
- **Start pose**: `[x, y, theta]` in meters and radians
- **Goal**: `[x, y]` in meters

### Outputs
- **Velocity commands**: `(v, w)` linear and angular velocity
- **Trajectory preview**: Array of predicted poses for visualization
- **Animation**: Real-time plot showing robot, LiDAR hits, and planned path

## Visualization

The simulation displays:
- **Gray circles**: Obstacles
- **Red dots**: Current LiDAR hit points
- **Cyan line**: DWA rollout (planned trajectory)
- **Blue dotted line**: Robot's traveled path
- **Green star**: Goal position
- **Blue arrow**: Robot heading

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_lidar.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Categories
- **Unit tests**: Individual component testing (LiDAR, DWA)
- **Integration tests**: End-to-end workflow testing
- **Deterministic tests**: Coordinate transforms, cost calculations

## Known Limitations

1. **Simulated LiDAR only**: No interface for real sensor hardware
2. **Circular obstacles**: All obstacles must be circles of equal radius
3. **2D plane**: No elevation changes or 3D planning
4. **Local planning**: No global path planner; may get stuck in local minima
5. **No SLAM**: Environment must be provided as explicit obstacle list

## Troubleshooting

### Issue: "No module named 'src'"
```bash
export PYTHONPATH=.
```

### Issue: Animation not showing
Ensure you have a display environment:
```bash
# For headless servers
export MPLBACKEND=Agg
```

### Issue: Slow performance
Reduce resolution:
```python
config.v_resolution = 0.1      # Reduce velocity search density
config.yaw_rate_resolution = 0.2
config.predict_time = 1.0       # Shorter horizon
```

## Future Improvements

Potential enhancements for this project:

1. **Global planner integration**: Add A* or RRT* for global path, DWA for local avoidance
2. **Occupancy grid mapping**: Build and maintain environment map from LiDAR
3. **Real sensor interface**: Add support for ROS topics or direct sensor APIs
4. **Dynamic obstacles**: Moving obstacle detection and tracking
5. **Multi-robot**: Coordination between multiple agents
6. **3D expansion**: 3D LiDAR and 3D DWA for aerial vehicles

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Developed by Israel Afriyie*
