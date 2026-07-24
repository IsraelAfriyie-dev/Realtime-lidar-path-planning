"""Integration tests for the complete path planning system."""
import numpy as np
import pytest
from src.lidar import Lidar
from src.planning import DWAConfig, plan


class TestIntegrationScanAndPlan:
    """Tests for complete scan-to-plan workflow."""

    def test_full_workflow_single_step(self):
        """Test one complete step: scan -> plan -> execute."""
        # Setup
        lidar = Lidar(max_range=5.0, num_beams=90)
        config = DWAConfig()
        obstacles = [(3.0, 3.0), (5.0, 2.0)]
        obstacle_radius = 0.5
        pose = np.array([0.0, 0.0, np.pi / 4])
        goal = np.array([10.0, 10.0])
        v, w = 0.0, 0.0

        # Execute workflow
        ranges, points = lidar.scan(pose, obstacles, obstacle_radius)
        obstacle_points = lidar.hits_only(ranges, points)
        v_new, w_new, traj = plan(pose, v, w, goal, obstacle_points, config)
        
        # Verify outputs are valid
        assert len(ranges) == 90
        assert v_new >= 0  # Should not reverse
        assert traj.shape[1] == 3

    def test_repeated_steps_maintain_state(self):
        """Test multiple steps maintain velocity continuity."""
        lidar = Lidar(max_range=5.0, num_beams=90)
        config = DWAConfig()
        obstacles = [(3.0, 3.0)]
        obstacle_radius = 0.5
        pose = np.array([0.0, 0.0, 0.0])
        goal = np.array([5.0, 0.0])
        v, w = 0.5, 0.0  # Start with some velocity

        velocities = []
        for _ in range(10):
            ranges, points = lidar.scan(pose, obstacles, obstacle_radius)
            obstacle_points = lidar.hits_only(ranges, points)
            v, w, traj = plan(pose, v, w, goal, obstacle_points, config)
            velocities.append((v, w))
            pose = traj[1]  # Execute one step

        # Velocities should be consistent (within acceleration limits)
        for i in range(1, len(velocities)):
            v_prev, _ = velocities[i-1]
            v_curr, _ = velocities[i]
            # Should not jump more than max_accel * dt
            assert abs(v_curr - v_prev) <= config.max_accel * config.dt + 0.01

    def test_navigation_to_goal(self):
        """Test robot can navigate to goal with obstacles."""
        lidar = Lidar(max_range=5.0, num_beams=90)
        config = DWAConfig()
        # Simple obstacle layout
        obstacles = [(2.5, 0.0)]  # Obstacle directly in path
        obstacle_radius = 0.5
        pose = np.array([0.0, 0.0, 0.0])
        goal = np.array([5.0, 0.0])
        v, w = 0.0, 0.0

        goal_reached = False
        steps = 0
        max_steps = 200

        while not goal_reached and steps < max_steps:
            ranges, points = lidar.scan(pose, obstacles, obstacle_radius)
            obstacle_points = lidar.hits_only(ranges, points)
            v, w, traj = plan(pose, v, w, goal, obstacle_points, config)
            pose = traj[1]
            steps += 1

            # Check if goal reached
            distance_to_goal = np.linalg.norm(goal - pose[:2])
            if distance_to_goal < config.robot_radius + 0.5:
                goal_reached = True

        assert goal_reached, f"Goal not reached after {steps} steps"
        assert steps < max_steps


class TestCoordinateTransformations:
    """Tests for coordinate system correctness."""

    def test_lidar_beam_angles(self):
        """Test LiDAR beam angles are evenly spaced."""
        lidar = Lidar(num_beams=72)
        
        # Angles should span 2*pi and be evenly spaced
        angle_diff = np.diff(lidar.angles)
        expected_diff = 2 * np.pi / 72
        
        np.testing.assert_almost_equal(angle_diff, [expected_diff] * 71, decimal=10)

    def test_world_frame_angles(self):
        """Test beam angles are correctly transformed to world frame."""
        lidar = Lidar(num_beams=4)
        # Robot facing 0 radians (positive x)
        # With 4 beams starting at -π: [-π, -π/2, 0, π/2]
        pose = np.array([0.0, 0.0, 0.0])
        obstacles = [(1.0, 0.0)]  # Obstacle directly ahead
        obstacle_radius = 0.1

        ranges, points = lidar.scan(pose, obstacles, obstacle_radius)
        
        # The beam at world angle 0 is at index 2
        assert ranges[2] < ranges[0]  # Ahead vs behind
        assert ranges[2] < ranges[3]  # Ahead vs right

    def test_robot_rotation_changes_beam_direction(self):
        """Test that rotating robot changes which beam sees obstacle."""
        lidar = Lidar(num_beams=4)
        obstacles = [(0.0, 1.0)]  # Obstacle at +y
        obstacle_radius = 0.1

        # Robot facing +x (obstacle to the left)
        pose_x = np.array([0.0, 0.0, 0.0])
        ranges_x, _ = lidar.scan(pose_x, obstacles, obstacle_radius)

        # Robot facing +y (obstacle ahead)
        pose_y = np.array([0.0, 0.0, np.pi / 2])
        ranges_y, _ = lidar.scan(pose_y, obstacles, obstacle_radius)

        # Different beams should detect the obstacle
        min_beam_x = np.argmin(ranges_x)
        min_beam_y = np.argmin(ranges_y)
        assert min_beam_x != min_beam_y


class TestCollisionChecking:
    """Tests for collision detection in planning."""

    def test_no_collision_within_obstacle_clearance(self):
        """Test planner maintains safe distance from obstacles."""
        lidar = Lidar(max_range=5.0, num_beams=90)
        config = DWAConfig()
        config.robot_radius = 0.3
        
        # Place obstacle
        obstacles = [(2.0, 0.0)]
        obstacle_radius = 0.5
        
        pose = np.array([0.0, 0.0, 0.0])
        goal = np.array([5.0, 0.0])
        v, w = 0.0, 0.0

        # Run several steps
        for _ in range(50):
            ranges, points = lidar.scan(pose, obstacles, obstacle_radius)
            obstacle_points = lidar.hits_only(ranges, points)
            v, w, traj = plan(pose, v, w, goal, obstacle_points, config)
            pose = traj[1]
            
            # Check no collision
            for obs in obstacles:
                dist = np.linalg.norm(pose[:2] - np.array(obs))
                assert dist > obstacle_radius + config.robot_radius - 0.1  # Small tolerance

    def test_planner_stops_for_imminent_collision(self):
        """Test robot slows/stops when collision is imminent."""
        lidar = Lidar(max_range=3.0, num_beams=90)
        config = DWAConfig()
        
        # Obstacle very close, directly ahead
        obstacles = [(1.0, 0.0)]
        obstacle_radius = 0.5
        
        pose = np.array([0.0, 0.0, 0.0])
        goal = np.array([10.0, 0.0])  # Goal behind obstacle
        v, w = 0.5, 0.0

        ranges, points = lidar.scan(pose, obstacles, obstacle_radius)
        obstacle_points = lidar.hits_only(ranges, points)
        v_new, w_new, traj = plan(pose, v, w, goal, obstacle_points, config)
        
        # Should reduce speed or turn
        assert v_new < v or abs(w_new) > 0.1


class TestSystemParameters:
    """Tests for system behavior under different parameters."""

    def test_higher_resolution_slower(self):
        """Test that finer resolution takes more computation."""
        import time
        
        config_low = DWAConfig()
        config_low.v_resolution = 0.1
        config_low.yaw_rate_resolution = 0.2
        
        config_high = DWAConfig()
        config_high.v_resolution = 0.01
        config_high.yaw_rate_resolution = 0.02
        
        pose = np.array([0.0, 0.0, 0.0])
        goal = np.array([5.0, 0.0])
        obstacles = np.array([[3.0, 0.0]])
        
        # Time low resolution
        start = time.time()
        for _ in range(10):
            plan(pose, 0.0, 0.0, goal, obstacles, config_low)
        time_low = time.time() - start
        
        # Time high resolution
        start = time.time()
        for _ in range(10):
            plan(pose, 0.0, 0.0, goal, obstacles, config_high)
        time_high = time.time() - start
        
        # High resolution should take longer
        assert time_high > time_low

    def test_shorter_predict_time_faster(self):
        """Test that shorter prediction horizon is faster."""
        import time
        
        config_short = DWAConfig()
        config_short.predict_time = 0.5
        
        config_long = DWAConfig()
        config_long.predict_time = 2.0
        
        pose = np.array([0.0, 0.0, 0.0])
        goal = np.array([5.0, 0.0])
        obstacles = np.array([[3.0, 0.0]])
        
        # Time short prediction
        start = time.time()
        for _ in range(20):
            plan(pose, 0.0, 0.0, goal, obstacles, config_short)
        time_short = time.time() - start
        
        # Time long prediction
        start = time.time()
        for _ in range(20):
            plan(pose, 0.0, 0.0, goal, obstacles, config_long)
        time_long = time.time() - start
        
        # Short prediction should be faster
        assert time_short < time_long
