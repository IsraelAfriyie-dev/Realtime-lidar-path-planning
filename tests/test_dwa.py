"""Unit tests for the DWA path planning module."""
import numpy as np
import pytest
from src.planning import DWAConfig, plan, motion, dynamic_window, predict_trajectory
from src.planning.dwa import calc_obstacle_cost, calc_to_goal_cost


class TestDWAConfig:
    """Tests for DWA configuration."""

    def test_default_config_values(self):
        """Test DWAConfig has correct default values."""
        config = DWAConfig()
        assert config.max_speed == 1.0
        assert config.min_speed == 0.0
        assert abs(config.max_yaw_rate - np.pi / 2) < 0.01
        assert config.max_accel == 1.0
        assert config.dt == 0.1
        assert config.predict_time == 2.0
        assert config.robot_radius == 0.3

    def test_config_mutable(self):
        """Test that config values can be modified."""
        config = DWAConfig()
        config.max_speed = 2.0
        config.robot_radius = 0.5
        assert config.max_speed == 2.0
        assert config.robot_radius == 0.5


class TestMotion:
    """Tests for unicycle motion model."""

    def test_stationary_robot(self):
        """Test motion with zero velocity."""
        pose = np.array([1.0, 2.0, 0.5])
        new_pose = motion(pose, 0.0, 0.0, 0.1)
        
        # Position should not change
        np.testing.assert_array_almost_equal(new_pose[:2], [1.0, 2.0])
        # Theta should not change
        assert abs(new_pose[2] - 0.5) < 0.001

    def test_straight_line_motion(self):
        """Test straight line motion along x-axis."""
        pose = np.array([0.0, 0.0, 0.0])
        new_pose = motion(pose, 1.0, 0.0, 0.1)
        
        # Should move 0.1m in x direction
        np.testing.assert_array_almost_equal(new_pose[:2], [0.1, 0.0])
        assert abs(new_pose[2]) < 0.001

    def test_pure_rotation(self):
        """Test pure rotational motion."""
        pose = np.array([0.0, 0.0, 0.0])
        new_pose = motion(pose, 0.0, 1.0, 0.1)
        
        # Position should not change
        np.testing.assert_array_almost_equal(new_pose[:2], [0.0, 0.0])
        # Theta should increase by 0.1 rad
        np.testing.assert_almost_equal(new_pose[2], 0.1)

    def test_circular_motion(self):
        """Test circular motion (arc)."""
        pose = np.array([0.0, 0.0, 0.0])
        # Move with both linear and angular velocity
        new_pose = motion(pose, 1.0, 1.0, 0.1)
        
        # Should have moved and rotated
        assert new_pose[0] > 0  # Positive x
        assert new_pose[1] > 0  # Positive y (due to left turn)
        assert new_pose[2] > 0  # Positive theta


class TestDynamicWindow:
    """Tests for dynamic window computation."""

    def test_stationary_robot_window(self):
        """Test window for robot starting from rest."""
        config = DWAConfig()
        v_min, v_max, w_min, w_max = dynamic_window(0.0, 0.0, config)
        
        assert v_min == 0.0
        assert v_max == config.max_accel * config.dt
        assert w_min == -config.max_delta_yaw_rate * config.dt
        assert w_max == config.max_delta_yaw_rate * config.dt

    def test_moving_robot_window(self):
        """Test window for robot already moving."""
        config = DWAConfig()
        v_min, v_max, w_min, w_max = dynamic_window(0.5, 0.5, config)
        
        # Should be limited by acceleration
        assert v_min >= 0.0
        assert v_max <= config.max_speed
        assert w_min >= -config.max_yaw_rate
        assert w_max <= config.max_yaw_rate

    def test_window_respects_max_speed(self):
        """Test that window respects maximum speed."""
        config = DWAConfig()
        v_min, v_max, _, _ = dynamic_window(0.9, 0.0, config)
        
        # Cannot exceed max_speed even with positive acceleration
        assert v_max <= config.max_speed


class TestPredictTrajectory:
    """Tests for trajectory prediction."""

    def test_trajectory_shape(self):
        """Test predicted trajectory has correct shape."""
        config = DWAConfig()
        config.dt = 0.1
        config.predict_time = 1.0
        
        pose = np.array([0.0, 0.0, 0.0])
        traj = predict_trajectory(pose, 1.0, 0.0, config)
        
        # Should have 12 points: 0, 0.1, 0.2, ..., 1.0 (inclusive)
        assert traj.shape[0] == 12
        assert traj.shape[1] == 3

    def test_trajectory_straight_line(self):
        """Test straight line trajectory."""
        config = DWAConfig()
        config.dt = 0.1
        config.predict_time = 0.3
        
        pose = np.array([0.0, 0.0, 0.0])
        traj = predict_trajectory(pose, 1.0, 0.0, config)
        
        # x should increase linearly, y should stay constant
        for i in range(len(traj)):
            expected_x = i * 0.1
            assert abs(traj[i, 0] - expected_x) < 0.01
            assert abs(traj[i, 1]) < 0.01

    def test_trajectory_includes_start(self):
        """Test trajectory starts at initial pose."""
        config = DWAConfig()
        pose = np.array([5.0, 10.0, 1.57])
        traj = predict_trajectory(pose, 0.0, 0.0, config)
        
        np.testing.assert_array_almost_equal(traj[0], pose)


class TestCalcObstacleCost:
    """Tests for obstacle cost calculation."""

    def test_empty_obstacles(self):
        """Test cost with no obstacles."""
        config = DWAConfig()
        traj = np.array([[0, 0, 0], [1, 0, 0]])
        
        cost = calc_obstacle_cost(traj, np.array([]).reshape(0, 2), config)
        assert cost == 0.0

    def test_distant_obstacle(self):
        """Test cost with obstacle far away."""
        config = DWAConfig()
        traj = np.array([[0, 0, 0], [1, 0, 0]])
        obstacles = np.array([[10.0, 10.0]])  # Far away
        
        cost = calc_obstacle_cost(traj, obstacles, config)
        assert cost > 0  # Should be positive
        assert cost < 1  # Should be small (far away)

    def test_collision_cost_infinite(self):
        """Test collision gives infinite cost."""
        config = DWAConfig()
        config.robot_radius = 0.3
        traj = np.array([[1.0, 0.0, 0]])  # Near obstacle
        obstacles = np.array([[1.0, 0.2]])  # Within robot radius
        
        cost = calc_obstacle_cost(traj, obstacles, config)
        assert np.isinf(cost)

    def test_cost_inversely_proportional_to_distance(self):
        """Test cost decreases as distance increases."""
        config = DWAConfig()
        traj = np.array([[0, 0, 0]])
        obstacles_far = np.array([[5.0, 0.0]])
        obstacles_near = np.array([[1.0, 0.0]])
        
        cost_far = calc_obstacle_cost(traj, obstacles_far, config)
        cost_near = calc_obstacle_cost(traj, obstacles_near, config)
        
        assert cost_far < cost_near


class TestCalcToGoalCost:
    """Tests for goal cost calculation."""

    def test_at_goal(self):
        """Test cost when at goal."""
        traj = np.array([[5.0, 5.0, 0]])
        goal = np.array([5.0, 5.0])
        
        cost = calc_to_goal_cost(traj, goal)
        assert abs(cost) < 0.01

    def test_distance_to_goal(self):
        """Test cost equals Euclidean distance."""
        traj = np.array([[3.0, 4.0, 0]])  # 3-4-5 triangle
        goal = np.array([0.0, 0.0])
        
        cost = calc_to_goal_cost(traj, goal)
        assert abs(cost - 5.0) < 0.01


class TestPlan:
    """Tests for main DWA planning function."""

    def test_plan_returns_velocity(self):
        """Test plan returns valid velocities."""
        config = DWAConfig()
        pose = np.array([0.0, 0.0, 0.0])
        goal = np.array([5.0, 0.0])
        obstacles = np.array([[3.0, 3.0]])
        
        v, w, traj = plan(pose, 0.0, 0.0, goal, obstacles, config)
        
        assert 0 <= v <= config.max_speed
        assert -config.max_yaw_rate <= w <= config.max_yaw_rate

    def test_plan_returns_trajectory(self):
        """Test plan returns valid trajectory."""
        config = DWAConfig()
        pose = np.array([0.0, 0.0, 0.0])
        goal = np.array([5.0, 0.0])
        obstacles = np.array([[3.0, 3.0]])
        
        v, w, traj = plan(pose, 0.0, 0.0, goal, obstacles, config)
        
        assert traj.shape[1] == 3  # x, y, theta
        assert len(traj) > 1  # Should have multiple points

    def test_plan_with_no_obstacles(self):
        """Test plan works with no obstacles."""
        config = DWAConfig()
        pose = np.array([0.0, 0.0, 0.0])
        goal = np.array([5.0, 0.0])
        obstacles = np.array([]).reshape(0, 2)
        
        v, w, traj = plan(pose, 0.0, 0.0, goal, obstacles, config)
        
        # Should return positive velocity toward goal
        assert v > 0

    def test_plan_collision_avoidance(self):
        """Test planner avoids obstacles."""
        config = DWAConfig()
        pose = np.array([0.0, 0.0, 0.0])
        goal = np.array([5.0, 0.0])
        # Obstacle directly in path
        obstacles = np.array([[2.5, 0.0]])
        
        v, w, traj = plan(pose, 0.0, 0.0, goal, obstacles, config)
        
        # Should not have collision
        for point in traj[:, :2]:
            dist = np.linalg.norm(point - obstacles[0])
            assert dist > config.robot_radius

    def test_plan_from_different_directions(self):
        """Test planner navigates from different starting directions."""
        config = DWAConfig()
        goal = np.array([5.0, 5.0])
        obstacles = np.array([[2.5, 2.5]])
        
        # From each quadrant
        poses = [
            np.array([0.0, 0.0, 0.0]),     # Bottom-left
            np.array([10.0, 0.0, np.pi]),   # Bottom-right
            np.array([0.0, 10.0, -np.pi/2]),  # Top-left
        ]
        
        for pose in poses:
            v, w, traj = plan(pose, 0.0, 0.0, goal, obstacles, config)
            assert 0 <= v <= config.max_speed
