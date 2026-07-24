"""Unit tests for the LiDAR sensor module."""
import numpy as np
import pytest
from src.lidar import Lidar


class TestLidarInitialization:
    """Tests for Lidar class initialization."""

    def test_default_initialization(self):
        """Test Lidar with default parameters."""
        lidar = Lidar()
        assert lidar.max_range == 6.0
        assert lidar.num_beams == 72
        assert lidar.noise_std == 0.0
        assert len(lidar.angles) == 72

    def test_custom_initialization(self):
        """Test Lidar with custom parameters."""
        lidar = Lidar(max_range=10.0, num_beams=180, noise_std=0.05)
        assert lidar.max_range == 10.0
        assert lidar.num_beams == 180
        assert lidar.noise_std == 0.05
        assert len(lidar.angles) == 180

    def test_invalid_max_range(self):
        """Test that invalid max_range raises ValueError."""
        with pytest.raises(ValueError, match="max_range must be positive"):
            Lidar(max_range=0)
        with pytest.raises(ValueError, match="max_range must be positive"):
            Lidar(max_range=-1)

    def test_invalid_num_beams(self):
        """Test that invalid num_beams raises ValueError."""
        with pytest.raises(ValueError, match="num_beams must be positive"):
            Lidar(num_beams=0)
        with pytest.raises(ValueError, match="num_beams must be positive"):
            Lidar(num_beams=-10)

    def test_invalid_noise_std(self):
        """Test that invalid noise_std raises ValueError."""
        with pytest.raises(ValueError, match="noise_std must be non-negative"):
            Lidar(noise_std=-0.1)


class TestLidarScan:
    """Tests for LiDAR scan functionality."""

    def test_scan_returns_correct_shapes(self):
        """Test that scan returns arrays with correct shapes."""
        lidar = Lidar(num_beams=90)
        pose = np.array([0.0, 0.0, 0.0])
        obstacles = [(2.0, 0.0)]
        obstacle_radius = 0.5

        ranges, points = lidar.scan(pose, obstacles, obstacle_radius)

        assert ranges.shape == (90,)
        assert points.shape == (90, 2)

    def test_scan_with_no_obstacles(self):
        """Test scan when no obstacles are present."""
        lidar = Lidar(max_range=5.0, num_beams=8)
        pose = np.array([0.0, 0.0, 0.0])
        obstacles = []
        obstacle_radius = 0.5

        ranges, points = lidar.scan(pose, obstacles, obstacle_radius)

        # All ranges should be max_range when no obstacles
        np.testing.assert_array_almost_equal(ranges, [5.0] * 8)

    def test_scan_with_obstacle_in_front(self):
        """Test scan detects obstacle directly in front."""
        lidar = Lidar(max_range=10.0, num_beams=4)
        pose = np.array([0.0, 0.0, 0.0])  # Facing +x direction
        obstacles = [(5.0, 0.0)]  # Obstacle 5m ahead
        obstacle_radius = 0.5

        ranges, points = lidar.scan(pose, obstacles, obstacle_radius)

        # With 4 beams starting at -π, endpoint=False:
        # angles = [-π, -π/2, 0, π/2]
        # So beam at index 2 (angle=0) points directly ahead
        assert ranges[2] < 5.0  # Should detect obstacle
        assert ranges[2] > 4.0  # But not too close

    def test_scan_robot_inside_obstacle(self):
        """Test scan when robot is inside an obstacle."""
        lidar = Lidar(max_range=5.0, num_beams=8)
        pose = np.array([0.0, 0.0, 0.0])
        obstacles = [(0.0, 0.0)]  # Obstacle at robot position
        obstacle_radius = 1.0

        ranges, points = lidar.scan(pose, obstacles, obstacle_radius)

        # All beams should detect obstacle immediately
        assert np.all(ranges < lidar.max_range)

    def test_scan_angle_rotation(self):
        """Test that scan angle rotates with robot heading."""
        lidar = Lidar(max_range=10.0, num_beams=4)
        obstacles = [(5.0, 0.0)]  # Obstacle at +x
        obstacle_radius = 0.5

        # Robot facing +x direction
        # With 4 beams starting at -π: [-π, -π/2, 0, π/2]
        # Obstacle at +x (angle 0) is at beam index 2
        pose_x = np.array([0.0, 0.0, 0.0])
        ranges_x, _ = lidar.scan(pose_x, obstacles, obstacle_radius)

        # Robot facing +y direction (angle π/2)
        # Now obstacle at +x is at beam index 1 (π/2 - π/2 = 0)
        pose_y = np.array([0.0, 0.0, np.pi / 2])
        ranges_y, _ = lidar.scan(pose_y, obstacles, obstacle_radius)

        # The obstacle is detected at different beam indices
        assert ranges_x[2] < ranges_x[0]  # Obstacle ahead (index 2) vs behind (index 0)
        assert ranges_y[1] < ranges_y[0]  # Obstacle at different index due to rotation

    def test_scan_with_multiple_obstacles(self):
        """Test scan with multiple obstacles."""
        lidar = Lidar(max_range=10.0, num_beams=8)
        pose = np.array([0.0, 0.0, 0.0])
        obstacles = [(3.0, 0.0), (0.0, 4.0), (5.0, 5.0)]
        obstacle_radius = 0.5

        ranges, points = lidar.scan(pose, obstacles, obstacle_radius)

        # At least some beams should detect obstacles
        assert np.any(ranges < lidar.max_range)


class TestLidarHitsOnly:
    """Tests for hits_only filtering."""

    def test_hits_only_filters_max_range(self):
        """Test that hits_only removes max-range points."""
        lidar = Lidar(max_range=5.0, num_beams=4)
        pose = np.array([0.0, 0.0, 0.0])
        obstacles = [(2.0, 0.0)]
        obstacle_radius = 0.5

        ranges, points = lidar.scan(pose, obstacles, obstacle_radius)
        hits = lidar.hits_only(ranges, points)

        # Only the beam that hit should be in hits
        assert len(hits) == 1
        assert hits[0][0] > 1.0  # x should be positive (obstacle ahead)

    def test_hits_only_with_no_hits(self):
        """Test hits_only when no obstacles detected."""
        lidar = Lidar(max_range=5.0, num_beams=4)
        ranges = np.array([5.0, 5.0, 5.0, 5.0])
        points = np.array([[5, 0], [0, 5], [-5, 0], [0, -5]], dtype=float)

        hits = lidar.hits_only(ranges, points)

        assert len(hits) == 0


class TestLidarRaycast:
    """Tests for internal raycast functionality."""

    def test_raycast_exact_hit(self):
        """Test raycast computes exact intersection."""
        lidar = Lidar(max_range=10.0)
        obstacle = [(5.0, 0.0)]
        obstacle_radius = 0.5

        # Distance should be 4.5 (5.0 - 0.5)
        distance = lidar._raycast(0, 0, 0, obstacle, obstacle_radius)
        assert abs(distance - 4.5) < 0.01

    def test_raycast_no_hit(self):
        """Test raycast returns max_range when no hit."""
        lidar = Lidar(max_range=10.0)
        obstacle = [(100.0, 100.0)]  # Far away
        obstacle_radius = 0.5

        distance = lidar._raycast(0, 0, 0, obstacle, obstacle_radius)
        assert distance == 10.0

    def test_raycast_closest_obstacle(self):
        """Test raycast returns closest hit among multiple obstacles."""
        lidar = Lidar(max_range=10.0)
        obstacles = [(8.0, 0.0), (3.0, 0.0)]  # Near and far
        obstacle_radius = 0.5

        distance = lidar._raycast(0, 0, 0, obstacles, obstacle_radius)
        # Should hit the closer one at 2.5m
        assert abs(distance - 2.5) < 0.01

    def test_raycast_oblique_angle(self):
        """Test raycast at oblique angle."""
        lidar = Lidar(max_range=10.0)
        # Obstacle center at (3, 3), robot at origin, facing 45 degrees
        obstacles = [(3.0 / np.sqrt(2), 3.0 / np.sqrt(2))]
        obstacle_radius = 0.5

        distance = lidar._raycast(0, 0, np.pi / 4, obstacles, obstacle_radius)
        # Should hit approximately 2.5m away
        assert abs(distance - 2.5) < 0.1
