"""Simulated 2D LiDAR sensor using raycasting against circular obstacles.

This module provides a simulated LiDAR sensor that casts rays around a robot
and detects intersections with circular obstacles. The sensor is designed for
use in navigation and path planning research.

Coordinate System:
    - Robot pose is (x, y, theta) in meters and radians
    - Theta = 0 points along the positive x-axis
    - Angles increase counterclockwise (standard math convention)
    - Distances are in meters

Units:
    - Distances: meters (m)
    - Angles: radians (rad)
    - Standard deviation: meters (m) for range noise

Example:
    >>> import numpy as np
    >>> from src.lidar import Lidar
    >>> lidar = Lidar(max_range=5.0, num_beams=90, noise_std=0.01)
    >>> pose = np.array([0.0, 0.0, 0.0])  # x, y, theta
    >>> obstacles = [(2.0, 2.0)]
    >>> ranges, points = lidar.scan(pose, obstacles, obstacle_radius=0.5)
"""
import numpy as np


class Lidar:
    """Simulated 2D LiDAR sensor using raycasting.
    
    Casts evenly-spaced rays around the robot and computes the closest
    intersection distance with circular obstacles using analytical
    ray-circle intersection.
    
    Attributes:
        max_range: Maximum sensing range in meters.
        num_beams: Number of evenly-spaced rays around 360 degrees.
        noise_std: Standard deviation of Gaussian noise on range measurements.
        angles: Array of beam angles relative to robot heading (radians).
    """
    
    def __init__(
        self,
        max_range: float = 6.0,
        num_beams: int = 72,
        noise_std: float = 0.0
    ) -> None:
        """Initialize the simulated LiDAR sensor.
        
        Args:
            max_range: Maximum sensing range in meters. Beams that don't hit
                an obstacle return this distance.
            num_beams: Number of evenly-spaced rays. More beams give higher
                angular resolution but increase computation.
            noise_std: Standard deviation of Gaussian noise added to range
                measurements. Set to 0 for deterministic behavior.
        """
        if max_range <= 0:
            raise ValueError("max_range must be positive")
        if num_beams <= 0:
            raise ValueError("num_beams must be positive")
        if noise_std < 0:
            raise ValueError("noise_std must be non-negative")
            
        self.max_range = max_range
        self.num_beams = num_beams
        self.noise_std = noise_std
        self.angles = np.linspace(-np.pi, np.pi, num_beams, endpoint=False)

    def scan(
        self,
        pose: np.ndarray,
        obstacles: list,
        obstacle_radius: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform a 360-degree LiDAR scan.
        
        Casts one ray per beam angle and computes the closest obstacle
        intersection distance using ray-circle intersection.
        
        Args:
            pose: Robot pose as array [x, y, theta] in meters and radians.
            obstacles: List of (x, y) tuples giving obstacle center positions.
            obstacle_radius: Radius of each circular obstacle in meters.
                All obstacles are assumed to have the same radius.
        
        Returns:
            A tuple (ranges, points) where:
            - ranges: Array of shape (num_beams,) with range distances.
            - points: Array of shape (num_beams, 2) with [x, y] hit points.
              Points on obstacles are exact; max-range points are at
              (x + max_range*cos(angle), y + max_range*sin(angle)).
        """
        x, y, theta = pose
        ranges = np.full(self.num_beams, self.max_range)
        points = np.zeros((self.num_beams, 2))

        for i, beam_offset in enumerate(self.angles):
            beam_angle = theta + beam_offset
            r = self._raycast(x, y, beam_angle, obstacles, obstacle_radius)
            if self.noise_std > 0:
                r = max(0.0, r + np.random.normal(0, self.noise_std))
            ranges[i] = r
            points[i] = (x + r * np.cos(beam_angle), y + r * np.sin(beam_angle))

        return ranges, points

    def _raycast(
        self,
        x: float,
        y: float,
        angle: float,
        obstacles: list,
        obstacle_radius: float
    ) -> float:
        """Compute closest ray-circle intersection distance.
        
        Uses analytical ray-circle intersection to find the distance to
        the nearest obstacle along the given direction.
        
        Args:
            x, y: Ray origin position in meters.
            angle: Ray direction in radians (world frame).
            obstacles: List of (ox, oy) obstacle center positions.
            obstacle_radius: Radius of each obstacle in meters.
        
        Returns:
            Distance to closest intersection, or max_range if no hit.
        """
        dx, dy = np.cos(angle), np.sin(angle)
        min_r = self.max_range

        for (ox, oy) in obstacles:
            fx, fy = x - ox, y - oy
            a = dx * dx + dy * dy
            b = 2 * (fx * dx + fy * dy)
            c = fx * fx + fy * fy - obstacle_radius ** 2
            disc = b * b - 4 * a * c
            if disc < 0:
                continue
            sqrt_disc = np.sqrt(disc)
            for t in ((-b - sqrt_disc) / (2 * a), (-b + sqrt_disc) / (2 * a)):
                if 0 <= t < min_r:
                    min_r = t

        return min_r

    def hits_only(
        self,
        ranges: np.ndarray,
        points: np.ndarray
    ) -> np.ndarray:
        """Filter out beams that didn't hit any obstacle.
        
        Args:
            ranges: Array of range measurements from scan().
            points: Array of hit points from scan().
        
        Returns:
            Array of shape (n_hits, 2) containing only the points
            where ranges are less than max_range (i.e., hit obstacles).
        """
        mask = ranges < self.max_range - 1e-6
        return points[mask]
