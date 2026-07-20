"""Simulated 2D LiDAR sensor using raycasting against circular obstacles."""
import numpy as np


class Lidar:
    def __init__(self, max_range=6.0, num_beams=72, noise_std=0.0):
        self.max_range = max_range
        self.num_beams = num_beams
        self.noise_std = noise_std
        self.angles = np.linspace(-np.pi, np.pi, num_beams, endpoint=False)

    def scan(self, pose, obstacles, obstacle_radius):
        """Cast one ray per beam angle and return (ranges, hit_points).

        pose: (x, y, theta) of the robot
        obstacles: list of (x, y) obstacle centers
        obstacle_radius: radius of each obstacle (all treated as circles)
        """
        x, y, theta = pose
        ranges = np.full(self.num_beams, self.max_range)
        points = np.zeros((self.num_beams, 2))

        for i, a in enumerate(self.angles):
            beam_angle = theta + a
            r = self._raycast(x, y, beam_angle, obstacles, obstacle_radius)
            if self.noise_std > 0:
                r = max(0.0, r + np.random.normal(0, self.noise_std))
            ranges[i] = r
            points[i] = (x + r * np.cos(beam_angle), y + r * np.sin(beam_angle))

        return ranges, points

    def _raycast(self, x, y, angle, obstacles, obstacle_radius):
        """Closest ray-circle intersection distance along one beam."""
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

    def hits_only(self, ranges, points):
        """Filter out beams that never hit anything (max range)."""
        mask = ranges < self.max_range - 1e-6
        return points[mask]
