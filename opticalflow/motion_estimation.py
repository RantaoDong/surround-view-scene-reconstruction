from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class MotionConfig:
    max_corners: int = 20000
    quality_level: float = 0.02
    minimum_distance: float = 10.0
    window_size: int = 13
    pyramid_levels: int = 10
    forward_backward_check: bool = False
    forward_backward_threshold: float = 1.5
    minimum_tracked_points: int = 8
    ransac_reprojection_threshold: float = 3.0
    horizontal_margin_ratio: float = 80 / 600
    vertical_margin_ratio: float = 10 / 600
    center_exclusion_width_ratio: float = 160 / 600
    estimator: str = "lmeds"


@dataclass(frozen=True)
class MotionEstimate:
    matrix: Optional[np.ndarray]
    tracked_count: int
    inlier_count: int

    @property
    def succeeded(self):
        return self.matrix is not None


class OpticalFlowMotionEstimator:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def to_grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def get_region_mask(self, points, image_shape):
        height, width = image_shape[:2]
        horizontal_margin = width * self.config.horizontal_margin_ratio
        vertical_margin = height * self.config.vertical_margin_ratio
        center_half_width = (
            width * self.config.center_exclusion_width_ratio / 2
        )
        center_left = width / 2 - center_half_width
        center_right = width / 2 + center_half_width
        x_coordinates = points[:, 0]
        y_coordinates = points[:, 1]
        return (
            (x_coordinates >= horizontal_margin)
            & (x_coordinates <= width - horizontal_margin)
            & (y_coordinates >= vertical_margin)
            & (y_coordinates <= height - vertical_margin)
            & ~(
                (x_coordinates > center_left)
                & (x_coordinates < center_right)
            )
        )

    def detect_features(self, grayscale_image):
        return cv2.goodFeaturesToTrack(
            grayscale_image,
            mask=None,
            maxCorners=self.config.max_corners,
            qualityLevel=self.config.quality_level,
            minDistance=self.config.minimum_distance,
        )

    def track_features(self, previous_gray, current_gray, previous_points):
        lk_parameters = {
            "winSize": (self.config.window_size, self.config.window_size),
            "maxLevel": self.config.pyramid_levels,
        }
        current_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            current_gray,
            previous_points,
            None,
            **lk_parameters,
        )
        if current_points is None or forward_status is None:
            return np.empty((0, 2)), np.empty((0, 2))

        previous_flat = previous_points.reshape(-1, 2)
        current_flat = current_points.reshape(-1, 2)
        valid = (
            forward_status.reshape(-1).astype(bool)
            & np.isfinite(current_flat).all(axis=1)
            & self.get_region_mask(previous_flat, previous_gray.shape)
        )

        if self.config.forward_backward_check:
            returned_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
                current_gray,
                previous_gray,
                current_points,
                None,
                **lk_parameters,
            )
            if returned_points is None or backward_status is None:
                return np.empty((0, 2)), np.empty((0, 2))
            returned_flat = returned_points.reshape(-1, 2)
            tracking_error = np.linalg.norm(
                previous_flat - returned_flat,
                axis=1,
            )
            valid &= (
                backward_status.reshape(-1).astype(bool)
                & (tracking_error <= self.config.forward_backward_threshold)
            )
        return previous_flat[valid], current_flat[valid]

    def estimate(self, previous_image, current_image):
        previous_gray = self.to_grayscale(previous_image)
        current_gray = self.to_grayscale(current_image)
        previous_points = self.detect_features(previous_gray)
        if previous_points is None:
            return MotionEstimate(None, 0, 0)

        source_points, destination_points = self.track_features(
            previous_gray,
            current_gray,
            previous_points,
        )
        tracked_count = len(source_points)
        if tracked_count < self.config.minimum_tracked_points:
            return MotionEstimate(None, tracked_count, 0)

        method = cv2.RANSAC if self.config.estimator == "ransac" else cv2.LMEDS
        matrix, inlier_mask = cv2.estimateAffinePartial2D(
            source_points,
            destination_points,
            method=method,
            ransacReprojThreshold=self.config.ransac_reprojection_threshold,
        )
        if matrix is None or not np.isfinite(matrix).all():
            return MotionEstimate(None, tracked_count, 0)

        inlier_count = (
            int(np.count_nonzero(inlier_mask))
            if inlier_mask is not None
            else 0
        )
        if inlier_count < self.config.minimum_tracked_points:
            return MotionEstimate(None, tracked_count, inlier_count)
        return MotionEstimate(matrix, tracked_count, inlier_count)
