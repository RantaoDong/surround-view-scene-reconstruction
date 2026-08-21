from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class InpaintingConfig:
    dark_threshold: int = 35
    fill_mode: str = "dark"
    center_width_ratio: float = 0.28
    center_height_ratio: float = 0.68


@dataclass(frozen=True)
class FrameResult:
    image: np.ndarray
    tracked_count: int
    inlier_count: int
    motion_found: bool
    filled_pixel_count: int


class BevInpainter:
    def __init__(self, motion_estimator, config):
        self.motion_estimator = motion_estimator
        self.config = config
        self.previous_original = None
        self.previous_result = None

    @staticmethod
    def to_grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def get_center_bounds(self, image_shape):
        height, width = image_shape[:2]
        center_width = round(width * self.config.center_width_ratio)
        center_height = round(height * self.config.center_height_ratio)
        left = max(0, (width - center_width) // 2)
        right = min(width, left + center_width)
        top = max(0, (height - center_height) // 2)
        bottom = min(height, top + center_height)
        return top, bottom, left, right

    def create_missing_mask(self, current_image):
        if self.config.fill_mode == "dark":
            return self.to_grayscale(current_image) <= self.config.dark_threshold

        missing = np.zeros(current_image.shape[:2], dtype=bool)
        top, bottom, left, right = self.get_center_bounds(current_image.shape)
        missing[top:bottom, left:right] = True
        return missing

    def fill_from_history(self, current_image, matrix):
        height, width = current_image.shape[:2]
        warped_history = cv2.warpAffine(
            self.previous_result,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        fill_mask = self.create_missing_mask(current_image)
        changed_mask = fill_mask & np.any(
            warped_history != current_image,
            axis=2,
        )
        result = current_image.copy()
        result[fill_mask] = warped_history[fill_mask]
        return result, int(np.count_nonzero(changed_mask))

    def initialize(self, image):
        self.previous_original = image.copy()
        self.previous_result = image.copy()
        return FrameResult(image.copy(), 0, 0, False, 0)

    def update_history(self, image, result):
        self.previous_original = image.copy()
        self.previous_result = result.copy()

    def process(self, current_image):
        if self.previous_original is None:
            return self.initialize(current_image)
        if self.previous_original.shape != current_image.shape:
            raise ValueError("All input images must have the same dimensions")

        motion = self.motion_estimator.estimate(
            self.previous_original,
            current_image,
        )
        if not motion.succeeded:
            result = current_image.copy()
            self.update_history(current_image, result)
            return FrameResult(
                result,
                motion.tracked_count,
                motion.inlier_count,
                False,
                0,
            )

        result, filled_pixel_count = self.fill_from_history(
            current_image,
            motion.matrix,
        )
        self.update_history(current_image, result)
        return FrameResult(
            result,
            motion.tracked_count,
            motion.inlier_count,
            True,
            filled_pixel_count,
        )
