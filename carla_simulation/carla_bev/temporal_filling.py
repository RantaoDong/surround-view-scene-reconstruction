import cv2
import numpy as np

from .bev_stitching import BevLayout


SEAM_LINE_ENDPOINTS = (
    ((143, 124), (915, 795), "left"),
    ((1900, 86), (1085, 795), "right"),
    ((1923, 1933), (1085, 1205), "right"),
    ((276, 1760), (879, 1236), "left"),
)


def point_is_away_from_seams(point, image_width):
    x_coordinate, y_coordinate = point
    for start, end, side in SEAM_LINE_ENDPOINTS:
        line_offset = (
            (x_coordinate - start[0]) / (end[0] - start[0])
            - (y_coordinate - start[1]) / (end[1] - start[1])
        )
        on_target_side = (
            x_coordinate < image_width / 2
            if side == "left"
            else x_coordinate > image_width / 2
        )
        if -0.1 < line_offset < 0.1 and on_target_side:
            return False
    return True


def compose_affine(second_matrix, first_matrix):
    combined = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    combined[0, 0] = (
        second_matrix[0, 0] * first_matrix[0, 0]
        + second_matrix[0, 1] * first_matrix[1, 0]
    )
    combined[0, 1] = (
        second_matrix[0, 0] * first_matrix[0, 1]
        + second_matrix[0, 1] * first_matrix[1, 1]
    )
    combined[0, 2] = (
        second_matrix[0, 0] * first_matrix[0, 2]
        + second_matrix[0, 1] * first_matrix[1, 2]
        + second_matrix[0, 2]
    )
    combined[1, 0] = (
        second_matrix[1, 0] * first_matrix[0, 0]
        + second_matrix[1, 1] * first_matrix[1, 0]
    )
    combined[1, 1] = (
        second_matrix[1, 0] * first_matrix[0, 1]
        + second_matrix[1, 1] * first_matrix[1, 1]
    )
    combined[1, 2] = (
        second_matrix[1, 0] * first_matrix[0, 2]
        + second_matrix[1, 1] * first_matrix[1, 2]
        + second_matrix[1, 2]
    )
    return combined


class FourFrameCenterFiller:
    def __init__(self, layout=None):
        self.layout = layout or BevLayout()
        self.previous_frames = []
        self.previous_transforms = []

    def _estimate_motion(self, previous_image, current_image):
        previous_gray = cv2.cvtColor(previous_image, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        feature_parameters = {
            "maxCorners": 1000,
            "qualityLevel": 0.01,
            "minDistance": 30,
        }
        optical_flow_parameters = {"winSize": (40, 40), "maxLevel": 8}
        previous_points = cv2.goodFeaturesToTrack(
            previous_gray,
            mask=None,
            **feature_parameters,
        )
        current_points, status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            current_gray,
            previous_points,
            None,
            **optical_flow_parameters,
        )
        tracked_previous = previous_points[status == 1]
        tracked_current = current_points[status == 1]
        valid_points = np.asarray(
            [
                point_is_away_from_seams(point, self.layout.width)
                for point in tracked_previous
            ],
            dtype=bool,
        )
        motion_matrix, _ = cv2.estimateAffine2D(
            tracked_previous[valid_points],
            tracked_current[valid_points],
        )
        return motion_matrix

    def _create_vehicle_mask(self, image_shape):
        mask = np.zeros(image_shape, dtype=np.uint8)
        top = (self.layout.height - self.layout.vehicle_height) // 2
        bottom = (self.layout.height + self.layout.vehicle_height) // 2
        left = (self.layout.width - self.layout.vehicle_width) // 2
        right = (self.layout.width + self.layout.vehicle_width) // 2
        mask[top:bottom, left:right] = 255
        return mask

    def process(self, current_image):
        if not self.previous_frames:
            self.previous_frames.append(current_image.copy())
            return current_image.copy()

        current_transform = self._estimate_motion(
            self.previous_frames[0],
            current_image,
        )
        accumulated_transform = current_transform
        for previous_transform in self.previous_transforms:
            accumulated_transform = compose_affine(
                previous_transform,
                accumulated_transform,
            )

        height, width = current_image.shape[:2]
        historical_image = self.previous_frames[-1]
        aligned_history = cv2.warpAffine(
            historical_image,
            accumulated_transform,
            (width, height),
            borderValue=0,
        )
        vehicle_mask = self._create_vehicle_mask(current_image.shape)
        center_fill = cv2.bitwise_and(aligned_history, vehicle_mask)
        filled_image = cv2.add(current_image, center_fill)

        self.previous_frames.insert(0, current_image.copy())
        self.previous_frames = self.previous_frames[:4]
        self.previous_transforms.insert(0, current_transform)
        self.previous_transforms = self.previous_transforms[:3]
        return filled_image

