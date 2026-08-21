from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class BevLayout:
    width: int = 2000
    height: int = 2000
    vehicle_width: int = 170
    vehicle_height: int = 410


class CameraRegionMask:
    def __init__(self, camera_name, layout):
        self.layout = layout
        self.mask = self._create_mask(camera_name)

    def _create_mask(self, camera_name):
        width = self.layout.width
        height = self.layout.height
        vehicle_width = self.layout.vehicle_width
        vehicle_height = self.layout.vehicle_height
        left = (width - vehicle_width) / 2
        right = (width + vehicle_width) / 2
        top = (height - vehicle_height) / 2
        bottom = (height + vehicle_height) / 2

        polygons = {
            "front": [[0, 0], [width, 0], [right, top], [left, top]],
            "back": [
                [0, height],
                [width, height],
                [right, bottom],
                [left, bottom],
            ],
            "left": [[0, 0], [0, height], [left, bottom], [left, top]],
            "right": [
                [width, 0],
                [width, height],
                [right, bottom],
                [right, top],
            ],
        }
        if camera_name not in polygons:
            raise ValueError("Unknown camera name: {}".format(camera_name))
        mask = np.zeros((height, width), dtype=np.uint8)
        points = np.asarray(polygons[camera_name], dtype=np.int32)
        return cv2.fillPoly(mask, [points], 255)

    def apply(self, image):
        return cv2.bitwise_and(image, image, mask=self.mask)


class SurroundViewStitcher:
    def __init__(self, layout=None):
        self.layout = layout or BevLayout()
        self.masks = {
            name: CameraRegionMask(name, self.layout)
            for name in ("front", "back", "left", "right")
        }

    def stitch(self, front_image, back_image, left_image, right_image):
        output_size = (self.layout.width, self.layout.height)
        front_transform = np.float32([[1, 0, 0], [0, 1, -185]])
        back_transform = np.float32([[-1, 0, 2000], [0, -1, 2180]])
        left_transform = np.float32([[0, 1, -145], [-1, 0, 2000]])
        right_transform = np.float32([[0, -1, 2150], [1, 0, 0]])

        aligned_images = {
            "front": cv2.warpAffine(front_image, front_transform, output_size),
            "back": cv2.warpAffine(back_image, back_transform, output_size),
            "left": cv2.warpAffine(left_image, left_transform, output_size),
            "right": cv2.warpAffine(right_image, right_transform, output_size),
        }
        masked_images = {
            name: self.masks[name].apply(image)
            for name, image in aligned_images.items()
        }
        surround_view = cv2.add(masked_images["front"], masked_images["back"])
        surround_view = cv2.add(surround_view, masked_images["left"])
        return cv2.add(surround_view, masked_images["right"])

