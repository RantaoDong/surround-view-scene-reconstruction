import argparse
import os

import cv2
import numpy as np


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMERA_NAMES = ("left", "front", "right", "back")
CHESSBOARD_SIZE = (8, 5)
UNDISTORTED_DIR = os.path.join(PROJECT_DIR, "outputs", "undistorted")
DIAGNOSTICS_DIR = os.path.join(PROJECT_DIR, "outputs", "diagnostics")
REFERENCE_DIR = os.path.join(PROJECT_DIR, "sample_data", "reference")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Estimate bird's-eye-view homographies"
    )
    parser.add_argument(
        "-bw", "--BEV_WIDTH", "--bev-width", dest="bev_width",
        default=1000, type=int, help="Bird's-eye-view width"
    )
    parser.add_argument(
        "-bh", "--BEV_HEIGHT", "--bev-height", dest="bev_height",
        default=1200, type=int, help="Bird's-eye-view height"
    )
    parser.add_argument(
        "-cw", "--CAR_WIDTH", "--car-width", dest="car_width",
        default=280, type=int, help="Vehicle mask width"
    )
    parser.add_argument(
        "-ch", "--CAR_HEIGHT", "--car-height", dest="car_height",
        default=600, type=int, help="Vehicle mask height"
    )
    return parser.parse_args()


def read_image(path):
    encoded_image = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)


def write_image(path, image):
    extension = os.path.splitext(path)[1]
    success, encoded_image = cv2.imencode(extension, image)
    if not success:
        raise RuntimeError("Failed to encode image: {}".format(path))
    encoded_image.tofile(path)


class ViewMask:
    def __init__(self, camera_name, arguments):
        self.bev_width = arguments.bev_width
        self.bev_height = arguments.bev_height
        self.car_width = arguments.car_width
        self.car_height = arguments.car_height
        self.mask = self.create_mask(camera_name)

    def get_polygon(self, camera_name):
        half_left = (self.bev_width - self.car_width) / 2
        half_right = (self.bev_width + self.car_width) / 2
        half_front = (self.bev_height - self.car_height) / 2
        half_back = (self.bev_height + self.car_height) / 2

        polygons = {
            "front": [
                [0, 0],
                [self.bev_width, 0],
                [self.bev_width, half_front],
                [0, half_front],
            ],
            "back": [
                [0, self.bev_height],
                [self.bev_width, self.bev_height],
                [self.bev_width, half_back],
                [0, half_back],
            ],
            "left": [
                [0, 0],
                [0, self.bev_height],
                [half_left, self.bev_height],
                [half_left, 0],
            ],
            "right": [
                [self.bev_width, 0],
                [self.bev_width, self.bev_height],
                [half_right, self.bev_height],
                [half_right, 0],
            ],
        }
        try:
            return np.asarray(polygons[camera_name], dtype=np.int32)
        except KeyError as error:
            raise ValueError("Unknown camera name: {}".format(camera_name)) from error

    def create_mask(self, camera_name):
        mask = np.zeros((self.bev_height, self.bev_width), dtype=np.uint8)
        return cv2.fillPoly(mask, [self.get_polygon(camera_name)], 255)

    def apply(self, image):
        return cv2.bitwise_and(image, image, mask=self.mask)


def find_chessboard_corners(image):
    found, corners = cv2.findChessboardCorners(image, CHESSBOARD_SIZE, None)
    return found, corners


def save_corner_preview(camera_name, image_role, image, corners):
    preview = image.copy()
    cv2.drawChessboardCorners(preview, CHESSBOARD_SIZE, corners, True)
    output_path = os.path.join(
        DIAGNOSTICS_DIR,
        "{}_{}_corners.jpg".format(camera_name, image_role),
    )
    write_image(output_path, preview)


def estimate_homography(camera_name, arguments):
    source_path = os.path.join(
        UNDISTORTED_DIR,
        "undistorted_{}.jpg".format(camera_name),
    )
    target_path = os.path.join(REFERENCE_DIR, "birdseye_reference.jpg")
    source_image = read_image(source_path)
    target_image = read_image(target_path)
    if source_image is None:
        raise FileNotFoundError("Input image not found: {}".format(source_path))
    if target_image is None:
        raise FileNotFoundError("Target image not found: {}".format(target_path))

    masked_target = ViewMask(camera_name, arguments).apply(target_image)
    source_found, source_corners = find_chessboard_corners(source_image)
    target_found, target_corners = find_chessboard_corners(masked_target)
    if not source_found or not target_found:
        missing_roles = []
        if not source_found:
            missing_roles.append("source")
        if not target_found:
            missing_roles.append("target")
        print(
            "{}: chessboard not found in {} image".format(
                camera_name,
                " and ".join(missing_roles),
            )
        )
        return None

    # Corner diagnostics
    save_corner_preview(camera_name, "src", source_image, source_corners)
    save_corner_preview(camera_name, "dst", masked_target, target_corners)

    homography, _ = cv2.findHomography(
        source_corners,
        target_corners,
        method=cv2.RANSAC,
    )
    if homography is None:
        raise RuntimeError("Homography estimation failed: {}".format(camera_name))

    output_path = os.path.join(
        PROJECT_DIR,
        "config",
        camera_name,
        "camera_{}_H.npy".format(camera_name),
    )
    np.save(output_path, homography)
    print("{} homography:\n{}".format(camera_name, homography))
    return homography


def main():
    arguments = parse_arguments()
    os.makedirs(DIAGNOSTICS_DIR, exist_ok=True)
    for camera_name in CAMERA_NAMES:
        estimate_homography(camera_name, arguments)


if __name__ == "__main__":
    main()
