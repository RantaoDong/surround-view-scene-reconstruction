import argparse
import json
import os

import cv2
import numpy as np

from exposure_compensation import (
    OverlapExposureCompensator,
    WholeViewExposureCompensator,
)


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CAMERA_NAMES = ("front", "left", "right", "back")
CAMERA_ADJACENCIES = (
    ("front", "left"),
    ("front", "right"),
    ("back", "left"),
    ("back", "right"),
)
RAW_DATA_DIR = os.path.join(PROJECT_DIR, "sample_data", "raw")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
UNMASKED_BEV_DIR = os.path.join(OUTPUT_DIR, "birdseye", "unmasked")
MASKED_BEV_DIR = os.path.join(OUTPUT_DIR, "birdseye", "masked")
BALANCED_BEV_DIR = os.path.join(OUTPUT_DIR, "birdseye", "balanced")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate a surround-view image")
    parser.add_argument(
        "-fw", "--FRAME_WIDTH", "--frame-width", dest="frame_width",
        default=1280, type=int, help="Camera frame width"
    )
    parser.add_argument(
        "-fh", "--FRAME_HEIGHT", "--frame-height", dest="frame_height",
        default=1024, type=int, help="Camera frame height"
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
    parser.add_argument(
        "-fs", "--FOCAL_SCALE", "--focal-scale", dest="focal_scale",
        default=1.2, type=float, help="Output focal scale"
    )
    parser.add_argument(
        "-ss", "--SIZE_SCALE", "--size-scale", dest="size_scale",
        default=1.0, type=float, help="Output size scale"
    )
    parser.add_argument(
        "--exposure-compensation",
        default="global",
        choices=("off", "global", "overlap"),
        help="Exposure compensation method",
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


class BirdseyeProjector:
    def __init__(self, camera_name, arguments):
        config_dir = os.path.join(PROJECT_DIR, "config", camera_name)
        filename_prefix = "camera_{}_".format(camera_name)
        self.camera_matrix = np.load(
            os.path.join(config_dir, filename_prefix + "K.npy")
        )
        self.distortion_coefficients = np.load(
            os.path.join(config_dir, filename_prefix + "D.npy")
        )
        self.homography = np.load(
            os.path.join(config_dir, filename_prefix + "H.npy")
        )
        self.bev_size = (arguments.bev_width, arguments.bev_height)

        output_camera_matrix = self.camera_matrix.copy()
        output_camera_matrix[0, 0] *= arguments.focal_scale
        output_camera_matrix[1, 1] *= arguments.focal_scale
        output_size = (
            int(arguments.frame_width * arguments.size_scale),
            int(arguments.frame_height * arguments.size_scale),
        )
        undistortion_maps = cv2.fisheye.initUndistortRectifyMap(
            self.camera_matrix,
            self.distortion_coefficients,
            np.eye(3),
            output_camera_matrix,
            output_size,
            cv2.CV_16SC2,
        )

        # Precomputed remap
        self.birdseye_maps = tuple(
            cv2.warpPerspective(remap, self.homography, self.bev_size)
            for remap in undistortion_maps
        )

    def project(self, image):
        return cv2.remap(
            image,
            *self.birdseye_maps,
            interpolation=cv2.INTER_LINEAR,
        )


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


def generate_camera_view(camera_name, arguments):
    input_path = os.path.join(RAW_DATA_DIR, camera_name + ".jpg")
    source_image = read_image(input_path)
    if source_image is None:
        raise FileNotFoundError("Input image not found: {}".format(input_path))

    projected_image = BirdseyeProjector(camera_name, arguments).project(source_image)
    unmasked_output_path = os.path.join(
        UNMASKED_BEV_DIR,
        "unmasked_bev_{}.jpg".format(camera_name),
    )
    write_image(unmasked_output_path, projected_image)

    masked_image = ViewMask(camera_name, arguments).apply(projected_image)
    masked_output_path = os.path.join(
        MASKED_BEV_DIR,
        "masked_bev_{}.jpg".format(camera_name),
    )
    write_image(masked_output_path, masked_image)
    return projected_image, masked_image


def combine_views(camera_views, arguments):
    weighted_sum = np.zeros(
        (arguments.bev_height, arguments.bev_width, 3),
        dtype=np.float64,
    )
    weight_sum = np.zeros(
        (arguments.bev_height, arguments.bev_width),
        dtype=np.float64,
    )

    for camera_name in ("left", "right", "back", "front"):
        image = camera_views[camera_name]
        view_mask = ViewMask(camera_name, arguments).mask
        # Distance-based feather weights
        distance_weight = cv2.distanceTransform(
            view_mask,
            cv2.DIST_L2,
            3,
        ).astype(np.float64)
        valid_pixels = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) > 0
        distance_weight = np.where(
            valid_pixels,
            np.maximum(distance_weight, 1.0),
            0.0,
        )
        weighted_sum += image.astype(np.float64) * distance_weight[:, :, None]
        weight_sum += distance_weight

    surround_view = np.zeros_like(weighted_sum)
    covered_pixels = weight_sum > 0
    surround_view[covered_pixels] = (
        weighted_sum[covered_pixels]
        / weight_sum[covered_pixels, None]
    )
    return np.clip(surround_view, 0, 255).astype(np.uint8)


def generate_balanced_view(masked_views, arguments, method):
    if method == "global":
        compensator = WholeViewExposureCompensator()
    else:
        compensator = OverlapExposureCompensator(CAMERA_ADJACENCIES)
    balanced_views, metrics = compensator.compensate(masked_views)
    os.makedirs(BALANCED_BEV_DIR, exist_ok=True)

    for camera_name, balanced_image in balanced_views.items():
        output_path = os.path.join(
            BALANCED_BEV_DIR,
            "balanced_bev_{}.jpg".format(camera_name),
        )
        write_image(output_path, balanced_image)

    balanced_surround_view = combine_views(balanced_views, arguments)
    write_image(
        os.path.join(OUTPUT_DIR, "surround_view_balanced.jpg"),
        balanced_surround_view,
    )
    metrics_path = os.path.join(OUTPUT_DIR, "exposure_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    for camera_name, gain in metrics["gains"].items():
        print("{} exposure gain: {:.4f}".format(camera_name, gain))
    return balanced_surround_view, metrics


def main():
    arguments = parse_arguments()
    os.makedirs(UNMASKED_BEV_DIR, exist_ok=True)
    os.makedirs(MASKED_BEV_DIR, exist_ok=True)

    # Directional masks with corner overlaps
    projected_views = {}
    masked_views = {}
    for camera_name in CAMERA_NAMES:
        projected_image, masked_image = generate_camera_view(
            camera_name,
            arguments,
        )
        projected_views[camera_name] = projected_image
        masked_views[camera_name] = masked_image

    surround_view = combine_views(masked_views, arguments)
    write_image(os.path.join(OUTPUT_DIR, "surround_view.jpg"), surround_view)

    if arguments.exposure_compensation != "off":
        generate_balanced_view(
            masked_views,
            arguments,
            arguments.exposure_compensation,
        )


if __name__ == "__main__":
    main()
