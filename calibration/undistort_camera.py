import argparse
import os

import cv2
import numpy as np


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CAMERA_NAMES = ("front", "back", "left", "right")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Undistort a fisheye camera image")
    parser.add_argument(
        "-fw", "--FRAME_WIDTH", "--frame-width", dest="frame_width",
        default=1280, type=int, help="Camera frame width"
    )
    parser.add_argument(
        "-fh", "--FRAME_HEIGHT", "--frame-height", dest="frame_height",
        default=1024, type=int, help="Camera frame height"
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
        "-c",
        "--CAMERA_NAME",
        "--camera-name",
        dest="camera_name",
        default="right",
        choices=CAMERA_NAMES,
        help="Camera name",
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


class FisheyeUndistorter:
    def __init__(
        self,
        camera_name,
        frame_width,
        frame_height,
        focal_scale,
        size_scale,
    ):
        config_dir = os.path.join(PROJECT_DIR, "config", camera_name)
        filename_prefix = "camera_{}_".format(camera_name)
        self.camera_matrix = np.load(
            os.path.join(config_dir, filename_prefix + "K.npy")
        )
        self.distortion_coefficients = np.load(
            os.path.join(config_dir, filename_prefix + "D.npy")
        )
        self.output_size = (
            int(frame_width * size_scale),
            int(frame_height * size_scale),
        )
        self.output_camera_matrix = self.camera_matrix.copy()
        self.output_camera_matrix[0, 0] *= focal_scale
        self.output_camera_matrix[1, 1] *= focal_scale
        self.undistortion_maps = cv2.fisheye.initUndistortRectifyMap(
            self.camera_matrix,
            self.distortion_coefficients,
            np.eye(3),
            self.output_camera_matrix,
            self.output_size,
            cv2.CV_16SC2,
        )

    def undistort(self, image):
        return cv2.remap(
            image,
            *self.undistortion_maps,
            interpolation=cv2.INTER_LINEAR,
        )


def main():
    arguments = parse_arguments()
    input_path = os.path.join(
        PROJECT_DIR,
        "sample_data",
        "raw",
        arguments.camera_name + ".jpg",
    )
    output_dir = os.path.join(PROJECT_DIR, "outputs", "undistorted")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        "undistorted_{}.jpg".format(arguments.camera_name),
    )
    source_image = read_image(input_path)
    if source_image is None:
        raise FileNotFoundError("Input image not found: {}".format(input_path))

    undistorter = FisheyeUndistorter(
        arguments.camera_name,
        arguments.frame_width,
        arguments.frame_height,
        arguments.focal_scale,
        arguments.size_scale,
    )
    undistorted_image = undistorter.undistort(source_image)
    write_image(output_path, undistorted_image)

    cv2.namedWindow("undistorted_image", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow("undistorted_image", undistorted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
