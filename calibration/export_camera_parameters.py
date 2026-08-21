import os

import cv2
import numpy as np


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CAMERA_NAMES = ("front", "left", "right", "back")


def load_camera_parameters(camera_name):
    camera_dir = os.path.join(PROJECT_DIR, "config", camera_name)
    filename_prefix = "camera_{}_".format(camera_name)
    camera_matrix = np.load(os.path.join(camera_dir, filename_prefix + "K.npy"))
    distortion_coefficients = np.load(
        os.path.join(camera_dir, filename_prefix + "D.npy")
    )
    homography = np.load(os.path.join(camera_dir, filename_prefix + "H.npy"))
    return camera_matrix, distortion_coefficients, homography


def write_camera_config(camera_name):
    camera_matrix, distortion_coefficients, homography = load_camera_parameters(
        camera_name
    )
    config_dir = os.path.join(PROJECT_DIR, "config")
    yaml_filename = camera_name + ".yaml"
    previous_directory = os.getcwd()
    storage = None
    try:
        # ASCII path for OpenCV FileStorage
        os.chdir(config_dir)
        storage = cv2.FileStorage(yaml_filename, cv2.FileStorage_WRITE)
        if not storage.isOpened():
            raise OSError("Failed to open output file: {}".format(yaml_filename))

        storage.write("FRAME_WIDTH", 1280)
        storage.write("FRAME_HEIGHT", 1024)
        storage.write("FOCAL_SCALE", 1.2)
        storage.write("SIZE_SCALE", 1)
        storage.write("K", camera_matrix)
        storage.write("D", distortion_coefficients)
        storage.write("H", homography)
    finally:
        if storage is not None:
            storage.release()
        os.chdir(previous_directory)


def main():
    for camera_name in CAMERA_NAMES:
        write_camera_config(camera_name)


if __name__ == "__main__":
    main()
