import re

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def read_image(path):
    encoded_image = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to read image: {}".format(path))
    return image


def write_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix.lower()
    success, encoded_image = cv2.imencode(extension, image)
    if not success:
        raise ValueError("Failed to encode image: {}".format(path))
    encoded_image.tofile(path)


def natural_sort_key(path):
    return [
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.split(r"(\d+)", str(path))
    ]


def collect_image_paths(input_dir, output_dir, recursive):
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    image_paths = []
    try:
        output_dir.relative_to(input_dir)
        output_is_nested = True
    except ValueError:
        output_is_nested = False

    for path in iterator:
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if output_is_nested:
            try:
                path.resolve().relative_to(output_dir)
                continue
            except ValueError:
                pass
        image_paths.append(path)

    return sorted(image_paths, key=natural_sort_key)
