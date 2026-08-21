import os

import cv2
import numpy as np


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def read_image(path):
    encoded_image = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)


def write_image(path, image):
    extension = os.path.splitext(path)[1]
    success, encoded_image = cv2.imencode(extension, image)
    if not success:
        raise RuntimeError("Failed to encode image: {}".format(path))
    encoded_image.tofile(path)


class ImageCenteringTool:
    def __init__(self):
        self.center_x = 0
        self.center_y = 0
        self.source_image = None
        self.selection = {
            "top_left": None,
            "bottom_right": None,
            "cursor": None,
            "complete": False,
        }
        self.window_title = (
            "Select image center and press Y/N to validate; ESC keeps the original"
        )

    def handle_mouse(self, event, x, y, _flags, selection):
        if event == cv2.EVENT_LBUTTONDOWN:
            selection["cursor"] = (x, y)
            if selection["top_left"] is None:
                selection["top_left"] = selection["cursor"]

        if (
            event == cv2.EVENT_MOUSEMOVE
            and selection["top_left"] is not None
            and not selection["complete"]
        ):
            preview = self.source_image.copy()
            selection["cursor"] = (x, y)
            cv2.rectangle(
                preview,
                selection["top_left"],
                selection["cursor"],
                (0, 0, 255),
            )
            cv2.imshow(self.window_title, preview)

        if event == cv2.EVENT_LBUTTONUP and selection["top_left"] is not None:
            preview = self.source_image.copy()
            selection["bottom_right"] = (x, y)
            selection["complete"] = True
            cv2.rectangle(
                preview,
                selection["top_left"],
                selection["bottom_right"],
                (0, 0, 255),
            )

            self.center_x = (
                selection["top_left"][0] + selection["bottom_right"][0]
            ) // 2
            self.center_y = (
                selection["top_left"][1] + selection["bottom_right"][1]
            ) // 2
            label = "{},{}? (y/n)".format(self.center_x, self.center_y)
            cv2.circle(
                preview,
                (self.center_x, self.center_y),
                1,
                (0, 0, 255),
                thickness=2,
            )
            cv2.putText(
                preview,
                label,
                (self.center_x, self.center_y),
                cv2.FONT_HERSHEY_PLAIN,
                1.0,
                (0, 0, 0),
                thickness=1,
            )
            cv2.imshow(self.window_title, preview)

        self.selection = selection

    def translate_to_center(self, image):
        shift_x = image.shape[1] // 2 - self.center_x
        shift_y = image.shape[0] // 2 - self.center_y
        translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        return cv2.warpAffine(image, translation_matrix, image.shape[1::-1])

    def reset_selection(self):
        self.center_x = 0
        self.center_y = 0
        self.selection.update(
            top_left=None,
            bottom_right=None,
            cursor=None,
            complete=False,
        )

    def __call__(self, source_image):
        self.source_image = source_image
        cv2.namedWindow(
            self.window_title,
            flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO,
        )
        cv2.setMouseCallback(self.window_title, self.handle_mouse, self.selection)

        while True:
            cv2.imshow(self.window_title, self.source_image)
            key = cv2.waitKey(0)
            if key in (ord("y"), ord("Y")):
                break
            if key in (ord("n"), ord("N")):
                self.reset_selection()
            if key == 27:
                self.reset_selection()
                break

        cv2.destroyAllWindows()
        if self.center_x == 0 and self.center_y == 0:
            return self.source_image
        return self.translate_to_center(self.source_image)


def main():
    reference_dir = os.path.join(PROJECT_DIR, "sample_data", "reference")
    input_path = os.path.join(reference_dir, "birdseye_reference_raw.jpg")
    output_path = os.path.join(reference_dir, "birdseye_reference.jpg")
    source_image = read_image(input_path)
    if source_image is None:
        raise FileNotFoundError("Input image not found: {}".format(input_path))

    centered_image = ImageCenteringTool()(source_image)
    write_image(output_path, centered_image)

    cv2.namedWindow("centered_image", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow("centered_image", centered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
