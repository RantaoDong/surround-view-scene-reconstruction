import argparse
import os

import cv2
import numpy as np


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized_value = value.lower()
    if normalized_value in ("true", "1", "yes", "y"):
        return True
    if normalized_value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Camera intrinsic calibration")
    parser.add_argument(
        "-input",
        "--INPUT_TYPE",
        "--input-type",
        dest="input_type",
        default="camera",
        choices=("camera", "video", "image"),
        help="Input source",
    )
    parser.add_argument(
        "-type",
        "--CAMERA_TYPE",
        "--camera-type",
        dest="camera_type",
        default="fisheye",
        choices=("fisheye", "normal"),
        help="Camera model",
    )
    parser.add_argument(
        "-id", "--CAMERA_ID", "--camera-id", dest="camera_id",
        default=2, type=int, help="Camera device ID"
    )
    parser.add_argument(
        "-path", "--INPUT_PATH", "--input-path", dest="input_path",
        default="./sample_data/intrinsic/", help="Input directory"
    )
    parser.add_argument(
        "-video", "--VIDEO_FILE", "--video-file", dest="video_file",
        default="video.mp4", help="Input video filename"
    )
    parser.add_argument(
        "-image", "--IMAGE_FILE", "--image-file", dest="image_prefix",
        default="img_raw", help="Input image prefix"
    )
    parser.add_argument(
        "-mode", "--SELECT_MODE", "--select-mode", dest="selection_mode",
        default="auto", choices=("auto", "manual"), help="Frame selection mode"
    )
    parser.add_argument(
        "-fw", "--FRAME_WIDTH", "--frame-width", dest="frame_width",
        default=1280, type=int, help="Camera frame width"
    )
    parser.add_argument(
        "-fh", "--FRAME_HEIGHT", "--frame-height", dest="frame_height",
        default=1024, type=int, help="Camera frame height"
    )
    parser.add_argument(
        "-bw", "--BORAD_WIDTH", "--BOARD_WIDTH", "--board-width",
        dest="board_width", default=8, type=int, help="Chessboard corner columns"
    )
    parser.add_argument(
        "-bh", "--BORAD_HEIGHT", "--BOARD_HEIGHT", "--board-height",
        dest="board_height", default=5, type=int, help="Chessboard corner rows"
    )
    parser.add_argument(
        "-size", "--SQUARE_SIZE", "--square-size", dest="square_size",
        default=100.0, type=float, help="Chessboard square size in millimetres"
    )
    parser.add_argument(
        "-num", "--CALIB_NUMBER", "--calibration-count",
        dest="calibration_count", default=15, type=int,
        help="Minimum valid frame count"
    )
    parser.add_argument(
        "-delay", "--FRAME_DELAY", "--frame-delay", dest="frame_delay",
        default=10, type=int, help="Automatic capture interval"
    )
    parser.add_argument(
        "-subpix", "--SUBPIX_REGION", "--subpixel-region",
        dest="subpixel_region", default=5, type=int,
        help="Corner refinement window"
    )
    parser.add_argument(
        "-fps", "--CAMERA_FPS", "--camera-fps", dest="camera_fps",
        default=20, type=int, help="Camera frame rate"
    )
    parser.add_argument(
        "-fs", "--FOCAL_SCALE", "--focal-scale", dest="focal_scale",
        default=1.2, type=float, help="Output focal scale"
    )
    parser.add_argument(
        "-ss", "--SIZE_SCALE", "--size-scale", dest="size_scale",
        default=2.0, type=float, help="Output size scale"
    )
    parser.add_argument(
        "-store", "--STORE_FLAG", "--store-images", dest="store_images",
        default=True, type=parse_bool, help="Store captured images"
    )
    parser.add_argument(
        "-store_path", "--STORE_PATH", "--store-path", dest="store_path",
        default="./outputs/captures/", help="Captured image directory"
    )
    parser.add_argument(
        "-crop", "--CROP_FLAG", "--crop", dest="crop",
        default=True, type=parse_bool, help="Center-crop input frames"
    )
    parser.add_argument(
        "-resize", "--RESIZE_FLAG", "--resize", dest="resize",
        default=True, type=parse_bool, help="Resize input frames"
    )
    return parser.parse_args()


class CalibrationResult:
    def __init__(self):
        self.camera_type = None
        self.camera_matrix = None
        self.distortion_coefficients = None
        self.rotation_vectors = None
        self.translation_vectors = None
        self.map_x = None
        self.map_y = None
        self.reprojection_errors = None
        self.rms_error = None
        self.valid = False


def create_board_points(arguments):
    # Chessboard geometry
    return np.array(
        [
            [
                (
                    column * arguments.square_size,
                    row * arguments.square_size,
                    0.0,
                )
            ]
            for row in range(arguments.board_height)
            for column in range(arguments.board_width)
        ],
        dtype=np.float32,
    )


class FisheyeCalibrationModel:
    def __init__(self, arguments):
        self.arguments = arguments
        self.result = CalibrationResult()
        self.initialized = False
        self.board_points = create_board_points(arguments)

    def update(self, detected_corners, frame_size):
        object_points = [self.board_points] * len(detected_corners)
        if self.initialized:
            self.refine(object_points, detected_corners, frame_size)
        else:
            self.initialize(object_points, detected_corners, frame_size)
            self.initialized = True
        self.calculate_reprojection_errors(detected_corners)
        self.create_undistortion_maps()

    def initialize(self, object_points, detected_corners, frame_size):
        result = self.result
        result.camera_type = "FISHEYE"
        result.camera_matrix = np.eye(3)
        result.distortion_coefficients = np.zeros((4, 1))
        (
            result.rms_error,
            result.camera_matrix,
            result.distortion_coefficients,
            result.rotation_vectors,
            result.translation_vectors,
        ) = cv2.fisheye.calibrate(
            object_points,
            detected_corners,
            frame_size,
            result.camera_matrix,
            result.distortion_coefficients,
            flags=(
                cv2.fisheye.CALIB_FIX_SKEW
                | cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
            ),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 1e-6),
        )
        result.valid = cv2.checkRange(result.camera_matrix)[0] and cv2.checkRange(
            result.distortion_coefficients
        )[0]

    def refine(self, object_points, detected_corners, frame_size):
        result = self.result
        (
            result.rms_error,
            result.camera_matrix,
            result.distortion_coefficients,
            result.rotation_vectors,
            result.translation_vectors,
        ) = cv2.fisheye.calibrate(
            object_points,
            detected_corners,
            frame_size,
            result.camera_matrix,
            result.distortion_coefficients,
            flags=(
                cv2.fisheye.CALIB_FIX_SKEW
                | cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
                | cv2.CALIB_USE_INTRINSIC_GUESS
            ),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 10, 1e-6),
        )
        result.valid = cv2.checkRange(result.camera_matrix)[0] and cv2.checkRange(
            result.distortion_coefficients
        )[0]

    def calculate_reprojection_errors(self, detected_corners):
        result = self.result
        result.reprojection_errors = []
        for index, observed_corners in enumerate(detected_corners):
            projected_corners, _ = cv2.fisheye.projectPoints(
                self.board_points,
                result.rotation_vectors[index],
                result.translation_vectors[index],
                result.camera_matrix,
                result.distortion_coefficients,
            )
            error = cv2.norm(
                projected_corners,
                observed_corners,
                cv2.NORM_L2,
            ) / len(projected_corners)
            result.reprojection_errors.append(error)

    def create_undistortion_maps(self):
        result = self.result
        output_camera_matrix = result.camera_matrix.copy()
        output_camera_matrix[0, 0] *= self.arguments.focal_scale
        output_camera_matrix[1, 1] *= self.arguments.focal_scale
        output_camera_matrix[0, 2] = (
            self.arguments.frame_width / 2 * self.arguments.size_scale
        )
        output_camera_matrix[1, 2] = (
            self.arguments.frame_height / 2 * self.arguments.size_scale
        )
        output_size = (
            int(self.arguments.frame_width * self.arguments.size_scale),
            int(self.arguments.frame_height * self.arguments.size_scale),
        )
        result.map_x, result.map_y = cv2.fisheye.initUndistortRectifyMap(
            result.camera_matrix,
            result.distortion_coefficients,
            np.eye(3),
            output_camera_matrix,
            output_size,
            cv2.CV_16SC2,
        )


class PinholeCalibrationModel:
    def __init__(self, arguments):
        self.arguments = arguments
        self.result = CalibrationResult()
        self.initialized = False
        self.board_points = create_board_points(arguments)

    def update(self, detected_corners, frame_size):
        object_points = [self.board_points] * len(detected_corners)
        if self.initialized:
            self.refine(object_points, detected_corners, frame_size)
        else:
            self.initialize(object_points, detected_corners, frame_size)
            self.initialized = True
        self.calculate_reprojection_errors(detected_corners)
        self.create_undistortion_maps()

    def initialize(self, object_points, detected_corners, frame_size):
        result = self.result
        result.camera_type = "NORMAL"
        result.camera_matrix = np.eye(3)
        result.distortion_coefficients = np.zeros((5, 1))
        (
            result.rms_error,
            result.camera_matrix,
            result.distortion_coefficients,
            result.rotation_vectors,
            result.translation_vectors,
        ) = cv2.calibrateCamera(
            object_points,
            detected_corners,
            frame_size,
            result.camera_matrix,
            result.distortion_coefficients,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 1e-6),
        )
        result.valid = cv2.checkRange(result.camera_matrix)[0] and cv2.checkRange(
            result.distortion_coefficients
        )[0]

    def refine(self, object_points, detected_corners, frame_size):
        result = self.result
        (
            result.rms_error,
            result.camera_matrix,
            result.distortion_coefficients,
            result.rotation_vectors,
            result.translation_vectors,
        ) = cv2.calibrateCamera(
            object_points,
            detected_corners,
            frame_size,
            result.camera_matrix,
            result.distortion_coefficients,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 10, 1e-6),
        )
        result.valid = cv2.checkRange(result.camera_matrix)[0] and cv2.checkRange(
            result.distortion_coefficients
        )[0]

    def calculate_reprojection_errors(self, detected_corners):
        result = self.result
        result.reprojection_errors = []
        for index, observed_corners in enumerate(detected_corners):
            projected_corners, _ = cv2.projectPoints(
                self.board_points,
                result.rotation_vectors[index],
                result.translation_vectors[index],
                result.camera_matrix,
                result.distortion_coefficients,
            )
            error = cv2.norm(
                projected_corners,
                observed_corners,
                cv2.NORM_L2,
            ) / len(projected_corners)
            result.reprojection_errors.append(error)

    def create_undistortion_maps(self):
        result = self.result
        output_camera_matrix = result.camera_matrix.copy()
        output_camera_matrix[0, 0] *= self.arguments.focal_scale
        output_camera_matrix[1, 1] *= self.arguments.focal_scale
        output_camera_matrix[0, 2] = (
            self.arguments.frame_width / 2 * self.arguments.size_scale
        )
        output_camera_matrix[1, 2] = (
            self.arguments.frame_height / 2 * self.arguments.size_scale
        )
        output_size = (
            int(self.arguments.frame_width * self.arguments.size_scale),
            int(self.arguments.frame_height * self.arguments.size_scale),
        )
        result.map_x, result.map_y = cv2.initUndistortRectifyMap(
            result.camera_matrix,
            result.distortion_coefficients,
            np.eye(3),
            output_camera_matrix,
            output_size,
            cv2.CV_16SC2,
        )


class IntrinsicCalibrator:
    def __init__(self, camera_type, arguments):
        self.arguments = arguments
        if camera_type == "fisheye":
            self.model = FisheyeCalibrationModel(arguments)
        elif camera_type == "normal":
            self.model = PinholeCalibrationModel(arguments)
        else:
            raise ValueError("Unknown camera type: {}".format(camera_type))
        self.detected_corner_sets = []

    def find_corners(self, image):
        board_size = (self.arguments.board_width, self.arguments.board_height)
        found, corners = cv2.findChessboardCorners(
            image,
            board_size,
            flags=(
                cv2.CALIB_CB_ADAPTIVE_THRESH
                | cv2.CALIB_CB_NORMALIZE_IMAGE
                | cv2.CALIB_CB_FAST_CHECK
            ),
        )
        if found:
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            corners = cv2.cornerSubPix(
                grayscale_image,
                corners,
                (self.arguments.subpixel_region, self.arguments.subpixel_region),
                (-1, -1),
                (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.01,
                ),
            )
        return found, corners

    def draw_corners(self, image):
        preview = image.copy()
        found, corners = self.find_corners(preview)
        if found:
            board_size = (self.arguments.board_width, self.arguments.board_height)
            cv2.drawChessboardCorners(preview, board_size, corners, found)
        return preview

    def undistort(self, image):
        return cv2.remap(
            image,
            self.model.result.map_x,
            self.model.result.map_y,
            cv2.INTER_LINEAR,
        )

    def calibrate(self, image):
        if len(self.detected_corner_sets) >= self.arguments.calibration_count:
            self.model.update(self.detected_corner_sets, image.shape[1::-1])
        return self.model.result

    def __call__(self, image):
        found, corners = self.find_corners(image)
        if found:
            self.detected_corner_sets.append(corners)
            return self.calibrate(image)
        return self.model.result


def center_crop(image, width, height):
    if image.shape[1] < width or image.shape[0] < height:
        raise ValueError("Crop size exceeds input image size")
    top = round((image.shape[0] - height) / 2)
    left = round((image.shape[1] - width) / 2)
    return image[top : top + height, left : left + width]


def find_images(directory, filename_prefix):
    filenames = [
        os.path.join(directory, filename)
        for filename in os.listdir(directory)
        if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS
        and filename_prefix in filename
    ]
    filenames.sort()
    if not filenames:
        raise FileNotFoundError("No matching images found in {}".format(directory))
    return filenames


class CalibrationRunner:
    def __init__(self, calibrator, arguments):
        self.calibrator = calibrator
        self.arguments = arguments

    def preprocess_image(self, image):
        if image is None:
            raise ValueError("Empty input frame")
        if self.arguments.crop:
            return center_crop(
                image,
                self.arguments.frame_width,
                self.arguments.frame_height,
            )
        if self.arguments.resize:
            return cv2.resize(
                image,
                (self.arguments.frame_width, self.arguments.frame_height),
            )
        return image

    def configure_camera(self, capture):
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("M", "J", "P", "G"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.arguments.frame_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.arguments.frame_height)
        capture.set(cv2.CAP_PROP_FPS, self.arguments.camera_fps)
        return capture

    def process_frame(self, source_image, display_raw=True, display_undistorted=True):
        processed_image = self.preprocess_image(source_image)
        result = self.calibrator(processed_image)
        corner_preview = self.calibrator.draw_corners(processed_image)

        if display_raw:
            cv2.namedWindow("raw_frame", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.imshow("raw_frame", corner_preview)
        if (
            len(self.calibrator.detected_corner_sets)
            >= self.arguments.calibration_count
            and display_undistorted
        ):
            undistorted_image = self.calibrator.undistort(processed_image)
            cv2.namedWindow(
                "undistorted_frame",
                cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO,
            )
            cv2.imshow("undistorted_frame", undistorted_image)
        cv2.waitKey(1)
        return result

    def store_frame(self, image):
        if not self.arguments.store_images:
            return
        os.makedirs(self.arguments.store_path, exist_ok=True)
        filename = "img_raw{}.jpg".format(
            len(self.calibrator.detected_corner_sets)
        )
        cv2.imwrite(os.path.join(self.arguments.store_path, filename), image)

    def run_image_auto(self):
        result = self.calibrator.model.result
        filenames = find_images(
            self.arguments.input_path,
            self.arguments.image_prefix,
        )
        for filename in filenames:
            print(filename)
            result = self.process_frame(cv2.imread(filename))
            if cv2.waitKey(1) == 27:
                break
        cv2.destroyAllWindows()
        return result

    def run_image_manual(self):
        result = self.calibrator.model.result
        filenames = find_images(
            self.arguments.input_path,
            self.arguments.image_prefix,
        )
        for filename in filenames:
            print(filename)
            source_image = self.preprocess_image(cv2.imread(filename))
            preview = self.calibrator.draw_corners(source_image)
            window_title = "SPACE: select | other: skip | ESC: quit"
            cv2.namedWindow(
                window_title,
                cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO,
            )
            cv2.imshow(window_title, preview)
            key = cv2.waitKey(0)
            if key == 32:
                result = self.process_frame(source_image, display_raw=False)
            if key == 27:
                break
        cv2.destroyAllWindows()
        return result

    def open_video(self):
        video_path = os.path.join(
            self.arguments.input_path,
            self.arguments.video_file,
        )
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise OSError("Failed to open video: {}".format(video_path))
        return capture

    def run_video_auto(self):
        result = self.calibrator.model.result
        capture = self.open_video()
        frame_index = 0
        while True:
            frame_available, source_image = capture.read()
            if not frame_available:
                break
            source_image = cv2.flip(source_image, 0)
            processed_image = self.preprocess_image(source_image)

            # Capture interval
            if frame_index % self.arguments.frame_delay == 0:
                self.store_frame(processed_image)
                result = self.process_frame(processed_image)
                print(len(self.calibrator.detected_corner_sets))
            frame_index += 1
            if cv2.waitKey(1) == 27:
                break
        capture.release()
        cv2.destroyAllWindows()
        return result

    def run_video_manual(self):
        result = self.calibrator.model.result
        capture = self.open_video()
        while True:
            frame_available, source_image = capture.read()
            if not frame_available:
                break
            processed_image = self.preprocess_image(source_image)
            window_title = "SPACE: capture | ESC: quit"
            cv2.namedWindow(
                window_title,
                cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO,
            )
            cv2.imshow(window_title, processed_image)
            key = cv2.waitKey(1)
            if key == 32:
                self.store_frame(processed_image)
                result = self.process_frame(processed_image)
                print(len(self.calibrator.detected_corner_sets))
            if key == 27:
                break
        capture.release()
        cv2.destroyAllWindows()
        return result

    def open_camera(self):
        capture = cv2.VideoCapture(self.arguments.camera_id)
        if not capture.isOpened():
            raise OSError(
                "Failed to open camera: {}".format(self.arguments.camera_id)
            )
        return self.configure_camera(capture)

    def run_camera_auto(self):
        result = self.calibrator.model.result
        capture = self.open_camera()
        frame_index = 0
        started = False
        while True:
            frame_available, source_image = capture.read()
            if not frame_available:
                break
            processed_image = self.preprocess_image(source_image)
            key = cv2.waitKey(1)
            if key == 32:
                started = True
            if key == 27:
                break
            if not started:
                cv2.putText(
                    processed_image,
                    "Press SPACE to start",
                    (self.arguments.frame_width // 4, self.arguments.frame_height // 2),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1.5,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow("raw_frame", processed_image)
                continue

            # Capture interval
            if frame_index % self.arguments.frame_delay == 0:
                self.store_frame(processed_image)
                result = self.process_frame(processed_image)
                print(len(self.calibrator.detected_corner_sets))
            frame_index += 1
        capture.release()
        cv2.destroyAllWindows()
        return result

    def run_camera_manual(self):
        result = self.calibrator.model.result
        capture = self.open_camera()
        while True:
            frame_available, source_image = capture.read()
            if not frame_available:
                break
            processed_image = self.preprocess_image(source_image)
            window_title = "SPACE: capture | ESC: quit"
            cv2.namedWindow(
                window_title,
                cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO,
            )
            cv2.imshow(window_title, processed_image)
            key = cv2.waitKey(1)
            if key == 32:
                self.store_frame(processed_image)
                result = self.process_frame(processed_image)
                print(len(self.calibrator.detected_corner_sets))
            if key == 27:
                break
        capture.release()
        cv2.destroyAllWindows()
        return result

    def run(self):
        runners = {
            ("image", "auto"): self.run_image_auto,
            ("image", "manual"): self.run_image_manual,
            ("video", "auto"): self.run_video_auto,
            ("video", "manual"): self.run_video_manual,
            ("camera", "auto"): self.run_camera_auto,
            ("camera", "manual"): self.run_camera_manual,
        }
        return runners[(self.arguments.input_type, self.arguments.selection_mode)]()


def main():
    arguments = parse_arguments()
    calibrator = IntrinsicCalibrator(arguments.camera_type, arguments)
    result = CalibrationRunner(calibrator, arguments).run()

    detected_count = len(calibrator.detected_corner_sets)
    if detected_count == 0:
        raise RuntimeError("Calibration failed: chessboard not found")
    if detected_count < arguments.calibration_count:
        raise RuntimeError(
            "Calibration failed: at least {} valid images required".format(
                arguments.calibration_count
            )
        )
    if not result.valid:
        raise RuntimeError("Calibration failed: invalid parameters")

    print("Calibration complete")
    print("Camera matrix: {}".format(result.camera_matrix.tolist()))
    print(
        "Distortion coefficients: {}".format(
            result.distortion_coefficients.tolist()
        )
    )
    print(
        "Mean reprojection error: {}".format(
            np.mean(result.reprojection_errors)
        )
    )
    np.save(
        "camera_{}_K.npy".format(arguments.camera_id),
        result.camera_matrix,
    )
    np.save(
        "camera_{}_D.npy".format(arguments.camera_id),
        result.distortion_coefficients,
    )


if __name__ == "__main__":
    main()
