import argparse
from pathlib import Path

from bev_inpainting import BevInpainter, InpaintingConfig
from image_sequence import collect_image_paths, read_image, write_image
from motion_estimation import MotionConfig, OpticalFlowMotionEstimator


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = PROJECT_DIR / "sample_data" / "bev_sequence"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "filled"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Fill missing BEV regions with motion-aligned history",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=DEFAULT_INPUT_DIR,
        type=Path,
        help="Input BEV image directory",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=DEFAULT_OUTPUT_DIR,
        type=Path,
        help="Output image directory",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search input subdirectories",
    )
    parser.add_argument(
        "--frame-step",
        default=6,
        type=int,
        help="Process every Nth image",
    )
    parser.add_argument(
        "--start-index",
        default=0,
        type=int,
        help="First sorted image index",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        help="Exclusive sorted image index",
    )
    parser.add_argument(
        "--fill-mode",
        default="dark",
        choices=("dark", "center"),
        help="Missing-region selection method",
    )
    parser.add_argument(
        "--dark-threshold",
        default=35,
        type=int,
        help="Maximum luminance treated as missing",
    )
    parser.add_argument(
        "--center-width-ratio",
        default=0.28,
        type=float,
        help="Excluded vehicle-region width ratio",
    )
    parser.add_argument(
        "--center-height-ratio",
        default=0.68,
        type=float,
        help="Excluded vehicle-region height ratio",
    )
    parser.add_argument(
        "--horizontal-margin-ratio",
        default=80 / 600,
        type=float,
        help="Left and right feature-free margin ratio",
    )
    parser.add_argument(
        "--vertical-margin-ratio",
        default=10 / 600,
        type=float,
        help="Top and bottom feature-free margin ratio",
    )
    parser.add_argument(
        "--feature-center-width-ratio",
        default=160 / 600,
        type=float,
        help="Feature-free center-strip width ratio",
    )
    parser.add_argument(
        "--forward-backward-check",
        action="store_true",
        help="Enable reverse optical-flow validation",
    )
    parser.add_argument(
        "--estimator",
        default="lmeds",
        choices=("ransac", "lmeds"),
        help="Robust affine estimator",
    )
    return parser.parse_args()


def validate_arguments(arguments):
    if not arguments.input_dir.is_dir():
        raise NotADirectoryError(arguments.input_dir)
    if arguments.input_dir.resolve() == arguments.output_dir.resolve():
        raise ValueError("Input and output directories must be different")
    if arguments.frame_step < 1:
        raise ValueError("Frame step must be at least 1")
    if arguments.start_index < 0:
        raise ValueError("Start index must not be negative")
    for value, name in (
        (arguments.center_width_ratio, "Center width ratio"),
        (arguments.center_height_ratio, "Center height ratio"),
    ):
        if not 0.0 < value < 1.0:
            raise ValueError("{} must be between 0 and 1".format(name))
    for value, name in (
        (arguments.horizontal_margin_ratio, "Horizontal margin ratio"),
        (arguments.vertical_margin_ratio, "Vertical margin ratio"),
        (arguments.feature_center_width_ratio, "Feature center width ratio"),
    ):
        if not 0.0 <= value < 0.5:
            raise ValueError("{} must be between 0 and 0.5".format(name))
    if not 0 <= arguments.dark_threshold <= 255:
        raise ValueError("Dark threshold must be between 0 and 255")


def create_inpainter(arguments):
    motion_config = MotionConfig(
        horizontal_margin_ratio=arguments.horizontal_margin_ratio,
        vertical_margin_ratio=arguments.vertical_margin_ratio,
        center_exclusion_width_ratio=arguments.feature_center_width_ratio,
        forward_backward_check=arguments.forward_backward_check,
        estimator=arguments.estimator,
    )
    inpainting_config = InpaintingConfig(
        dark_threshold=arguments.dark_threshold,
        fill_mode=arguments.fill_mode,
        center_width_ratio=arguments.center_width_ratio,
        center_height_ratio=arguments.center_height_ratio,
    )
    motion_estimator = OpticalFlowMotionEstimator(motion_config)
    return BevInpainter(motion_estimator, inpainting_config)


def process_directory(arguments):
    validate_arguments(arguments)
    input_dir = arguments.input_dir.resolve()
    output_dir = arguments.output_dir.resolve()
    image_paths = collect_image_paths(
        input_dir,
        output_dir,
        arguments.recursive,
    )
    selected_paths = image_paths[
        arguments.start_index:arguments.end_index:arguments.frame_step
    ]
    if not selected_paths:
        raise FileNotFoundError("No supported images found in input directory")

    inpainter = create_inpainter(arguments)
    first_image = read_image(selected_paths[0])
    first_result = inpainter.initialize(first_image)
    first_relative_path = selected_paths[0].relative_to(input_dir)
    write_image(output_dir / first_relative_path, first_result.image)

    for index, current_path in enumerate(selected_paths[1:], start=1):
        current_image = read_image(current_path)
        frame_result = inpainter.process(current_image)
        relative_path = current_path.relative_to(input_dir)
        write_image(output_dir / relative_path, frame_result.image)
        status = "filled" if frame_result.motion_found else "unchanged"
        print(
            (
                "[{}/{}] {}: {}, filled_pixels={}, "
                "tracked={}, inliers={}"
            ).format(
                index + 1,
                len(selected_paths),
                relative_path,
                status,
                frame_result.filled_pixel_count,
                frame_result.tracked_count,
                frame_result.inlier_count,
            )
        )


def main():
    process_directory(parse_arguments())


if __name__ == "__main__":
    main()
