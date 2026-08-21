import argparse
from pathlib import Path

from .bev_stitching import SurroundViewStitcher
from .runtime import run_online_pipeline


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run CARLA surround-view BEV")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=2000, type=int)
    parser.add_argument("--timeout", default=2.0, type=float)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--max-frames", default=0, type=int)
    parser.add_argument(
        "--output-dir",
        default=Path("outputs/surround"),
        type=Path,
    )
    return parser.parse_args()


def main():
    stitcher = SurroundViewStitcher()
    run_online_pipeline(parse_arguments(), stitcher.stitch)


if __name__ == "__main__":
    main()

