import argparse
from pathlib import Path

from .bev_stitching import SurroundViewStitcher
from .runtime import run_online_pipeline
from .temporal_filling import FourFrameCenterFiller


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run transparent CARLA BEV")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=2000, type=int)
    parser.add_argument("--timeout", default=2.0, type=float)
    parser.add_argument("--fps", default=20, type=int)
    parser.add_argument("--max-frames", default=0, type=int)
    parser.add_argument(
        "--output-dir",
        default=Path("outputs/transparent"),
        type=Path,
    )
    return parser.parse_args()


def main():
    stitcher = SurroundViewStitcher()
    center_filler = FourFrameCenterFiller(stitcher.layout)

    def process(front, back, left, right):
        surround_view = stitcher.stitch(front, back, left, right)
        return center_filler.process(surround_view)

    run_online_pipeline(parse_arguments(), process)


if __name__ == "__main__":
    main()

