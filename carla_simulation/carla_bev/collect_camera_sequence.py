import argparse
from pathlib import Path
import time

import carla

from .sensor_setup import (
    create_vehicle_transform,
    destroy_actors,
    spawn_vehicle_with_cameras,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Collect four CARLA camera streams")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=2000, type=int)
    parser.add_argument("--timeout", default=2.0, type=float)
    parser.add_argument("--duration", default=1000.0, type=float)
    parser.add_argument(
        "--output-dir",
        default=Path("outputs/raw"),
        type=Path,
    )
    return parser.parse_args()


def frame_number(image):
    return image.frame if hasattr(image, "frame") else image.frame_number


def register_camera_output(camera, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    camera.listen(
        lambda image: image.save_to_disk(
            str(output_dir / "{:06d}.png".format(frame_number(image)))
        )
    )


def main():
    arguments = parse_arguments()
    actors = None
    try:
        client = carla.Client(arguments.host, arguments.port)
        client.set_timeout(arguments.timeout)
        world = client.get_world()
        vehicle_transform = create_vehicle_transform(-43.0, 23.0)
        actors = spawn_vehicle_with_cameras(world, vehicle_transform)
        actors.vehicle.apply_control(
            carla.VehicleControl(throttle=1.0, steer=0.0)
        )
        for camera_name, camera in (
            ("front", actors.front_camera),
            ("back", actors.back_camera),
            ("left", actors.left_camera),
            ("right", actors.right_camera),
        ):
            register_camera_output(
                camera,
                arguments.output_dir / camera_name,
            )
        time.sleep(arguments.duration)
    finally:
        destroy_actors(actors)


if __name__ == "__main__":
    main()

