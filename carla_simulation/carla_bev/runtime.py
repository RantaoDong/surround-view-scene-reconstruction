from pathlib import Path
import random

import carla
import cv2
import pygame

from .image_conversion import (
    carla_image_to_bgr,
    draw_camera_image,
    exit_requested,
    get_display_font,
)
from .sensor_setup import (
    create_vehicle_transform,
    destroy_actors,
    spawn_vehicle_with_cameras,
)
from .sync_mode import CarlaSyncMode


def run_online_pipeline(arguments, processor):
    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    actors = None
    pygame.init()
    display = pygame.display.set_mode(
        (1000, 600),
        pygame.HWSURFACE | pygame.DOUBLEBUF,
    )
    font = get_display_font()
    clock = pygame.time.Clock()

    try:
        client = carla.Client(arguments.host, arguments.port)
        client.set_timeout(arguments.timeout)
        world = client.get_world()
        vehicle_transform = create_vehicle_transform(10.0, 10.0)
        waypoint = world.get_map().get_waypoint(vehicle_transform.location)
        actors = spawn_vehicle_with_cameras(world, vehicle_transform)
        world.tick()

        synchronized_cameras = (
            actors.front_camera,
            actors.left_camera,
            actors.right_camera,
            actors.back_camera,
        )
        with CarlaSyncMode(
            world,
            *synchronized_cameras,
            fps=arguments.fps,
        ) as sync_mode:
            processed_count = 0
            while True:
                if exit_requested():
                    break
                if arguments.max_frames and processed_count >= arguments.max_frames:
                    break
                clock.tick()
                snapshot, front, left, right, back = sync_mode.tick(
                    timeout=arguments.timeout
                )
                waypoint = random.choice(waypoint.next(1.5))
                actors.vehicle.set_transform(waypoint.transform)

                draw_camera_image(display, front, (0, 0))
                draw_camera_image(display, back, (490, 0))
                draw_camera_image(display, left, (0, 280))
                draw_camera_image(display, right, (490, 280))
                simulated_fps = round(1.0 / snapshot.timestamp.delta_seconds)
                display.blit(
                    font.render(
                        "%5d FPS (real)" % clock.get_fps(),
                        True,
                        (255, 255, 255),
                    ),
                    (8, 10),
                )
                display.blit(
                    font.render(
                        "%5d FPS (simulated)" % simulated_fps,
                        True,
                        (255, 255, 255),
                    ),
                    (8, 28),
                )
                pygame.display.flip()

                result = processor(
                    carla_image_to_bgr(front),
                    carla_image_to_bgr(back),
                    carla_image_to_bgr(left),
                    carla_image_to_bgr(right),
                )
                output_path = output_dir / "{:06d}_bev.png".format(
                    snapshot.frame
                )
                cv2.imwrite(str(output_path), result)
                cv2.namedWindow(
                    "surround_view",
                    flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO,
                )
                cv2.imshow("surround_view", result)
                if cv2.waitKey(1) == ord("q"):
                    break
                processed_count += 1
    finally:
        destroy_actors(actors)
        cv2.destroyAllWindows()
        pygame.quit()

