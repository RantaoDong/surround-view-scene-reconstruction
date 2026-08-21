from dataclasses import dataclass
import random

import carla


@dataclass(frozen=True)
class CameraRigConfig:
    image_width: int = 2000
    image_height: int = 2000
    field_of_view: float = 170.0
    camera_height: float = 1.0
    longitudinal_offset: float = 2.30
    lateral_offset: float = 1.70


@dataclass
class SimulationActors:
    vehicle: object
    front_camera: object
    back_camera: object
    left_camera: object
    right_camera: object

    @property
    def cameras(self):
        return (
            self.front_camera,
            self.back_camera,
            self.left_camera,
            self.right_camera,
        )

    @property
    def all(self):
        return (self.vehicle,) + self.cameras


def create_vehicle_transform(x, y, z=0.1):
    return carla.Transform(
        carla.Location(x=x, y=y, z=z),
        carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
    )


def spawn_vehicle_with_cameras(world, vehicle_transform, config=None):
    config = config or CameraRigConfig()
    blueprint_library = world.get_blueprint_library()
    vehicle_blueprint = random.choice(
        blueprint_library.filter("vehicle.tesla.model3")
    )
    vehicle = world.spawn_actor(vehicle_blueprint, vehicle_transform)
    vehicle.set_simulate_physics(False)

    camera_blueprint = blueprint_library.find("sensor.camera.rgb")
    camera_blueprint.set_attribute("image_size_x", str(config.image_width))
    camera_blueprint.set_attribute("image_size_y", str(config.image_height))
    camera_blueprint.set_attribute("fov", str(config.field_of_view))

    attachment_type = carla.AttachmentType.Rigid
    camera_transforms = {
        "front": carla.Transform(
            carla.Location(
                x=config.longitudinal_offset,
                z=config.camera_height,
            ),
            carla.Rotation(pitch=-90.0),
        ),
        "back": carla.Transform(
            carla.Location(
                x=-config.longitudinal_offset,
                z=config.camera_height,
            ),
            carla.Rotation(pitch=-90.0, yaw=180.0),
        ),
        "left": carla.Transform(
            carla.Location(y=-config.lateral_offset, z=config.camera_height),
            carla.Rotation(pitch=-90.0, yaw=-90.0),
        ),
        "right": carla.Transform(
            carla.Location(y=config.lateral_offset, z=config.camera_height),
            carla.Rotation(pitch=-90.0, yaw=90.0),
        ),
    }

    cameras = {
        name: world.spawn_actor(
            camera_blueprint,
            transform,
            attach_to=vehicle,
            attachment_type=attachment_type,
        )
        for name, transform in camera_transforms.items()
    }
    return SimulationActors(
        vehicle=vehicle,
        front_camera=cameras["front"],
        back_camera=cameras["back"],
        left_camera=cameras["left"],
        right_camera=cameras["right"],
    )


def destroy_actors(actors):
    if actors is None:
        return
    for actor in reversed(actors.all):
        actor.destroy()

