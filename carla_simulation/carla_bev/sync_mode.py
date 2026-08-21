import queue

import carla


class CarlaSyncMode:
    def __init__(self, world, *sensors, fps=20):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / fps
        self.queues = []
        self.previous_settings = None

    def __enter__(self):
        self.previous_settings = self.world.get_settings()
        self.frame = self.world.apply_settings(
            carla.WorldSettings(
                no_rendering_mode=False,
                synchronous_mode=True,
                fixed_delta_seconds=self.delta_seconds,
            )
        )

        def register_queue(register_event):
            sensor_queue = queue.Queue()
            register_event(sensor_queue.put)
            self.queues.append(sensor_queue)

        register_queue(self.world.on_tick)
        for sensor in self.sensors:
            register_queue(sensor.listen)
        return self

    def __exit__(self, *args):
        self.world.apply_settings(self.previous_settings)

    def tick(self, timeout):
        self.frame = self.world.tick()
        synchronized_data = [
            self._retrieve_data(sensor_queue, timeout)
            for sensor_queue in self.queues
        ]
        assert all(data.frame == self.frame for data in synchronized_data)
        return synchronized_data

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data

