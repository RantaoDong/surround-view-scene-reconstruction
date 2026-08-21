import cv2
import numpy as np


class ExposureCompensator:
    @staticmethod
    def get_luminance(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float64)

    @staticmethod
    def get_valid_mask(image, luminance):
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return (
            (grayscale > 5)
            & (luminance > 10)
            & (luminance < 245)
        )

    def measure_pair(self, first_name, second_name, luminances, masks):
        overlap_mask = masks[first_name] & masks[second_name]
        overlap_pixels = int(np.count_nonzero(overlap_mask))
        if overlap_pixels < self.minimum_overlap_pixels:
            raise ValueError(
                "Insufficient overlap for {} and {}: {} pixels".format(
                    first_name,
                    second_name,
                    overlap_pixels,
                )
            )

        first_values = luminances[first_name][overlap_mask]
        second_values = luminances[second_name][overlap_mask]
        log_difference = np.log(second_values + 1.0) - np.log(
            first_values + 1.0
        )
        return {
            "first": first_name,
            "second": second_name,
            "overlap_pixels": overlap_pixels,
            "median_log_difference": float(np.median(log_difference)),
        }

    @staticmethod
    def solve_log_gains(camera_names, measurements):
        camera_indices = {
            camera_name: index for index, camera_name in enumerate(camera_names)
        }
        system_rows = []
        targets = []

        for measurement in measurements:
            row = np.zeros(len(camera_names), dtype=np.float64)
            row[camera_indices[measurement["first"]]] = 1.0
            row[camera_indices[measurement["second"]]] = -1.0
            system_rows.append(row)
            targets.append(measurement["median_log_difference"])

        # Zero-mean log gain
        system_rows.append(np.ones(len(camera_names)))
        targets.append(0.0)
        system_matrix = np.vstack(system_rows)
        target_vector = np.asarray(targets, dtype=np.float64)
        log_gains, _, rank, _ = np.linalg.lstsq(
            system_matrix,
            target_vector,
            rcond=None,
        )
        if rank < len(camera_names):
            raise ValueError("Camera overlap graph is not fully connected")
        return {
            camera_name: float(log_gains[camera_indices[camera_name]])
            for camera_name in camera_names
        }

    @staticmethod
    def apply_log_gain(image, log_gain):
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        luminance = lab_image[:, :, 0].astype(np.float64)
        valid_mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) > 0
        adjusted_luminance = np.expm1(np.log1p(luminance) + log_gain)
        lab_image[:, :, 0] = np.where(
            valid_mask,
            np.clip(adjusted_luminance, 0, 255),
            0,
        ).astype(np.uint8)
        return cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)


class WholeViewExposureCompensator(ExposureCompensator):
    @classmethod
    def measure_views(cls, images):
        measurements = {}
        for camera_name, image in images.items():
            luminance = cls.get_luminance(image)
            valid_mask = cls.get_valid_mask(image, luminance)
            valid_values = luminance[valid_mask]
            if valid_values.size == 0:
                raise ValueError(
                    "No valid pixels for camera: {}".format(camera_name)
                )
            mean_log_luminance = float(np.mean(np.log1p(valid_values)))
            measurements[camera_name] = {
                "valid_pixels": int(valid_values.size),
                "mean_luminance": float(np.mean(valid_values)),
                "geometric_mean_luminance": float(
                    np.expm1(mean_log_luminance)
                ),
                "mean_log_luminance": mean_log_luminance,
            }
        return measurements

    def compensate(self, images):
        measurements_before = self.measure_views(images)
        target_log_luminance = float(
            np.mean([
                measurement["mean_log_luminance"]
                for measurement in measurements_before.values()
            ])
        )
        log_gains = {
            camera_name: (
                target_log_luminance
                - measurement["mean_log_luminance"]
            )
            for camera_name, measurement in measurements_before.items()
        }
        compensated_images = {
            camera_name: self.apply_log_gain(image, log_gains[camera_name])
            for camera_name, image in images.items()
        }
        measurements_after = self.measure_views(compensated_images)
        spread_before = float(np.std([
            measurement["mean_log_luminance"]
            for measurement in measurements_before.values()
        ]))
        spread_after = float(np.std([
            measurement["mean_log_luminance"]
            for measurement in measurements_after.values()
        ]))
        reduction = 0.0
        if spread_before > 0:
            reduction = (spread_before - spread_after) / spread_before * 100.0

        return compensated_images, {
            "method": "masked_whole_view_log_luminance",
            "target_geometric_mean_luminance": float(
                np.expm1(target_log_luminance)
            ),
            "gains": {
                camera_name: float(np.exp(log_gain))
                for camera_name, log_gain in log_gains.items()
            },
            "cameras_before": measurements_before,
            "cameras_after": measurements_after,
            "summary": {
                "log_luminance_spread_before": spread_before,
                "log_luminance_spread_after": spread_after,
                "spread_reduction_percent": reduction,
            },
        }


class OverlapExposureCompensator(ExposureCompensator):
    def __init__(self, camera_adjacencies, minimum_overlap_pixels=5000):
        self.camera_adjacencies = tuple(camera_adjacencies)
        self.minimum_overlap_pixels = minimum_overlap_pixels

    def calculate_pair_biases(self, images, reference_masks=None):
        luminances = {
            camera_name: self.get_luminance(image)
            for camera_name, image in images.items()
        }
        if reference_masks is None:
            masks = {
                camera_name: self.get_valid_mask(image, luminances[camera_name])
                for camera_name, image in images.items()
            }
        else:
            masks = reference_masks
        measurements = []
        for first_name, second_name in self.camera_adjacencies:
            measurements.append(
                self.measure_pair(
                    first_name,
                    second_name,
                    luminances,
                    masks,
                )
            )
        return measurements, masks

    def compensate(self, images):
        camera_names = tuple(images)
        unknown_cameras = set(
            camera_name
            for adjacency in self.camera_adjacencies
            for camera_name in adjacency
        ) - set(camera_names)
        if unknown_cameras:
            raise ValueError(
                "Missing camera images: {}".format(sorted(unknown_cameras))
            )

        measurements_before, reference_masks = self.calculate_pair_biases(images)
        log_gains = self.solve_log_gains(camera_names, measurements_before)
        compensated_images = {
            camera_name: self.apply_log_gain(image, log_gains[camera_name])
            for camera_name, image in images.items()
        }
        measurements_after, _ = self.calculate_pair_biases(
            compensated_images,
            reference_masks,
        )

        pair_metrics = {}
        biases_before = []
        biases_after = []
        for before, after in zip(measurements_before, measurements_after):
            pair_name = "{}-{}".format(before["first"], before["second"])
            before_bias = abs(before["median_log_difference"])
            after_bias = abs(after["median_log_difference"])
            biases_before.append(before_bias)
            biases_after.append(after_bias)
            reduction = 0.0
            if before_bias > 0:
                reduction = (before_bias - after_bias) / before_bias * 100.0
            pair_metrics[pair_name] = {
                "overlap_pixels": before["overlap_pixels"],
                "absolute_log_bias_before": before_bias,
                "absolute_log_bias_after": after_bias,
                "bias_reduction_percent": reduction,
            }

        metrics = {
            "method": "masked_overlap_log_luminance_least_squares",
            "gains": {
                camera_name: float(np.exp(log_gain))
                for camera_name, log_gain in log_gains.items()
            },
            "pairs": pair_metrics,
            "summary": {
                "mean_absolute_log_bias_before": float(np.mean(biases_before)),
                "mean_absolute_log_bias_after": float(np.mean(biases_after)),
                "mean_bias_reduction_percent": float(
                    (np.mean(biases_before) - np.mean(biases_after))
                    / np.mean(biases_before)
                    * 100.0
                ),
            },
        }
        return compensated_images, metrics
