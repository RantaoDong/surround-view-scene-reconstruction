# CARLA Surround-View Simulation

This module organizes the CARLA simulation code from the thesis project. Four vehicle-mounted large-FOV RGB cameras generate an online bird's-eye-view surround image, and motion-aligned history is used to recover the central vehicle region. The cleaned implementation retains the original camera layout, stitching transforms, masks, optical-flow parameters, and four-frame transform accumulation logic.

## Directory Structure

```text
carla_simulation/
├── carla_bev/
│   ├── collect_camera_sequence.py  # Four-camera raw sequence collection
│   ├── run_surround_view.py        # Online surround-view BEV
│   ├── run_transparent_view.py     # Online transparent-vehicle BEV
│   ├── sensor_setup.py             # Vehicle and camera creation
│   ├── sync_mode.py                # CARLA multi-sensor synchronization
│   ├── image_conversion.py         # CARLA/OpenCV/Pygame conversion
│   ├── bev_stitching.py            # Alignment, masking, and stitching
│   └── temporal_filling.py         # Four-frame temporal recovery
├── outputs/
│   ├── raw/                        # Raw four-camera sequences
│   ├── surround/                   # Standard surround-view results
│   └── transparent/                # Transparent-view results
└── README.md
```

Generated output files are excluded from Git. Copy selected presentation images to the repository-level `assets/images/` directory.

## Simulation Setup

The simulation uses four `sensor.camera.rgb` actors:

- Resolution: `2000 × 2000`
- Field of view: `170°`
- Camera pitch: `-90°`
- Front and rear offsets: `±2.30 m`
- Left and right offsets: `±1.70 m`
- Camera height: `1.0 m`

All four cameras point directly toward the ground. Fixed rotations and translations align the images, and front, rear, left, and right masks compose a `2000 × 2000` BEV. This simulation path does not use the fisheye undistortion or image-to-ground homographies from the real-camera calibration module.

## Transparent-View Method

```text
Synchronized four-camera images
→ surround-view BEV stitching
→ Shi–Tomasi + LK flow between neighboring BEV frames
→ affine motion estimation
→ accumulation of four motion transforms
→ historical BEV alignment
→ central vehicle-region recovery
```

The implementation intentionally preserves the behavior of the final readable experiment script. See the repository history or project notes before changing transform order, failure behavior, or center-region compositing.

## Environment

The code requires a running CARLA Server. The CARLA Python API must match the server version and be importable as `carla` in the selected Python environment. CARLA itself and its Python API are not included in this repository.

From the repository root, install the remaining dependencies:

```bash
python -m pip install -r requirements/carla.txt
```

## Running the Simulation

Run the following commands from this directory after starting CARLA Server.

Collect raw four-camera sequences:

```bash
python -m carla_bev.collect_camera_sequence
```

Generate an online standard surround-view BEV:

```bash
python -m carla_bev.run_surround_view
```

Generate an online transparent-vehicle BEV:

```bash
python -m carla_bev.run_transparent_view
```

All three entry points accept `--host`, `--port`, and `--output-dir`. Online entry points also accept `--max-frames`. With the default value, processing continues until the Pygame window is closed, Esc is pressed, or `q` is pressed in the OpenCV result window.

Examples:

```bash
python -m carla_bev.run_surround_view --max-frames 300
```

```bash
python -m carla_bev.run_transparent_view \
  --host localhost \
  --port 2000 \
  --output-dir outputs/transparent
```

## Module Responsibilities

- `sensor_setup.py` creates the Tesla Model 3 and the four rigidly attached cameras.
- `sync_mode.py` advances the world in synchronous mode and returns one frame from every sensor.
- `image_conversion.py` converts CARLA BGRA buffers to OpenCV BGR images and Pygame previews.
- `bev_stitching.py` applies the archived camera transforms and directional masks.
- `temporal_filling.py` estimates inter-frame motion and fills the center rectangle from historical BEV data.
- `runtime.py` manages the online loop, waypoint updates, display windows, and output files.

## Limitations

- CARLA cameras are large-FOV perspective RGB cameras, not a strict fisheye model.
- Stitching transforms and seam filters are tied to the current `2000 × 2000` configuration.
- The vehicle follows waypoints through direct pose updates rather than vehicle dynamics.
- The temporal recovery implementation retains the original experiment's transform accumulation and center compositing behavior.
- CARLA and the original Ubuntu runtime are not required to inspect the source, but they are required for online execution.
