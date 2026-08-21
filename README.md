# Surround-View Scene Reconstruction for Parking Assistance

An undergraduate thesis project on panoramic perception and hidden-road reconstruction for an automotive surround-view parking assistance system. The repository is organized into three modules that match the project directories:

1. `calibration/` — real-camera calibration, bird's-eye-view transformation, and four-camera stitching
2. `opticalflow/` — temporal road-surface reconstruction from real-camera BEV sequences
3. `carla_simulation/` — surround-view and transparent-vehicle validation in CARLA

<!-- <p align="center">
  <img src="assets/images/calibration/surround_view_balanced.jpg"
       width="420"
       alt="Final real-camera surround-view result">
</p> -->

## Project Workflow

```text
Real-camera pipeline
Fisheye images
-> intrinsic calibration and undistortion
-> ground-plane perspective transformation
-> masking, exposure compensation, and overlap blending
-> stitched bird's-eye view
-> optical-flow alignment of historical frames
-> reconstructed road surface beneath the vehicle

Simulation pipeline
Synchronized CARLA cameras
-> camera alignment and surround-view stitching
-> inter-frame motion estimation
-> historical BEV alignment
-> transparent-vehicle reconstruction
```

The real-camera processing is divided between `calibration/` and `opticalflow/`. The CARLA module is a separate simulation and validation path rather than the source of the real-camera data.

## 1. Camera Calibration and Surround-View Generation

The [`calibration/`](calibration/) module contains the complete real-camera preprocessing pipeline. Four fisheye cameras provide front, rear, left, and right views. Intrinsic calibration estimates the camera matrix `K` and distortion coefficients `D`, while a reference bird's-eye image is used to estimate the homography `H` for each camera.

### Calibration and undistortion

| Raw fisheye inputs | Undistorted views |
|---|---|
| <img src="assets/images/calibration/raw.jpg" alt="Four raw fisheye images"> | <img src="assets/images/calibration/undistorted.jpg" alt="Four undistorted images"> |

The module supports pinhole and fisheye intrinsic calibration. The public version retains the final NPY parameters used by the scripts and matching human-readable OpenCV YAML files. A complete original chessboard dataset is not included.

### BEV transformation and stitching

Each undistorted image is projected into a shared ground-plane coordinate system. Directional masks select valid regions, adaptive exposure compensation reduces brightness differences, and distance-weighted feathering blends camera overlaps.

| Four projected BEV views | Exposure-compensated surround view |
|---|---|
| <img src="assets/images/calibration/birdeye.jpg" alt="Four BEV projections"> | <img src="assets/images/calibration/surround_view_balanced.jpg" alt="Exposure-compensated surround view"> |

The black center region represents the road area hidden by the vehicle. This stitched BEV is the input expected by the optical-flow module.

### Real-vehicle acquisition interface

<p align="center">
  <img src="assets/images/calibration/real-vehicle-collection-interface.jpg"
       width="1000"
       alt="Real-vehicle four-camera collection and surround-view interface">
</p>

This interface shows the four physical camera streams and generated surround view running on the original Ubuntu vehicle platform. It is a real-vehicle data-acquisition interface, not CARLA.

See the [calibration module documentation](calibration/README.md) for parameter files, exposure compensation, blending, recalibration, and execution details.

## 2. Optical-Flow Road Reconstruction

The [`opticalflow/`](opticalflow/) module processes a time-ordered sequence of stitched real-camera BEV images. It samples one frame every six source frames, tracks Shi–Tomasi features with pyramidal Lucas–Kanade optical flow, estimates affine motion, aligns the previous reconstructed result, and fills the missing road region in the current frame.

<p align="center">
  <img src="assets/images/optical-flow/opticalflow.jpg"
       width="1000"
       alt="Optical-flow road-surface reconstruction process">
</p>

The following sequence illustrates recovered road content as the vehicle moves through the scene:

<p align="center">
  <img src="assets/images/optical-flow/transparentbirdview.jpg"
       width="700"
       alt="Real-camera transparent under-vehicle sequence">
</p>

Motion is estimated only from original sampled frames so that recovered pixels do not affect feature tracking. The previous sampled output carries historical road information forward through the sequence. Missing-region selection supports either dark-pixel detection or a fixed center region.

See the [optical-flow module documentation](opticalflow/README.md) for input naming, frame sampling, region selection, and command-line options.

## 3. CARLA Surround-View Simulation

The [`carla_simulation/`](carla_simulation/) module is an independent simulation branch. It creates a vehicle-mounted four-camera rig, synchronizes the sensors, generates an online surround-view image, and aligns historical BEV content to reconstruct the central vehicle region.

| CARLA vehicle environment | Four camera views |
|---|---|
| <img src="assets/images/carla-simulation/carlacar.jpg" alt="CARLA vehicle environment"> | <img src="assets/images/carla-simulation/carlacamera.jpg" alt="Four CARLA camera views"> |

| Projected CARLA views | Stitched CARLA surround view |
|---|---|
| <img src="assets/images/carla-simulation/carlabirdeye.jpg" alt="Four projected CARLA camera views"> | <img src="assets/images/carla-simulation/carlasurroundview.jpg" alt="CARLA surround-view result"> |

The transparent-view experiment accumulates inter-frame transformations and aligns historical content with the current vehicle region:

<p align="center">
  <img src="assets/images/carla-simulation/carlatransparent.png"
       width="900"
       alt="CARLA transparent-view sequence">
</p>

This path uses large-FOV RGB cameras directed toward the ground. It does not use the fisheye parameters or homographies from the real-camera calibration module.

See the [CARLA module documentation](carla_simulation/README.md) for the camera setup, synchronization, runtime commands, and implementation limitations.

## Repository Structure

```text
.
├── calibration/                    # Real-camera calibration and stitching
├── opticalflow/                    # Real-camera temporal BEV reconstruction
├── carla_simulation/               # CARLA simulation and validation
├── requirements/
│   ├── base.txt                    # NumPy and OpenCV
│   └── carla.txt                   # Base dependencies and Pygame
├── assets/
│   └── images/                     # Selected presentation results
├── LICENSE
└── README.md
```

| Module | Main responsibilities | Documentation |
|---|---|---|
| `calibration` | Estimate `K`, `D`, and `H`; undistort images; project BEV views; compensate exposure; blend and stitch | [README](calibration/README.md) |
| `opticalflow` | Sample real-camera BEV frames; estimate motion; align history; recover the hidden road region | [README](opticalflow/README.md) |
| `carla_simulation` | Create the simulated camera rig; synchronize sensors; generate standard and transparent BEV results | [README](carla_simulation/README.md) |

## Installation

Python 3.8 or later is recommended.

Install dependencies for the real-camera calibration and optical-flow modules:

```bash
python -m pip install -r requirements/base.txt
```

Install the additional CARLA visualization dependency:

```bash
python -m pip install -r requirements/carla.txt
```

The CARLA Python API is not bundled with this repository. It must match the CARLA Server version and be importable as `carla` in the selected Python environment.

## Quick Start

Generate a surround view from the included real-camera examples:

```bash
cd calibration
python generate_surround_view.py
```

Process a local stitched-BEV sequence with optical-flow reconstruction:

```bash
cd opticalflow
python optical_flow_inpainting.py
```

Run the transparent-view simulation after starting CARLA Server:

```bash
cd carla_simulation
python -m carla_bev.run_transparent_view
```

Detailed parameters and additional commands are documented in each module README.

## Data and Reproducibility

- The calibration module retains final camera parameters and a compact real-camera example set.
- Optical-flow source sequences and generated outputs remain local and are excluded from Git.
- CARLA raw frames and generated output sequences are excluded from Git.
- Images under `assets/` are selected presentation results rather than runtime dependencies.
- CARLA and the original Ubuntu environment are not required to inspect the implementation.

## Limitations

- Calibration parameters are tied to the original cameras, mounting geometry, and image resolution.
- Temporal reconstruction assumes a mostly planar, static road surface.
- Dynamic objects and non-planar structures can introduce optical-flow errors.
- The CARLA branch uses large-FOV perspective RGB cameras rather than a strict fisheye camera model.
- The real-camera and CARLA branches are related experiments, not one fully automated cross-platform runtime.

## Project Background

This repository is a cleaned portfolio version of the undergraduate thesis *Research on Scene Reconstruction Technology for a Panoramic Parking Assistance System*. Historical experiments, corrupted source files, large datasets, and generated runtime media are intentionally excluded.

## Third-Party Notice

Part of the calibration implementation was derived from `dyfcalid/CameraCalibration`. See [THIRD_PARTY_NOTICES.md](calibration/THIRD_PARTY_NOTICES.md) before publishing or redistributing the complete repository.
