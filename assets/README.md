# Image Assets

This directory contains selected images used by the repository-level README. Runtime inputs and generated frame sequences remain in their module-specific directories and are excluded from Git.

## Structure

```text
assets/
└── images/
    ├── overview/                   # Optional future pipeline diagrams
    ├── calibration/                # Real-camera calibration and stitching
    ├── carla-simulation/           # CARLA setup and validation results
    └── optical-flow/               # Temporal reconstruction results
```

## Current Assets

### Calibration and Real-Camera Processing

- `raw.jpg`: four raw fisheye inputs
- `undistorted.jpg`: four corrected camera views
- `birdeye.jpg`: four homography-projected BEV views
- `surround_view_balanced.jpg`: final exposure-compensated surround view
- `real-vehicle-collection-interface.jpg`: real-vehicle four-camera acquisition and online surround-view interface

### CARLA Simulation

- `carlacar.jpg`: CARLA vehicle and environment
- `carlacamera.jpg`: four simulated camera views
- `carlabirdeye.jpg`: four projected CARLA views
- `carlasurroundview.jpg`: stitched CARLA surround view
- `carlatransparent.png`: temporal transparent-view sequence

### Optical-Flow Reconstruction

- `opticalflow.jpg`: historical alignment and center-region recovery
- `transparentbirdview.jpg`: real-camera transparent-view sequence

Use descriptive lowercase filenames for additional assets. Markdown links should remain relative to the repository root, for example:

```markdown
![Surround-view result](assets/images/calibration/surround_view_balanced.jpg)
```

For consistent display sizing in the root README:

```html
<p align="center">
  <img src="assets/images/optical-flow/opticalflow.jpg"
       width="1000"
       alt="Optical-flow reconstruction">
</p>
```
