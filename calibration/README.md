# Camera Calibration and Surround-View Preparation

This module contains the real-camera calibration and four-camera surround-view preprocessing pipeline from the undergraduate thesis project. The cleaned version retains the final calibration parameters, a small real-camera example set, and a reproducible processing pipeline. It does not depend on CARLA or Ubuntu.

## Pipeline

```text
Chessboard images → intrinsic matrix K and distortion coefficients D
Four fisheye images → undistorted camera views
Undistorted views + BEV reference image → homography H
K / D / H + four raw images → four BEV views
→ directional masks → exposure compensation → overlap blending
→ surround-view image
```

## Directory Structure

```text
calibration/
├── config/                          # Final K, D, and H parameters
│   ├── front|back|left|right/       # Runtime NPY parameters
│   └── front|back|left|right.yaml   # Human-readable OpenCV files
├── sample_data/
│   ├── raw/                         # Four real fisheye inputs
│   ├── reference/                   # Raw and centered BEV references
│   └── intrinsic/                   # Optional intrinsic dataset
├── outputs/
│   ├── undistorted/                 # Undistorted camera views
│   ├── birdseye/
│   │   ├── unmasked/                # Perspective-warped views
│   │   ├── balanced/                # Exposure-compensated masked views
│   │   └── masked/                  # Directionally masked views
│   ├── diagnostics/                 # Chessboard detection previews
│   ├── exposure_metrics.json        # Brightness metrics and gains
│   ├── surround_view.jpg            # Baseline surround view
│   └── surround_view_balanced.jpg   # Compensated surround view
├── intrinsic_calibration/
│   └── calibrate_intrinsics.py      # Pinhole/fisheye calibration
├── hmatrix/
│   └── estimate_homography.py       # Chessboard homography estimation
├── center_reference_image.py        # Interactive reference alignment
├── exposure_compensation.py         # Adaptive exposure compensation
├── export_camera_parameters.py      # NPY-to-YAML export
├── generate_surround_view.py        # Main four-camera pipeline
├── undistort_camera.py              # Single-camera undistortion
└── README.md
```

## Installation

From the repository root:

```bash
python -m pip install -r requirements/base.txt
```

## Quick Start

Run the following commands from this directory.

Generate the four BEV views, baseline surround view, and exposure-compensated surround view:

```bash
python generate_surround_view.py
```

Generated files include:

- `outputs/birdseye/unmasked/unmasked_bev_<camera>.jpg`
- `outputs/birdseye/balanced/balanced_bev_<camera>.jpg`
- `outputs/birdseye/masked/masked_bev_<camera>.jpg`
- `outputs/surround_view.jpg`
- `outputs/surround_view_balanced.jpg`
- `outputs/exposure_metrics.json`

Generated outputs are excluded from Git. Copy only selected presentation results to the repository-level `assets/images/calibration/` directory.

Disable exposure compensation and generate only the baseline result:

```bash
python generate_surround_view.py --exposure-compensation off
```

Use the original overlap-based compensation method:

```bash
python generate_surround_view.py --exposure-compensation overlap
```

## Adaptive Exposure Compensation

The default method does not use fixed per-camera gains. It first applies the directional masks, excluding regions outside each camera's assigned field. It then measures LAB luminance over all valid pixels and uses the geometric mean of the four camera luminances as a shared target:

```text
target = mean(mean(log(L_i + 1)))
log(gain_i) = target - mean(log(L_i + 1))
```

Pixels outside the directional mask, very dark pixels, near-saturated pixels, and invalid black regions do not contribute to the estimate. The dynamic gain is applied only to the LAB L channel. Log-domain statistics reduce the influence of a small number of very bright pixels.

Per-camera luminance values, dynamic gains, and before-and-after metrics are written to `outputs/exposure_metrics.json`.

## Overlap Blending

The directional masks retain overlapping regions near the image corners. These pixels are not directly added. Distance-based feathering assigns a larger weight to pixels deeper inside a camera mask and a smaller weight near its boundary. Normalized weights are used to blend overlap regions, while non-overlapping regions retain their single-camera values. The central uncovered vehicle region remains black.

## Tools

Undistort one camera image:

```bash
python undistort_camera.py --camera-name front
```

The result is written to `outputs/undistorted/undistorted_front.jpg`. Valid camera names are `front`, `back`, `left`, and `right`.

Interactively center the BEV reference image:

```bash
python center_reference_image.py
```

The tool reads `sample_data/reference/birdseye_reference_raw.jpg` and updates `birdseye_reference.jpg` after confirmation.

Export NPY parameters to OpenCV YAML:

```bash
python export_camera_parameters.py
```

Re-estimate the homographies:

```bash
python hmatrix/estimate_homography.py
```

This command overwrites `config/<camera>/camera_<camera>_H.npy`. The small chessboards in the archived left and right examples are not detected reliably by the current OpenCV version. Use the retained homographies for normal demonstrations and create a backup before recalibration.

Run intrinsic calibration again:

```bash
python intrinsic_calibration/calibrate_intrinsics.py \
  --input-type image \
  --input-path ./sample_data/intrinsic/
```

A complete intrinsic calibration requires at least 15 valid chessboard images. The full original dataset is not included in the public example data.

## Parameter Files

- `camera_<camera>_K.npy`: intrinsic camera matrix
- `camera_<camera>_D.npy`: fisheye distortion coefficients
- `camera_<camera>_H.npy`: image-to-BEV homography
- `config/<camera>.yaml`: human-readable copy of the corresponding NPY data

The Python tools load NPY files at runtime. YAML files are retained for inspection and interoperability with other OpenCV applications.

## Verification

- All source comments and identifiers use English.
- Unicode-safe image I/O supports non-ASCII parent directories.
- All seven Python files pass syntax and command-line entry checks.
- The four masked BEV views are pixel-identical to the retained archived results.
- The original direct-add pipeline was checked pixel-by-pixel against the archived `surround2.jpg` result.
- The current pipeline adds distance-weighted overlap blending and adaptive exposure compensation.
- YAML and NPY copies of K, D, and H contain identical values.

## Limitations

- Parameters are tied to the original cameras, mounting geometry, and `1280 × 1024` resolution.
- Replacing or repositioning a camera requires recalculating K, D, and H.
- The current stitcher uses fixed directional masks and distance-based feathering without dynamic extrinsic correction.
- This is a cleaned research implementation, not a general-purpose calibration package.

## Third-Party Source

The intrinsic calibration and part of the BEV implementation were derived from or modified from `dyfcalid/CameraCalibration`. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before publication or redistribution.
