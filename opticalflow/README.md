# Optical-Flow BEV Inpainting

This module uses motion-aligned historical BEV frames to recover vehicle-occluded or dark missing regions in a real-camera surround-view sequence. The input is expected to have already passed through undistortion, BEV projection, and four-camera stitching. CARLA is not required.

## Directory Structure

```text
opticalflow/
├── sample_data/
│   └── bev_sequence/               # Local BEV input sequence
├── outputs/
│   └── filled/                     # Generated inpainted sequence
├── optical_flow_inpainting.py      # CLI and sequence orchestration
├── image_sequence.py               # Scanning, sorting, and image I/O
├── motion_estimation.py            # Feature tracking and motion estimation
├── bev_inpainting.py               # History alignment and region recovery
└── README.md
```

The entry script depends on the sequence, motion-estimation, and BEV-inpainting modules. Algorithm modules do not import the command-line entry point, and the module graph contains no circular imports.

## Input Requirements

Images placed in `sample_data/bev_sequence/` should meet the following requirements:

- All frames use the same BEV coordinate system.
- All frames have the same dimensions and channel count.
- Filenames contain ordered frame numbers, such as `frame_0001.jpg`.
- The temporal gap between neighboring source frames is not excessively large.

JPG, JPEG, PNG, BMP, and TIFF files are supported. Files are naturally sorted by the numeric parts of their names. Recursive scanning preserves input subdirectory paths in the output directory.

Local input images and generated outputs are excluded from Git. Final presentation media should be copied to the repository-level `assets/` directory.

## Method

```text
BEV frames sampled at a fixed interval
→ Shi–Tomasi feature detection
→ pyramidal Lucas–Kanade optical flow
→ original experiment region filtering
→ LMEDS partial affine estimation
→ alignment of the previous sampled result
→ missing-region update
```

By default, the program samples one image every six frames and only writes results for sampled frames. Motion is estimated between the original images at consecutive sampled times, preventing recovered content from affecting feature tracking. Region recovery uses the previous sampled inpainting result, allowing road information to accumulate through a single temporal chain. If motion estimation fails, the current source frame is retained and processing continues.

## Installation

From the repository root:

```bash
python -m pip install -r requirements/base.txt
```

## Running the Pipeline

Run the following commands from this directory.

After placing images in `sample_data/bev_sequence/`:

```bash
python optical_flow_inpainting.py
```

With the default interval, a sequence named `frame_0001`, `frame_0002`, and so on produces `frame_0001`, `frame_0007`, `frame_0013`, and subsequent sampled results in `outputs/filled/`. Unsampled frames do not produce output files.

Specify custom input and output directories:

```bash
python optical_flow_inpainting.py <input_dir> <output_dir>
```

Example:

```bash
python optical_flow_inpainting.py D:/data/bev D:/data/bev_filled
```

The input and output directories must be different.

## Missing-Region Selection

The default `dark` mode treats pixels at or below the selected luminance threshold as missing:

```bash
python optical_flow_inpainting.py --fill-mode dark --dark-threshold 35
```

This mode is close to the original experiment, but naturally dark road regions can also be classified as missing. Adjust the threshold for the black region produced by the selected BEV mask.

The `center` mode replaces only the configured central vehicle area:

```bash
python optical_flow_inpainting.py \
  --fill-mode center \
  --center-width-ratio 0.28 \
  --center-height-ratio 0.68
```

The width and height values are ratios relative to the full BEV image. Prefer `dark` when the missing area is already represented by a black mask. Use `center` when a fixed transparent-vehicle region is required.

## Sequence Selection

Process one image every six frames:

```bash
python optical_flow_inpainting.py --frame-step 6
```

The first sampled image initializes the temporal chain. Every subsequent sampled image uses the previous sampled inpainting result.

Use a different interval:

```bash
python optical_flow_inpainting.py --frame-step 5
```

Process naturally sorted images with indices 100 through 299:

```bash
python optical_flow_inpainting.py --start-index 100 --end-index 300
```

Scan input subdirectories:

```bash
python optical_flow_inpainting.py --recursive
```

## Motion Estimation

The default robust estimator is LMEDS, matching the later original experiment:

```bash
python optical_flow_inpainting.py --estimator lmeds
```

RANSAC is also available:

```bash
python optical_flow_inpainting.py --estimator ransac
```

The default feature region scales the original `600 × 600` coordinates: 80 pixels are excluded on the left and right, 10 pixels are excluded at the top and bottom, and the central vertical strip from `x=220` to `x=380` is excluded. Adjust the corresponding ratios with `--horizontal-margin-ratio`, `--vertical-margin-ratio`, and `--feature-center-width-ratio`.

The original implementation used only the forward LK status. Enable an additional forward-backward consistency check with:

```bash
python optical_flow_inpainting.py --forward-backward-check
```


