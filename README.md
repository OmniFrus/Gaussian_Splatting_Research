<div align="center">
<h1>A2: Semantic Scene Understanding for 3D Representation</h1>
[**Wesley Haverkort**](mailto:w.j.haverkort@student.rug.nl)
</div>

## Sync slides
[https://docs.google.com/presentation/d/1ioU0Bz3pH50mXnqPpxjEqHX3ZRrUq6BBKUTdcfEVIbw/edit?usp=drive_link](https://docs.google.com/presentation/d/1jmmtQW7J8PL43zMLR3hJySDBDvEeZ5zam7DdNb1mGiw/edit?slide=id.g3e5fd4c6abe_1_0#slide=id.g3e5fd4c6abe_1_0)

## Progress
Last updated: Jan 30 2026

- [ ] Replace object-centric pipeline with full-image semantic segmentation (SegFormer).
- [ ] Publish semantic segmentation outputs (`/semantic_image`).
- [ ] Extend pipeline to generate semantic point clouds.
- [ ] Integrate depth data for 3D semantic reconstruction.
- [ ] Explore advanced models (e.g., DFormer) as potential improvements.
- [ ] Prepare pipeline for Gaussian Splatting integration.
- [ ] Overhaul UI for semantic-based interaction instead of object tracking.
- [ ] Improve performance on Jetson Orin Nano by decreasing CPU usage.
- [ ] Run new performance and robustness tests.
- [ ] Update launch file and write documentation.

## Overview

This project focuses on **semantic scene understanding and 3D representation** using RGB-D data.

Instead of tracking individual objects, the system processes the **entire image** and assigns a semantic label to each pixel. These labels are then combined with depth data to construct a **semantically enriched 3D representation** of the scene.

The pipeline is designed to support:
- Semantic segmentation (2D)
- Semantic point cloud generation (3D)
- Future integration with **Gaussian Splatting** for multi-view scene reconstruction

## Setup

<ol>
	<li>[ROS2 Humble](https://docs.ros.org/en/humble/Installation.html) A LTS ROS2 Distribution for Ubuntu Jammy (22.04).</li>
	<li>
	Install [PyTorch](https://pytorch.org/get-started/locally/). If you are using a Jetson Orin Nano either install PyTorch using NVIDIA's guides or install the following wheels:

```sh
	pip install https://pypi.jetson-ai-lab.io/jp6/cu126/+f/62a/1beee9f2f1470/torch-2.8.0-cp310-cp310-linux_aarch64.whl#sha256=62a1beee9f2f147076a974d2942c90060c12771c94740830327cae705b2595fc
```

Install torchvision:

```sh
	pip install https://pypi.jetson-ai-lab.io/jp6/cu126/+f/907/c4c1933789645/torchvision-0.23.0-cp310-cp310-linux_aarch64.whl#sha256=907c4c1933789645ebb20dd9181d40f8647978e6bd30086ae7b01febb937d2d1
```

</li>
	<li>[realsense-ros](https://github.com/realsenseai/realsense-ros) A ROS wrapper for Intel® RealSense™ cameras.</li>
</li>
</ol>

## Usage

Run the following in the working directory:
```sh
colcon build --packages-select localizer
source install/setup.bash
ros2 run localizer config_ui
```

Setup your parameters and click start. Click somewhere on the image to start tracking that object (It can only track things that are trained in YOLO).

## Remote viewing
To remotely view your screen use nomachine by starting the `nxserver` service and connecting to it using the nomachine app on your mobile device.
