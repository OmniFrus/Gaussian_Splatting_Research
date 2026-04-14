<div align="center">
<h1>A2: Foundation Models for 3D Object Tracking (FMObject)</h1>
[**Wesley Haverkort**](mailto:w.j.haverkort@student.rug.nl)
[**Lars Hidding**](mailto:l.hidding.1@student.rug.nl)
</div>

## Sync slides
https://docs.google.com/presentation/d/1ioU0Bz3pH50mXnqPpxjEqHX3ZRrUq6BBKUTdcfEVIbw/edit?usp=drive_link

## Progress
Last updated: Jan 30 2026


- [ ] Switch between pointcloud algorithms.
- [ ] Add pre- and post-processing to segmentation step.
- [ ] Explore other tracking methods.
- [ ] Overhaul UI
- [ ] Improve performance on Jetson Orin Nano by decreasing CPU usage.
- [ ] Run new performance tests.
- [ ] Update launch file and write documentation.
- [ ] Implement reading of IMU data.

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

## Performance

### Performance on lab computers (Cuda 13) [Nov 24 2025]
| **Duration per step**         | **Mean duration** |
|-------------------------------|-------------------|
| Total time elapsed            |        1.2607 sec |
| Depth Anything                |        0.7893 sec |
| MobileSAM                     |        0.3694 sec |
| MobileSAM (without set_image) |        0.0239 sec |
| Others                        |        0.1020 sec |

### Performance on Jetson Orin Nano (Cuda 12.6) [Dec 4 2025]
| **Duration per step**         | **Mean duration** |
|-------------------------------|-------------------|
| Total time elapsed            |             - sec |
| Depth Anything                |             - sec |
| MobileSAM                     |             - sec |
| MobileSAM (without set_image) |             - sec |
| Others                        |             - sec |

*More performance details coming soon...*

## Remote viewing
To remotely view your screen use nomachine by starting the `nxserver` service and connecting to it using the nomachine app on your mobile device.
