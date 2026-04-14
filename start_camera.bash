#!/bin/bash

source ./install/setup.bash
ros2 launch realsense2_camera rs_launch.py enable_rgbd:=true enable_sync:=true align_depth.enable:=true enable_color:=true enable_depth:=true enable_accel:=true enable_gyro:=true unite_imu_method:=2
