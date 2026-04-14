from launch import LaunchDescription
from launch_ros.actions import Node

params = {
    'enable_rgbd': True,
    'enable_sync': True,
    'align_depth.enable': True,
    'enable_color': True,
    'enable_depth': True,
}

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="realsense2_camera",
            executable='realsense2_camera_node',
            name='camera',
            parameters=[params]
        ),
        Node(
            package="localizer",
            executable="save_images",
            name="subscriber"
        )
    ])