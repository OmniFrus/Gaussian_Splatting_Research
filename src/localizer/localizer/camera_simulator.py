import rclpy
from rclpy.node import Node
from realsense2_camera_msgs.msg import RGBD

import cv2
import cv_bridge

INPUT_IMAGE = "src/segmentation/segmentation/test.jpg"
bridge = cv_bridge.CvBridge()

class CameraSimulator(Node):
    def __init__(self):
        super().__init__('camera_simulator')
        self.publisher_ = self.create_publisher(RGBD, '/camera/camera/rgbd', 10)
        timer_period = 1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        raw_image = cv2.imread(INPUT_IMAGE)
        msg = bridge.cv2_to_imgmsg(raw_image, encoding="passthrough")

        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    camera_simulator = CameraSimulator()
    rclpy.spin(camera_simulator)
    camera_simulator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()