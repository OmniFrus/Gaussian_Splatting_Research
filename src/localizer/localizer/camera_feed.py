import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
import cv2
from cv_bridge import CvBridge
from PIL import Image as PILImage
from PIL import ImageTk


class CameraFeed(Node):
    def __init__(self):
        super().__init__("camera_feed")
        self.bridge = CvBridge()
        self.latest_image = None
        self.latest_image_tk = None
        
        self.image_sub = self.create_subscription(
            Image,
            "/detection",
            self.image_callback,
            10)
        
        self.click_pub = self.create_publisher(
            PointStamped,
            "/clicked_point",
            10)
        
        self.get_logger().info("Camera feed initialized")
        
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            self.latest_image = cv_image
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")
    
    def get_image_for_tkinter(self, max_width=None, max_height=None):
        if self.latest_image is None:
            return None
        
        if max_width is not None or max_height is not None:
            height, width = self.latest_image.shape[:2]
            
            if max_width is not None and width > max_width:
                scale = max_width / width
                new_width = max_width
                new_height = int(height * scale)
            else:
                scale = 1.0
                new_width = width
                new_height = height
            
            if max_height is not None and new_height > max_height:
                scale = max_height / new_height
                new_width = int(new_width * scale)
                new_height = max_height
            
            if scale != 1.0:
                resized = cv2.resize(self.latest_image, (new_width, new_height))
            else:
                resized = self.latest_image.copy()
        else:
            resized = self.latest_image.copy()
        
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb_image)
        photo = ImageTk.PhotoImage(image=pil_image)
        
        self.latest_image_tk = photo
        return photo
    
    def handle_click(self, x, y, display_width, display_height):
        if self.latest_image is None:
            return
        
        actual_height, actual_width = self.latest_image.shape[:2]
        
        scale_x = actual_width / display_width
        scale_y = actual_height / display_height
        
        actual_x = int(x * scale_x)
        actual_y = int(y * scale_y)
        
        self.get_logger().info(f"Clicked at: ({actual_x}, {actual_y})")
        
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "image"
        msg.point.x = float(actual_x)
        msg.point.y = float(actual_y)
        msg.point.z = 0.0
        self.click_pub.publish(msg)

def start_ros_spin(node):
    rclpy.spin(node)
