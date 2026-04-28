import json

import cv2
import cv_bridge
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from realsense2_camera_msgs.msg import RGBD
from realsense2_camera_msgs.msg import IMUInfo
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs
from .sam3_wrapper import SAM3Wrapper
import traceback
import torch
from . import pointcloud
from . import marker
# import pointcloud
# import marker
from visualization_msgs.msg import Marker

qos_profile = QoSProfile(
    depth=2,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    durability=DurabilityPolicy.VOLATILE
)


class CameraSubscriber(Node):
    def __init__(self):
        self.processing = False
        self.frame_count = 0
        super().__init__("localizer_3D")
        self.sam3 = SAM3Wrapper(default_prompt="chair", resolution=256)
        self.current_prompt = "chair"
        
        self.declare_parameter('confidence', 0.2)
        self.declare_parameter('num_points', 10000)
        self.declare_parameter('sam3_run_every_n_frames', 4)
        
        self.confidence = self.get_parameter('confidence').get_parameter_value().double_value
        self.num_points = self.get_parameter('num_points').get_parameter_value().integer_value
        self.sam3_run_every_n_frames = self.get_parameter('sam3_run_every_n_frames').get_parameter_value().integer_value
        if self.sam3_run_every_n_frames < 1:
            self.sam3_run_every_n_frames = 1

        self.track_id = -1
        self.objects = []
        self.clicked_point = None

        self.position = [0,0,0]
        self.last_mask = None
        self.last_box = None
        self.last_score = None
        self.last_sam3_frame = 0
        
        self.bridge = cv_bridge.CvBridge()

        self.rgbd_subscription = self.create_subscription(
            RGBD, 
            '/camera/camera/rgbd',
            self.rgbd_callback,
            qos_profile
        )
        
        self.selected_object_subscription = self.create_subscription(
            std_msgs.Int16, 
            '/select_by_tracker_id',
            self.select_by_tracker_id,
            10
        )

        self.num_points_subscription = self.create_subscription(
            std_msgs.Int32, 
            '/config/num_points',
            self.set_num_points,
            10
        )

        # self.confidence_subscription = self.create_subscription(
        #     std_msgs.Int16, 
        #     '/select_by_tracker_id',
        #     self.select_by_tracker_id,
        #     10
        # )

        self.selected_object_subscription = self.create_subscription(
            std_msgs.String, 
            '/select_by_class_name',
            self.select_by_class_name,
            10
        )
        
        self.clicked_point_subscription = self.create_subscription(
            PointStamped,
            '/clicked_point',
            self.clicked_point_callback,
            10
        )
        
        self.detection_publisher = self.create_publisher(
            Image,
            '/detection',
            10
        )

        self.mask_publisher = self.create_publisher(
            Image,
            '/mask',
            10
        )

        self.pointcloud_publisher = self.create_publisher(
            sensor_msgs.PointCloud2,
            '/pointcloud',
            10
        )

        self.marker_publisher = self.create_publisher(
            Marker,
            '/object_pose_marker',
            10
        )
        self.get_logger().info(
            f"CameraSubscriber initialized | prompt={self.current_prompt} | "
            f"sam3_processor_resolution={self.sam3.resolution} | "
            f"sam3_run_every_n_frames={self.sam3_run_every_n_frames}"
        )

    def set_num_points(self, msg):
        self.num_points = msg.data

    def select_by_tracker_id(self, msg):
        self.track_id = msg.data

    def select_by_class_name(self, msg):
        prompt = msg.data.strip()

        if prompt == "" or prompt.lower() == "none":
            self.current_prompt = "chair"
        else:
            self.current_prompt = prompt

        self.sam3.set_prompt(self.current_prompt)
        self.get_logger().info(f"SAM3 prompt set to: {self.current_prompt}")
    
    def clicked_point_callback(self, msg):
        self.clicked_point = (int(msg.point.x), int(msg.point.y))
        self.get_logger().info(f"Received clicked point: {self.clicked_point}")

    def _estimate_3d_pose(self, points_array):
        if len(points_array) < 150:
            return None
        
        P = points_array[:, 0:3].astype(np.float64)
        
        centroid = P.mean(axis=0)
        Q = P - centroid
        U, S, Vt = np.linalg.svd(Q, full_matrices=False)
        
        pca_x = marker.normalize(Vt[0])
        pca_y = marker.normalize(Vt[1])
        normal = marker.normalize(Vt[2])
        
        view_dir = marker.normalize(centroid)
        if np.dot(normal, view_dir) > 0.0:
            normal = -normal
        
        pca_y = marker.normalize(np.cross(normal, pca_x))
        pca_x = marker.normalize(np.cross(pca_y, normal))
        
        x_axis = normal
        world_up = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(world_up, x_axis)) > 0.95:
            world_up = np.array([0.0, 0.0, -1.0])
        
        y_axis = marker.normalize(np.cross(world_up, x_axis))
        if np.linalg.norm(y_axis) < 0.1:
            world_up = np.array([1.0, 0.0, 0.0])
            y_axis = marker.normalize(np.cross(world_up, x_axis))
        
        z_axis = marker.normalize(np.cross(x_axis, y_axis))
        R = np.column_stack((x_axis, y_axis, z_axis))
        
        quaternion = marker.rotation_matrix_to_quaternion(R)
        
        return (centroid, quaternion)


    def create_pointcloud(self, color_img, depth_img, num_points, original_img_size, offset):
        h, w = depth_img.shape[:2]
        granularity = int(np.sqrt((w*h)/num_points))
        if granularity < 1: granularity = 1

        num_points = min(num_points, np.count_nonzero(depth_img))

        points = np.zeros((num_points, 6))
        i = 0
        for x in range(0, w, granularity):
            for y in range(0, h, granularity):
                depth = depth_img[y][x]
                if depth == 0: continue
                x_pos = (((float(x)+offset[0])/original_img_size[0]) - 0.5)* depth
                y_pos = -(((float(y)+offset[1])/original_img_size[1]) - 0.5) * original_img_size[1]/original_img_size[0] * depth
                color = color_img[y][x]
                points[i] = [x_pos, depth, y_pos, *color]
                i += 1
                if i == num_points:
                    break
            else:
                continue
            break

        points.resize((i, 6))
        return points

    def create_pointcloud_adaptive(self, color_img, depth_img, num_points, original_img_size, offset):
        h, w = depth_img.shape[:2]
        hp, wp = h/original_img_size[1], w/original_img_size[0]
        step = max(1, int(np.count_nonzero(depth_img) / num_points))

        points = np.zeros((num_points, 6))
        points_found = 0
        points_stored = 0

        non_zero_indices = np.nonzero(depth_img)
        non_zero_indices = zip(non_zero_indices[0], non_zero_indices[1])

        for y, x in list(non_zero_indices):
            depth = depth_img[y][x]
            points_found += 1
            if (points_found-1) % step != 0:
                continue
            
            x_pos = ((float(x) + offset[0])/original_img_size[0] -.5) * depth
            y_pos = -((float(y) +  offset[1])/original_img_size[1] -.5) * depth *(original_img_size[1]/original_img_size[0])
            color = color_img[y][x]
                
            points[points_stored] = [x_pos, depth, y_pos, *color]
            
            points_stored += 1
            if points_stored == num_points:
                break
        
        points.resize((points_stored, 6))
        return points
    
    def rgbd_callback(self, msg):
        if self.processing:
            self.get_logger().debug("Skipping frame because previous frame is still processing")
            return

        self.processing = True
        self.frame_count += 1
        self.get_logger().info(f"RGBD callback received, frame {self.frame_count}")

        try:
            self.get_logger().info(f"Frame {self.frame_count}: converting RGB image")
            image = self.bridge.imgmsg_to_cv2(msg.rgb, desired_encoding="rgb8")
            color_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            detection_color_img = np.copy(color_img)

            self.get_logger().info(f"Frame {self.frame_count}: converting depth image")
            depth_img = self.bridge.imgmsg_to_cv2(msg.depth, desired_encoding="16UC1")
            depth_img = (depth_img / 1000.0) * 2

            original_h, original_w = color_img.shape[:2]
            self.get_logger().info(
                f"Frame {self.frame_count}: color_shape={color_img.shape}, "
                f"depth_shape={depth_img.shape}, depth_nonzero={int(np.count_nonzero(depth_img))}"
            )

            run_sam3_this_frame = (self.frame_count % self.sam3_run_every_n_frames) == 0 or self.last_mask is None
            if run_sam3_this_frame:
                self.get_logger().info(f"Running SAM3 on frame {self.frame_count} with prompt '{self.current_prompt}'")
                mask, box, score = self.sam3.best_mask(color_img, self.current_prompt)
                self.last_mask = None if mask is None else mask.copy()
                self.last_box = None if box is None else box.copy()
                self.last_score = score
                self.last_sam3_frame = self.frame_count
                self.get_logger().info(
                    f"SAM3 done on frame {self.frame_count}: mask_found={mask is not None}, score={score}"
                )
            else:
                mask = None if self.last_mask is None else self.last_mask.copy()
                box = None if self.last_box is None else self.last_box.copy()
                score = self.last_score
                self.get_logger().info(
                    f"Frame {self.frame_count}: skipping SAM3, reusing result from frame {self.last_sam3_frame}"
                )

            if mask is None:
                self.get_logger().warn(f"No SAM3 mask found for prompt: {self.current_prompt}")
                img_msg = self.bridge.cv2_to_imgmsg(detection_color_img, encoding="bgr8")
                self.detection_publisher.publish(img_msg)
                return

            if mask.shape[:2] != depth_img.shape[:2]:
                self.get_logger().warn(
                    f"Frame {self.frame_count}: mask/depth mismatch mask_shape={mask.shape} "
                    f"depth_shape={depth_img.shape}, resizing mask"
                )
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (depth_img.shape[1], depth_img.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

            self.get_logger().info(
                f"Frame {self.frame_count}: mask_shape={mask.shape}, mask_nonzero={int(np.count_nonzero(mask))}"
            )
            mask_vis = (mask * 255).astype(np.uint8)
            mask_msg = self.bridge.cv2_to_imgmsg(mask_vis, encoding="mono8")
            self.mask_publisher.publish(mask_msg)

            detection_color_img[mask == 1] = (
                0.6 * detection_color_img[mask == 1] + 0.4 * np.array([0, 255, 0])
            ).astype(np.uint8)

            tracked_position = (0, 0)

            if box is not None:
                x1, y1, x2, y2 = [int(v) for v in box]
                self.get_logger().info(
                    f"Frame {self.frame_count}: box=({x1},{y1},{x2},{y2}), score={score}"
                )
                cv2.rectangle(detection_color_img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(
                    detection_color_img,
                    f"{self.current_prompt}: {score:.2f}" if score is not None else self.current_prompt,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 255),
                    2,
                )
                tracked_position = (x1, y1)

            img_msg = self.bridge.cv2_to_imgmsg(detection_color_img, encoding="bgr8")
            self.detection_publisher.publish(img_msg)

            masked_depth = depth_img.copy()
            masked_depth[mask == 0] = 0
            self.get_logger().info(
                f"Frame {self.frame_count}: masked_depth_nonzero={int(np.count_nonzero(masked_depth))}"
            )

            points = self.create_pointcloud(
                color_img,
                masked_depth,
                self.num_points,
                (original_w, original_h),
                tracked_position,
            )
            self.get_logger().info(
                f"Frame {self.frame_count}: generated_pointcloud_points={len(points)}"
            )

            if len(points) > 0:
                self.pointcloud_publisher.publish(
                    pointcloud.create_pointcloud_msg(points, 'map')
                )
                self.get_logger().info(f"Frame {self.frame_count}: pointcloud published")
            else:
                self.get_logger().warn(f"Frame {self.frame_count}: no points generated from mask")

        except Exception as e:
            self.get_logger().error(f"rgbd_callback failed on frame {self.frame_count}: {repr(e)}")
            tb = traceback.format_exc()
            for line in tb.splitlines():
                self.get_logger().error(line)
        finally:
            self.get_logger().info(f"Frame {self.frame_count}: processing complete")
            self.processing = False

def main(args=None):
    try:
        rclpy.init(args=args)
        camera_subscriber = CameraSubscriber()
        rclpy.spin(camera_subscriber)
        camera_subscriber.destroy_node()
        rclpy.shutdown()
    except KeyboardInterrupt:
        print("\r")
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()
