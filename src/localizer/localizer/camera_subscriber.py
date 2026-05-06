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
from . import semantic_pointcloud
from .class_registry import ClassRegistry
from . import marker
import time
from .timing_logger import TimingLogger
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Imu
from .semantic_processor import SemanticProcessor

# import marker
from visualization_msgs.msg import Marker

qos_profile = QoSProfile(
    depth=2,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    durability=DurabilityPolicy.VOLATILE
)

imu_qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)

class CameraSubscriber(Node):
    def __init__(self):
        self.timing_logger = TimingLogger()
        self.processing = False
        self.frame_count = 0
        super().__init__("localizer_3D")
        self.sam3 = SAM3Wrapper(default_prompt="", resolution=1008)
        self.current_prompt = "" 

        # fallback (camera intrinsics)
        self.fx = 912.66455078125
        self.fy = 912.659912109375
        self.cx = 646.881591796875
        self.cy = 376.11798095703125

        self.has_intrinsics = False
        self.class_registry = ClassRegistry()
        self.semantic_processor = SemanticProcessor(
            self.sam3,
            self.class_registry,
            self.timing_logger,
            self.get_logger()
        )

        self.declare_parameter('confidence', 0.2)
        self.declare_parameter('num_points', 10000)
        self.declare_parameter('sam3_run_every_n_frames', 1)
        
        self.confidence = self.get_parameter('confidence').get_parameter_value().double_value
        self.num_points = self.get_parameter('num_points').get_parameter_value().integer_value
        self.sam3_run_every_n_frames = self.get_parameter('sam3_run_every_n_frames').get_parameter_value().integer_value

        self.declare_parameter("prompt", "")
        self.current_prompt = self.get_parameter("prompt").get_parameter_value().string_value

        if self.current_prompt.strip() == "":
            self.get_logger().info(
                f"SAM3 prompt empty: using default classes {self.class_registry.default_classes}"
            )
        else:
            self.sam3.set_prompt(self.current_prompt)

        self.track_id = -1
        self.objects = []
        self.clicked_point = None

        self.position = [0,0,0]
        self.last_mask = None
        self.last_box = None
        self.last_score = None
        self.last_labels = None
        self.last_overlay = None
        self.last_semantic_map = None
        self.last_semantic_map_color = None
        self.latest_accel = None
        self.latest_gyro = None
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

        self.pointcloud_visual_publisher = self.create_publisher(
            sensor_msgs.PointCloud2,
            '/pointcloud_visual',
            10
        )

        self.marker_publisher = self.create_publisher(
            Marker,
            '/object_pose_marker',
            10
        )

        self.semantic_map_publisher = self.create_publisher(
            Image,
            "/semantic_map",
            10
        )

        self.semantic_map_color_publisher = self.create_publisher(
            Image,
            "/semantic_map_color",
            10
        )

        self.create_subscription(
            CameraInfo,
            '/camera/camera/aligned_depth_to_color/camera_info',
            self.camera_info_callback,
            10
        )

        self.create_subscription(
            Imu,
            "/camera/camera/accel/sample",
            self.accel_callback,
            imu_qos
        )

        self.create_subscription(
            Imu,
            "/camera/camera/gyro/sample",
            self.gyro_callback,
            imu_qos
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

        # Empty prompt means: run default class list
        self.current_prompt = prompt

        if self.current_prompt == "":
            self.get_logger().info(
                f"SAM3 prompt empty: using default classes {self.class_registry.default_classes}"
            )
        else:
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
    
    def rgbd_callback(self, msg):
        if self.processing:
            self.get_logger().debug("Skipping frame because previous frame is still processing")
            return

        self.processing = True
        self.frame_count += 1
        self.get_logger().info(f"RGBD callback received, frame {self.frame_count}")

        if self.frame_count % 5 == 0:
            if self.latest_accel is not None:
                self.get_logger().info(
                    f"ACCEL=({self.latest_accel.linear_acceleration.x:.3f}, "
                    f"{self.latest_accel.linear_acceleration.y:.3f}, "
                    f"{self.latest_accel.linear_acceleration.z:.3f})"
                )

            if self.latest_gyro is not None:
                self.get_logger().info(
                    f"GYRO=({self.latest_gyro.angular_velocity.x:.3f}, "
                    f"{self.latest_gyro.angular_velocity.y:.3f}, "
                    f"{self.latest_gyro.angular_velocity.z:.3f})"
                )
        
        try:
            self.get_logger().info(f"Frame {self.frame_count}: converting RGB image")
            image = self.bridge.imgmsg_to_cv2(msg.rgb, desired_encoding="rgb8")
            color_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            detection_color_img = np.copy(color_img)

            self.get_logger().info(f"Frame {self.frame_count}: converting depth image")
            depth_img = self.bridge.imgmsg_to_cv2(msg.depth, desired_encoding="16UC1")
            depth_img = depth_img.astype(np.float32) / 1000.0

            original_h, original_w = color_img.shape[:2]
            self.get_logger().info(
                f"Frame {self.frame_count}: color_shape={color_img.shape}, "
                f"depth_shape={depth_img.shape}, depth_nonzero={int(np.count_nonzero(depth_img))}"
            )

            run_sam3_this_frame = (
                self.frame_count % self.sam3_run_every_n_frames
            ) == 0 or self.last_mask is None

            if run_sam3_this_frame:
                result = self.semantic_processor.run(
                    color_img=color_img,
                    depth_img=depth_img,
                    detection_color_img=detection_color_img,
                    current_prompt=self.current_prompt,
                    frame_count=self.frame_count,
                    confidence=self.confidence
                )

                mask = result["mask"]
                box = result["box"]
                score = result["score"]
                labels = result["labels"]
                semantic_overlay = result["semantic_overlay"]
                semantic_map = result["semantic_map"]
                semantic_map_color = result["semantic_map_color"]

                self.last_mask = None if mask is None else mask.copy()
                self.last_box = None if box is None else [None if b is None else b.copy() for b in box]
                self.last_score = None if score is None else list(score)
                self.last_labels = None if labels is None else list(labels)
                self.last_overlay = semantic_overlay.copy()
                self.last_semantic_map = semantic_map.copy()
                self.last_semantic_map_color = semantic_map_color.copy()
                self.last_sam3_frame = self.frame_count

            else:
                mask = None if self.last_mask is None else self.last_mask.copy()
                box = None if self.last_box is None else [None if b is None else b.copy() for b in self.last_box]
                score = self.last_score
                labels = getattr(self, "last_labels", None)
                semantic_overlay = getattr(self, "last_overlay", np.zeros_like(detection_color_img, dtype=np.uint8))
                semantic_map = getattr(self, "last_semantic_map", np.zeros(depth_img.shape[:2], dtype=np.uint8))
                semantic_map_color = getattr(self, "last_semantic_map_color", np.zeros_like(detection_color_img, dtype=np.uint8))
                
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

            semantic_msg = self.bridge.cv2_to_imgmsg(semantic_map, encoding="mono8")
            self.semantic_map_publisher.publish(semantic_msg)

            semantic_color_msg = self.bridge.cv2_to_imgmsg(semantic_map_color, encoding="bgr8")
            self.semantic_map_color_publisher.publish(semantic_color_msg)

            alpha = 0.45
            colored_pixels = mask == 1

            detection_color_img[colored_pixels] = (
                (1 - alpha) * detection_color_img[colored_pixels]
                + alpha * semantic_overlay[colored_pixels]
            ).astype(np.uint8)

            tracked_position = (0, 0)

            if box is not None and len(box) > 0:
                for i, b in enumerate(box):
                    if b is None:
                        continue

                    x1, y1, x2, y2 = [int(v) for v in b]

                    s = None
                    if score is not None and len(score) > i:
                        s = score[i]

                    label = labels[i] if labels is not None and len(labels) > i else self.current_prompt

                    class_id = self.class_registry.get_class_id(label)
                    color = self.class_registry.get_class_color(label)

                    cv2.rectangle(detection_color_img, (x1, y1), (x2, y2), color, 2)

                    text = f"{label}: {s:.2f}" if s is not None else label
                    cv2.putText(
                        detection_color_img,
                        text,
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

                first_box = box[0]
                if first_box is not None:
                    tracked_position = (int(first_box[0]), int(first_box[1]))

            img_msg = self.bridge.cv2_to_imgmsg(detection_color_img, encoding="bgr8")
            self.detection_publisher.publish(img_msg)

            masked_depth = depth_img.copy()
            masked_depth[mask == 0] = 0
            self.get_logger().info(
                f"Frame {self.frame_count}: masked_depth_nonzero={int(np.count_nonzero(masked_depth))}"
            )

            pc_start = time.perf_counter()

            # points = semantic_pointcloud.create_pointcloud(
            #     semantic_map_color,
            #     masked_depth,
            #     semantic_map,
            #     self.num_points,
            #     (original_w, original_h),
            #     tracked_position,
            #     self.fx,
            #     self.fy,
            #     self.cx,
            #     self.cy
            # )

            # points = semantic_pointcloud.create_balanced_semantic_pointcloud(
            #     semantic_map_color,
            #     depth_img,
            #     semantic_map,
            #     self.num_points,
            #     self.fx,
            #     self.fy,
            #     self.cx,
            #     self.cy
            # )

            points = semantic_pointcloud.create_area_aware_semantic_pointcloud(
                semantic_map_color,
                depth_img,
                semantic_map,
                self.num_points,
                self.fx,
                self.fy,
                self.cx,
                self.cy
            )

            pc_elapsed = time.perf_counter() - pc_start

            self.get_logger().info(
                f"Frame {self.frame_count}: generated_pointcloud_points={len(points)}"
            )

            self.timing_logger.log(
                frame=self.frame_count,
                class_name="ALL",
                stage="pointcloud_generation",
                elapsed_seconds=pc_elapsed,
                num_masks=0,
                num_points=len(points)
            )

            self.get_logger().info(
                f"TIMING frame={self.frame_count}, pointcloud_time={pc_elapsed:.4f}s"
            )

            if len(points) > 0:
                # Correct metric pointcloud
                self.pointcloud_publisher.publish(
                    pointcloud.create_pointcloud_msg(points, 'map')
                )

                # Visualization-only scaled pointcloud
                points_visual = points.copy()
                points_visual[:, 0:3] *= 2.0

                self.pointcloud_visual_publisher.publish(
                    pointcloud.create_pointcloud_msg(points_visual, 'map')
                )

                self.get_logger().info(
                    f"Frame {self.frame_count}: pointcloud published "
                    f"(/pointcloud metric, /pointcloud_visual scaled)"
                )
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
    
    def camera_info_callback(self, msg):
        k = msg.k

        self.fx = k[0]
        self.fy = k[4]
        self.cx = k[2]
        self.cy = k[5]

        if not self.has_intrinsics:
            self.get_logger().info(
                f"Received intrinsics: fx={self.fx:.2f}, fy={self.fy:.2f}, "
                f"cx={self.cx:.2f}, cy={self.cy:.2f}"
            )
            self.has_intrinsics = True

    def accel_callback(self, msg):
        self.latest_accel = msg

    def gyro_callback(self, msg):
        self.latest_gyro = msg

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
