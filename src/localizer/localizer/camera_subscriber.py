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
from ultralytics import YOLO

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
        super().__init__("localizer_3D")

        self.declare_parameter('confidence', 0.2)
        self.declare_parameter('num_points', 10000)
        
        self.confidence = self.get_parameter('confidence').get_parameter_value().double_value
        self.num_points = self.get_parameter('num_points').get_parameter_value().integer_value

        self.detection_model = YOLO("yolo11n.pt")
        self.segmentation_model = YOLO("yolo11n-seg.pt")
        self.track_id = -1
        self.objects = []
        self.clicked_point = None

        self.position = [0,0,0]
        
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

    def set_num_points(self, msg):
        self.num_points = msg.data

    def select_by_tracker_id(self, msg):
        self.track_id = msg.data

    def select_by_class_name(self, msg):
        class_name = msg.data
        if ('none' in class_name):
            self.track_id = -1
            return

        objects_of_class = [o for o in self.objects if o['class_name'] == class_name]

        if len(objects_of_class) == 0:
            self.track_id = -1
        else:
            self.track_id = sorted(objects_of_class, key=lambda o: o['confidence'])[0]['track_id']
    
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
        image = self.bridge.imgmsg_to_cv2(msg.rgb, desired_encoding="rgb8")
        color_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        detection_color_img = np.copy(color_img)
        depth_img = self.bridge.imgmsg_to_cv2(msg.depth, desired_encoding="16UC1")
        depth_img = (depth_img / 1000.0) * 2
        original_h, original_w = color_img.shape[:2]

        results = self.detection_model.track(
            source=color_img,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
            conf=self.confidence
        )

        mask = None
        is_valid = True
        tracked_position = (0,0)

        self.objects = []
        parsed_results = []
        
        for result in results[0]:
            result = json.loads(result.to_json())[0]
            parsed_results.append(result)

            box = result['box']
            x = int((box['x1'] + box['x2']) / 2)
            y = int((box['y1'] + box['y2']) / 2)

            if "track_id" in result:        
                track_id = result['track_id']
            else:
                track_id = -2

            self.objects.append({'track_id': track_id, 'class_name': result['name'], 'confidence': result['confidence']})
        
        if self.clicked_point is not None:
            click_x, click_y = self.clicked_point

            candidates = []
            for result in parsed_results:
                box = result["box"]
                if box["x1"] <= click_x <= box["x2"] and box["y1"] <= click_y <= box["y2"]:
                    area = (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])
                    candidates.append((area, result))
            if candidates:
                candidates.sort(key=lambda t: t[0])  # smallest area first
                best = candidates[0][1]
                if "track_id" in best:
                    self.track_id = best["track_id"]
                else:
                    self.track_id = -2
                self.get_logger().info(
                    f"Selected object: {best['name']} track_id={self.track_id} at ({click_x}, {click_y})"
                )
            else:
                self.get_logger().info(f"No object found at clicked point ({click_x}, {click_y})")
                self.track_id = -2
            self.clicked_point = None

        for result in parsed_results:
            box = result['box']
            x = int((box['x1'] + box['x2']) / 2)
            y = int((box['y1'] + box['y2']) / 2)

            if "track_id" in result:        
                track_id = result['track_id']
            else:
                track_id = -2

            if track_id != self.track_id:
                cv2.circle(detection_color_img, (x, y), 10, (0, 255, 255), -1)
                continue

            tracked_position = (box['x1'], box['y1'])

            cv2.circle(detection_color_img, (x, y), 10, (255, 0, 255), -1)
            cv2.rectangle(detection_color_img, (int(box['x1']), int(box['y1'])), (int(box['x2']), int(box['y2'])), (255, 0, 255), 3)

            cropped_img = color_img[int(box['y1']):int(box['y2']), int(box['x1']):int(box['x2'])]
            segmentation_results = self.segmentation_model(cropped_img, verbose=False)[0]
            

            if segmentation_results.masks == None:
                # FIXME: Figure out why this happens and fix if necessary
                self.get_logger().warning("Segmentation resulted in no masks!")
                is_valid = False
                continue

            indices = segmentation_results.boxes.data[:, 5]
            
            if (segmentation_results.masks.data == result['class']).any():
                masked_img = segmentation_results.masks.data[torch.where(indices == result['class'])]
            else:
                masked_img = segmentation_results.masks.data
            masked_img = torch.any(masked_img, dim=0).int() * 255
            mask = masked_img.cpu().numpy().astype(np.uint8)
            h, w = cropped_img.shape[:2]
            mask = cv2.resize(mask, (w, h))
            mask = mask > 0

            depth_img = depth_img[int(box['y1']):int(box['y1'])+mask.shape[0], int(box['x1']):int(box['x1'])+mask.shape[1]] * mask
            color_img = color_img[int(box['y1']):int(box['y1'])+mask.shape[0], int(box['x1']):int(box['x1'])+mask.shape[1]]
            img_msg = self.bridge.cv2_to_imgmsg(cv2.cvtColor(255 - mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR))
            self.mask_publisher.publish(img_msg)
        
        img_msg = self.bridge.cv2_to_imgmsg(detection_color_img, encoding="rgb8")
        self.detection_publisher.publish(img_msg)

        # self.get_logger().info(str(self.objects))

        if not is_valid:
            return

        num_points = self.num_points
        points = self.create_pointcloud(color_img, depth_img, num_points, (original_w, original_h), tracked_position)

        if len(points) > 0:
            self.pointcloud_publisher.publish(pointcloud.create_pointcloud_msg(points, 'map'))

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
