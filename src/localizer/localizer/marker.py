import math
import numpy as np
from visualization_msgs.msg import Marker

def normalize(v, eps=1e-9):
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n

def rotation_matrix_to_quaternion(R):
    tr = np.trace(R)
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        i = np.argmax(np.diag(R))
        if i == 0:
            S = math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif i == 1:
            S = math.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = math.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))

def create_arrow_marker(node, position_xyz, quat_xyzw, frame_id, marker_id=0):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = node.get_clock().now().to_msg()

    marker.ns = "object_pose"
    marker.id = 1
    marker.type = Marker.ARROW
    marker.action = Marker.ADD

    marker.pose.position.x = float(position_xyz[0])
    marker.pose.position.y = float(position_xyz[1])
    marker.pose.position.z = float(position_xyz[2])

    marker.pose.orientation.x = quat_xyzw[0]
    marker.pose.orientation.y = quat_xyzw[1]
    marker.pose.orientation.z = quat_xyzw[2]
    marker.pose.orientation.w = quat_xyzw[3]

    marker.scale.x = 10.30
    marker.scale.y = 5.05
    marker.scale.z = 1.08

    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0

    marker.lifetime.sec = 0
    marker.lifetime.nanosec = 0

    return marker
