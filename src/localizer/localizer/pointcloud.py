import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs
import numpy as np

# I adapted this from https://github.com/SebastianGrans/ROS2-Point-Cloud-Demo so credit goes to Sebastian Grans :)
# Creates a pointcloud msg for Rviz2. The messages contain the position and color for each point.
def create_pointcloud_msg(points, parent_frame):
    position_ros_dtype = sensor_msgs.PointField.FLOAT32
    position_dtype=np.float32
    color_dtype=np.uint8
    item_size = np.dtype(position_dtype).itemsize

    header = std_msgs.Header(frame_id=parent_frame)
    fields = [sensor_msgs.PointField(name=n, offset=i*item_size, datatype=position_ros_dtype, count=1)
              for i, n in enumerate('xyz')]
    fields.append(sensor_msgs.PointField(name='rgb', offset=3*item_size, datatype=position_ros_dtype, count=1))

    positions = points[:, 0:3].astype(position_dtype).tobytes()
    colors = points[:, 3:6].astype(color_dtype).tobytes()

    data = b''
    for i in range(points.shape[0]):
        data += positions[i*3*item_size:(i+1)*3*item_size]
        data += colors[i*3:(i+1)*3]
        data += b'\x00'

    return sensor_msgs.PointCloud2(
            header=header,
            height=1,
            width=points.shape[0],
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            point_step=(4 * item_size),
            row_step=(4 * item_size * points.shape[0]),
            data=data,
        )