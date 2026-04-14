#!/usr/bin/env python3
"""
Simple script to verify that the marker normal direction is correct.
Shows normal direction, length, and basic statistics.
"""

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
import numpy as np
from scipy.spatial.transform import Rotation

class MarkerVerifier(Node):
    def __init__(self):
        super().__init__('marker_verifier')
        self.subscription = self.create_subscription(
            Marker,
            '/object_pose_marker',
            self.marker_callback,
            10)
        
        self.normal_history = []
        self.frame_count = 0
        
    def marker_callback(self, msg):
        self.frame_count += 1
        
        # Extract quaternion
        qx = msg.pose.orientation.x
        qy = msg.pose.orientation.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w
        
        # Extract position
        px = msg.pose.position.x
        py = msg.pose.position.y
        pz = msg.pose.position.z
        
        # Convert to rotation matrix
        try:
            r = Rotation.from_quat([qx, qy, qz, qw])
            R = r.as_matrix()
            
            # X-axis is the normal (in normal mode)
            normal = R[:, 0]
            
            # Verify normal properties
            length = np.linalg.norm(normal)
            is_normalized = abs(length - 1.0) < 0.01
            
            # Store for statistics
            self.normal_history.append(normal.copy())
            if len(self.normal_history) > 100:
                self.normal_history.pop(0)
            
            # Calculate statistics
            if len(self.normal_history) > 1:
                normals_array = np.array(self.normal_history)
                std = np.std(normals_array, axis=0)
                mean_normal = np.mean(normals_array, axis=0)
                consistency = np.linalg.norm(std)
            else:
                std = np.array([0, 0, 0])
                mean_normal = normal
                consistency = 0.0
            
            # Log results
            status = "✓" if is_normalized else "✗"
            self.get_logger().info(
                f"\n{'='*60}\n"
                f"Frame: {self.frame_count}\n"
                f"Position: [{px:.3f}, {py:.3f}, {pz:.3f}]\n"
                f"Normal: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]\n"
                f"Normal Length: {length:.4f} {status}\n"
                f"Mean Normal (last {len(self.normal_history)} frames): "
                f"[{mean_normal[0]:.3f}, {mean_normal[1]:.3f}, {mean_normal[2]:.3f}]\n"
                f"Consistency (std): {consistency:.4f} "
                f"{'Stable' if consistency < 0.1 else 'Varying'}\n"
                f"{'='*60}"
            )
            
            # Warnings
            if not is_normalized:
                self.get_logger().warn("WARNING: Normal is not normalized!")
            
            if consistency > 0.2:
                self.get_logger().warn("WARNING: Normal direction is varying a lot!")
            
            # Interpret normal direction
            self.interpret_normal(normal)
            
        except Exception as e:
            self.get_logger().error(f"Error processing marker: {e}")
    
    def interpret_normal(self, normal):
        """Interpret what the normal direction means."""
        abs_normal = np.abs(normal)
        max_idx = np.argmax(abs_normal)
        max_val = normal[max_idx]
        
        interpretations = {
            0: "→ Right" if max_val > 0 else "← Left",
            1: "↑ Up" if max_val > 0 else "↓ Down", 
            2: "Towards camera" if max_val > 0 else "Away from camera"
        }
        
        direction = interpretations.get(max_idx, "Unknown")
        self.get_logger().info(f"Normal points mostly: {direction} (component {max_idx} = {max_val:.3f})")

def main(args=None):
    rclpy.init(args=args)
    verifier = MarkerVerifier()
    
    print("\n" + "="*60)
    print("Marker Verifier Started")
    print("="*60)
    print("Waiting for marker messages on /object_pose_marker...")
    print("Make sure the hybrid tracker is running!")
    print("="*60 + "\n")
    
    try:
        rclpy.spin(verifier)
    except KeyboardInterrupt:
        print("\n\nStopping verifier...")
    finally:
        verifier.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

