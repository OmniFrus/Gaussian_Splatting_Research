import tkinter as tk
import subprocess
import rclpy
import threading
import os
import signal
from .camera_feed import CameraFeed, start_ros_spin

class ConfigUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Localizer Configuration")
        self.root.geometry("800x600")
        
        # Controls frame
        controls_frame = tk.Frame(root)
        controls_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

        # Points setting
        tk.Label(controls_frame, text="Number of points:", anchor="w").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.num_points_entry = tk.Entry(controls_frame, width=20)
        self.num_points_entry.insert(0, "10000")
        self.num_points_entry.grid(row=0, column=1, padx=10, pady=10)

        tk.Label(controls_frame, text="(Higher = more detailed pointcloud, default: 10000)",
                font=("Arial", 8), fg="gray").grid(row=0, column=2, padx=5)

        # Confidence threshold setting
        tk.Label(controls_frame, text="Confidence:", anchor="w").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.confidence_entry = tk.Entry(controls_frame, width=20)
        self.confidence_entry.insert(0, "0.2")
        self.confidence_entry.grid(row=1, column=1, padx=10, pady=10)

        tk.Label(controls_frame, text="(0.0 - 1.0, default: 0.2)",
                font=("Arial", 8), fg="gray").grid(row=1, column=2, padx=5)
        
        # Class name setting
        tk.Label(controls_frame, text="Class name:", anchor="w").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.class_name_entry = tk.Entry(controls_frame, width=20)
        self.class_name_entry.insert(0, "0.2")
        self.class_name_entry.grid(row=2, column=1, padx=10, pady=10)

        # Status label
        self.status_label = tk.Label(controls_frame, text="Ready to start", fg="green")
        self.status_label.grid(row=3, column=0, columnspan=3, pady=20)

        # Start button
        self.start_button = tk.Button(
            controls_frame, text="Start", command=self.start_nodes,
            bg="#4CAF50", fg="white", font=("Arial", 12),
            width=15, height=2
        )
        self.start_button.grid(row=4, column=0, columnspan=3, pady=10)

        # Restart subscriber button
        self.restart_button = tk.Button(
            controls_frame, text="Restart Subscriber", command=self.restart_subscriber,
            bg="#FF9800", fg="white", font=("Arial", 12),
            width=15, height=2, state="disabled"
        )
        self.restart_button.grid(row=5, column=0, columnspan=3, pady=10)

        # Stop button (same behavior as closing the UI)
        self.stop_button = tk.Button(
            controls_frame, text="Stop", command=self.stop_nodes,
            bg="#F44336", fg="white", font=("Arial", 12),
            width=15, height=2, state="disabled"
        )
        self.stop_button.grid(row=6, column=0, columnspan=3, pady=10)

        self.send_button = tk.Button(
            controls_frame, text="Send", command=self.send_config,
            bg="#D436F4", fg="white", font=("Arial", 12),
            width=15, height=2
        )
        self.send_button.grid(row=7, column=0, columnspan=3, pady=10)

        # Camera frame
        camera_frame = tk.Frame(root)
        camera_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")

        tk.Label(camera_frame, text="Camera Feed (click to select point)",
                font=("Arial", 10)).grid(row=0, column=0, pady=5)

        self.camera_label = tk.Label(camera_frame, text="Waiting for camera...", bg="black", fg="white")
        self.camera_label.grid(row=1, column=0, padx=10, pady=10)
        self.camera_label.bind("<Button-1>", self.on_camera_click)

        self.camera_active = False
        self.camera_process = None
        self.subscriber_process = None
        self.camera_feed = None
        self.ros_thread = None
        
        self.update_feed()

    def is_running(self, process):
        return process is not None and process.poll() is None
    
    def send_config(self):
        num_points = int(self.num_points_entry.get())
        subscriber_cmd = ["ros2", "topic", "pub", "--once", 
                             "/config/num_points", "std_msgs/msg/Int32", "{data: "+f"{num_points}"+"}"]
            
        subprocess.Popen(subscriber_cmd, start_new_session=True)
    
    def kill_process(self, process):
        if process is None:
            return
        
        try:
            if process.poll() is not None:
                return

            # Try to kill the whole process group (ROS2 spawns children),
            # but avoid killing our own process group (would SIGINT the UI).
            try:
                child_pgid = os.getpgid(process.pid)
            except Exception:
                child_pgid = None

            my_pgid = os.getpgrp()

            # Graceful shutdown first (Ctrl+C style)
            try:
                if child_pgid is not None and child_pgid != my_pgid:
                    os.killpg(child_pgid, signal.SIGINT)
                else:
                    process.send_signal(signal.SIGINT)
            except Exception:
                pass

            try:
                process.wait(timeout=2)
                return
            except subprocess.TimeoutExpired:
                pass

            # Escalate to SIGTERM / terminate
            try:
                if child_pgid is not None and child_pgid != my_pgid:
                    os.killpg(child_pgid, signal.SIGTERM)
                else:
                    process.terminate()
            except Exception:
                try:
                    process.terminate()
                except Exception:
                    pass

            try:
                process.wait(timeout=2)
                return
            except subprocess.TimeoutExpired:
                pass

            # Force kill
            try:
                if child_pgid is not None and child_pgid != my_pgid:
                    os.killpg(child_pgid, signal.SIGKILL)
                else:
                    process.kill()
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass

            try:
                process.wait()
            except Exception:
                pass

        except Exception:
            pass
        
    def start_nodes(self):
        try:
            num_points = int(self.num_points_entry.get())
            confidence = float(self.confidence_entry.get())
            
            if num_points < 1:
                self.status_label.config(text="Error: Points must be >= 1", fg="red")
                return
            
            if confidence < 0.0 or confidence > 1.0:
                self.status_label.config(text="Error: Confidence must be between 0.0 and 1.0", fg="red")
                return
            
            # Kill any existing subscriber first
            self.kill_process(self.subscriber_process)
            self.subscriber_process = None
            
            self.status_label.config(text="Starting subscriber...", fg="blue")
            self.root.update()
                        
            # Start camera node (only if not already running)
            if not self.is_running(self.camera_process):
                camera_cmd = ["ros2", "run", "realsense2_camera", "realsense2_camera_node", 
                             "--ros-args", "-p", "enable_rgbd:=True", "-p", "enable_sync:=True",
                             "-p", "align_depth.enable:=True", "-p", "enable_color:=True", 
                             "-p", "enable_depth:=True", "-p", "enable_gyro:=True",
                             "-p", "enable_accel:=True", "-r", "__node:=camera"]
                
                self.camera_process = subprocess.Popen(camera_cmd, start_new_session=True)
                self.camera_active = True

            # Start subscriber node with parameters
            subscriber_cmd = ["ros2", "run", "localizer", "save_images", 
                             "--ros-args", "-p", f"num_points:={num_points}",
                             "-p", f"confidence:={confidence}"]
            
            self.subscriber_process = subprocess.Popen(subscriber_cmd, start_new_session=True)
            
            self.status_label.config(text="Subscriber started!", fg="green")
            self.start_button.config(state="disabled", text="Running...")
            self.restart_button.config(state="normal")
            self.stop_button.config(state="normal")
            
        except ValueError:
            self.status_label.config(text="Error: Invalid input values", fg="red")
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}", fg="red")

    def restart_subscriber(self):
        try:
            num_points = int(self.num_points_entry.get())
            confidence = float(self.confidence_entry.get())
            
            if num_points < 1:
                self.status_label.config(text="Error: Points must be >= 1", fg="red")
                return
            
            if confidence < 0.0 or confidence > 1.0:
                self.status_label.config(text="Error: Confidence must be between 0.0 and 1.0", fg="red")
                return
            
            self.status_label.config(text="Restarting subscriber with new values...", fg="blue")
            self.root.update()
            
            # Kill current subscriber (do NOT touch camera)
            self.kill_process(self.subscriber_process)
            self.subscriber_process = None
            
            # Start subscriber node with new parameters
            subscriber_cmd = ["ros2", "run", "localizer", "save_images", 
                             "--ros-args", "-p", f"num_points:={num_points}",
                             "-p", f"confidence:={confidence}"]
            
            self.subscriber_process = subprocess.Popen(subscriber_cmd, start_new_session=True)
            
            self.status_label.config(text="Subscriber restarted with new values!", fg="green")
            self.stop_button.config(state="normal")
            
        except ValueError:
            self.status_label.config(text="Error: Invalid input values", fg="red")
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}", fg="red")

    def stop_nodes(self):
        # Stop camera + subscriber, but keep UI open and keep rclpy running
        self.status_label.config(text="Stopping...", fg="blue")
        self.root.update()

        self.kill_process(self.subscriber_process)
        self.subscriber_process = None

        self.kill_process(self.camera_process)
        self.camera_process = None
        self.camera_active = False

        # UI state
        self.start_button.config(state="normal", text="Start")
        self.restart_button.config(state="disabled")
        self.stop_button.config(state="disabled")
        self.status_label.config(text="Stopped (UI still open)", fg="green")

        # Reset the camera label to "waiting"
        self.camera_label.config(image="", text="Waiting for camera...", bg="black", fg="white")
        self.camera_label.image = None

    def update_feed(self):
        # If camera is stopped, keep showing the startup message and no image
        if not self.camera_active:
            self.camera_label.config(image="", text="Waiting for camera...", bg="black", fg="white")
            self.camera_label.image = None
            self.root.after(30, self.update_feed)
            return

        if self.camera_feed is not None:
            image = self.camera_feed.get_image_for_tkinter(max_width=640, max_height=480)
            if image is not None:
                self.camera_label.config(image=image, text="")
                self.camera_label.image = image

        self.root.after(30, self.update_feed)
    
    def on_camera_click(self, event):
        if not self.camera_active:
            return
        
        if self.camera_feed is not None and self.camera_label.image is not None:
            x = event.x
            y = event.y
            img = self.camera_label.image
            width = img.width()
            height = img.height()
            self.camera_feed.handle_click(x, y, width, height)
    
    def on_closing(self):
        self.kill_process(self.subscriber_process)
        self.kill_process(self.camera_process)
        
        if rclpy.ok():
            rclpy.shutdown()
        self.root.destroy()

def main():
    if not rclpy.ok():
        rclpy.init()
    
    root = tk.Tk()
    app = ConfigUI(root)
    
    # Initialize camera feed
    app.camera_feed = CameraFeed()
    app.ros_thread = threading.Thread(target=start_ros_spin, args=(app.camera_feed,), daemon=True)
    app.ros_thread.start()
    
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == '__main__':
    main()
