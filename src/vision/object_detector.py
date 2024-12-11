#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo  # For camera intrinsic parameters
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
import os
import time
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from std_msgs.msg import Header
from table_log import TableLog
from tf_transformations import quaternion_matrix, quaternion_from_matrix


PLOTS_DIR = os.path.join(os.getcwd(), 'plots')

class ObjectDetector:
    def __init__(self):
        rospy.init_node('object_detector', anonymous=True)

        self.bridge = CvBridge()

        self.cv_color_image = None

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.usb_camera_info_sub = rospy.Subscriber("/usb_cam/camera_info", CameraInfo, self.usb_camera_info_callback)
        self.head_camera_info_sub = rospy.Subscriber("/io/internal_camera/head_camera/camera_info", CameraInfo, self.head_camera_info_callback)
        
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)
        rospy.sleep(2)

        self.point_pub = rospy.Publisher("goal_point", Point, queue_size=10)
        self.image_pub = rospy.Publisher('detected_cup', Image, queue_size=10)

        self.message_type = PoseStamped
        self.pubs = {
            "white" : rospy.Publisher("ball/white", self.message_type, queue_size=10),
            "yellow" : rospy.Publisher("ball/yellow", self.message_type, queue_size=10),
            "blue" : rospy.Publisher("ball/blue", self.message_type, queue_size=10),
            "red" : rospy.Publisher("ball/red", self.message_type, queue_size=10),
            "purple" : rospy.Publisher("ball/purple", self.message_type, queue_size=10),
            "orange" : rospy.Publisher("ball/orange", self.message_type, queue_size=10),
            "maroon" : rospy.Publisher("ball/maroon", self.message_type, queue_size=10),
            "black" : rospy.Publisher("ball/black", self.message_type, queue_size=10),
            "green" : rospy.Publisher("ball/green", self.message_type, queue_size=10)
        }
        self.color_image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.color_image_callback)

        rospy.spin()

    def usb_camera_info_callback(self, msg):
        # Extract the intrinsic parameters from the CameraInfo message
        self.fx_usb = msg.K[0]
        self.fy_usb = msg.K[4]
        self.cx_usb = msg.K[2]
        self.cy_usb = msg.K[5]

    def head_camera_info_callback(self, msg):
        # Extract the intrinsic parameters from the CameraInfo message
        self.fx_head = msg.K[0]
        self.fy_head = msg.K[4]
        self.cx_head = msg.K[2]
        self.cy_head = msg.K[5]

    def pose_to_matrix(self, pose):
        """Convert PoseStamped to a 4x4 transformation matrix."""
        q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        t = [pose.position.x, pose.position.y, pose.position.z]
        matrix = quaternion_matrix(q)
        matrix[:3, 3] = t
        return matrix

    def matrix_to_transform(self, matrix):
        """Convert 4x4 transformation matrix to TransformStamped."""
        q = quaternion_from_matrix(matrix)
        t = matrix[:3, 3]
        transform = TransformStamped()
        transform.transform.translation.x = t[0]
        transform.transform.translation.y = t[1]
        transform.transform.translation.z = t[2]
        transform.transform.rotation.x = q[0]
        transform.transform.rotation.y = q[1]
        transform.transform.rotation.z = q[2]
        transform.transform.rotation.w = q[3]
        return transform

    def get_usb_cam_transform(self, tf1, tf2): #tf1 from head cam, tf2 from usb_cam
        T_usb_to_ar = self.pose_to_matrix(tf2.pose)
        T_head_to_ar = self.pose_to_matrix(tf1.pose)

        T_ar_to_usb = np.linalg.inv(T_usb_to_ar)
        T_head_to_usb = np.dot(T_head_to_ar, T_ar_to_usb)

        base_to_head = self.tf_buffer.lookup_transform('base', 'head_cam', rospy.Time(0))
        T_base_to_head = self.pose_to_matrix(base_to_head.transform)
        T_base_to_usb = np.dot(T_base_to_head, T_head_to_usb)

    def color_image_callback(self, msg):

        def image_to_cartesian(image, Z):
            """
            Converts a tuple of (x, y) into cartesian coordinates (X, Y, Z)
            where Z is fixed depth
            """
            K = np.array([
                [self.fx, 0, 0],
                [0, self.fy, 0],
                [0, 0, 1]
            ])
            image = np.array(image)
            image = np.concatenate((image, Z))
            cartesian = np.linalg.inv(K) @ image

        try:
            
            # Convert the ROS Image message to an OpenCV image (BGR8 format)
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            ball_dict = self.detect_balls(frame)
            for (key, value) in ball_dict.items():
                pub = self.pubs[key]
                # ball = PoseStamped()
                # ball.pose.position.x = value[0]
                # ball.pose.position.y = value[1]
                # pub.publish(ball)
                Z = self.get_table_height()
                x, y = value
                ball = self.pixel_to_world(x, y, Z)
                pub.publish(ball)

        except Exception as e:
            print("Error:", e)

    def get_table_height(self, ar_tag_number = 3):
        """
        Get the height of the table using the AR tag transform.
        """
        # Create a tf2 buffer and listener
    
        try:
            # Lookup the transform for the AR tag
            transform = self.tfBuffer.lookup_transform("usb_cam", "ar_marker_3", rospy.Time(0))
            # Extract the z-coordinate (height)
            table_height = transform.transform.translation.z
            
            #rospy.loginfo(f"Table height (z): {table_height} meters")
            return table_height
        
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            print(e)
            rospy.logerr(f"Failed to get table height: {e}")
            return None
        
    def pixel_to_world(self, u, v, Z):

        # Pixel to camera coordinates
        X_camera = Z * (u - self.cx) / self.fx
        Y_camera = Z * (v - self.cy) / self.fy
        Z_camera = Z

        # Create point in camera frame
        ball_in_camera_frame = PoseStamped()
        ball_in_camera_frame.header.frame_id = "usb_cam"
        ball_in_camera_frame.pose.position.x = X_camera
        ball_in_camera_frame.pose.position.y = Y_camera
        ball_in_camera_frame.pose.position.z = Z_camera

        #print(type(ball_in_camera_frame))
        ball_in_world_frame = self.tfBuffer.transform(ball_in_camera_frame, "base", rospy.Duration(1.0))
        ball_in_base_frame = PoseStamped()
        ball_in_base_frame.header.stamp = rospy.get_rostime()
        ball_in_base_frame.header.frame_id = "base"
        ball_in_base_frame.pose.position.x = ball_in_world_frame.pose.position.x
        ball_in_base_frame.pose.position.y = ball_in_world_frame.pose.position.y
        ball_in_base_frame.pose.position.z = ball_in_world_frame.pose.position.z
        return ball_in_base_frame

    def detect_balls(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to reduce noise and improve circle detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detect circles using HoughCircles
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.3, minDist=30,
                                    param1=100, param2=30, minRadius=15, maxRadius=25)

        # If some circles are detected
        if circles is not None:
            # Round the circle values and convert to integer
            circles = np.round(circles[0, :]).astype("int")

            # Define HSV ranges for target colors
            color_ranges = {
                "white": ([0, 0, 200], [180, 30, 255]),
                "yellow": ([20, 100, 100], [30, 255, 255]),
                "blue": ([70, 0, 50], [120, 255, 255]),
                "red": ([0, 0, 0], [5, 255, 255]),
                "purple": ([110, 0, 0], [179, 255, 255]),
                "orange": ([10, 100, 0], [20, 255, 255]),
                "maroon": ([0, 100, 0], [18, 240, 255]),
                "black": ([0, 0, 0], [180, 255, 50]),
                "green": ([40, 100, 100], [80, 255, 255]),
            }
            ball_dict = {}

            # Loop through the circles and process each one
            for (x, y, r) in circles:
                # Draw the circle in green
                cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
                # Draw the center of the circle in red
                cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)

                # Define bounding box for ROI
                top_left_x = max(x - r, 0)
                top_left_y = max(y - r, 0)
                bottom_right_x = min(x + r, frame.shape[1])
                bottom_right_y = min(y + r, frame.shape[0])

                # Extract the region of interest (ROI)
                roi = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

                if roi.size > 0:
                    # Convert ROI to HSV
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                    # Find the closest matching color
                    max_percentage = 0
                    detected_color = "unknown"

                    for color, (lower_hsv, upper_hsv) in color_ranges.items():
                        lower_hsv = np.array(lower_hsv, dtype="uint8")
                        upper_hsv = np.array(upper_hsv, dtype="uint8")
                        mask = cv2.inRange(hsv_roi, lower_hsv, upper_hsv)

                        # Calculate the percentage of the ROI matching this color
                        color_percentage = np.sum(mask) / mask.size

                        if color_percentage > max_percentage:
                            max_percentage = color_percentage
                            detected_color = color

                    # If a color is detected, display it on the frame
                    if max_percentage > 0.1:  # Threshold to avoid noise
                        #print(f"{detected_color.capitalize()} ball detected at ({x}, {y}) with radius {r}")
                        cv2.putText(
                            frame,
                            detected_color,
                            (x, y - r - 10),  # Position above the circle
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 0),
                            2
                        )
                        ball_dict[detected_color] = (x, y) # populate ball dict with (x, y) values
            cv2.imshow("Circle Detection", frame)
            cv2.waitKey(1)
            return ball_dict


    def get_rot_and_translation(self, ar_tag):
        trans = self.tf_buffer.lookup_transform()



if __name__ == '__main__':
    w = ObjectDetector()
    #w.get_table_height()
