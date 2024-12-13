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
from tf import TransformListener
from tf.transformations import inverse_matrix, concatenate_matrices


PLOTS_DIR = os.path.join(os.getcwd(), 'plots')

class ObjectDetector:
    def __init__(self):
        rospy.init_node('object_detector', anonymous=True)

        self.bridge = CvBridge()
        
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)
        rospy.sleep(2)

        self.message_type = PoseStamped

        self.camera_info_sub = rospy.Subscriber("/usb_cam/camera_info", CameraInfo, self.usb_camera_info_callback)
        self.color_image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.color_image_callback)

        rospy.spin()

    def usb_camera_info_callback(self, msg):
        #Extract the intrinsic parameters from the CameraInfo message
        self.fx_usb = msg.K[0]
        self.fy_usb = msg.K[4]
        self.cx_usb = msg.K[2]
        self.cy_usb = msg.K[5]

    def color_image_callback(self, msg):
        try:
            
            # Convert the ROS Image message to an OpenCV image (BGR8 format)
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            ball_dict = self.detect_balls(frame)
            for (key, value) in ball_dict.items():
                pub = rospy.Publisher(f"ball/{key}", self.message_type, queue_size=10)
                Z = self.get_table_height(3)
                #print(f"table height: {Z}")
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
            transform = self.tfBuffer.lookup_transform("usb_cam", f"ar_marker_{ar_tag_number}", rospy.Time(0))
            # Extract the z-coordinate (height)
            table_height = transform.transform.translation.z
            
            #rospy.loginfo(f"Table height (z): {table_height} meters")
            return table_height
        
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            print(e)
            rospy.logerr(f"Failed to get table height: {e}")
            return None
        
    def compute_camera_transform(self):
        listener = TransformListener()

        try:
            # Wait for transforms to be available
            listener.waitForTransform('/usb_cam', '/ar_marker_3', rospy.Time(0), rospy.Duration(4.0))
            listener.waitForTransform('/head_camera', '/ar_marker_3', rospy.Time(0), rospy.Duration(4.0))

            # Get transforms
            (trans_space_to_tag, rot_space_to_tag) = listener.lookupTransform('/usb_cam', '/ar_marker_3', rospy.Time(0))
            (trans_head_to_tag, rot_head_to_tag) = listener.lookupTransform('/head_camera', '/ar_marker_3', rospy.Time(0))

            # Convert to transformation matrices
            T_space_to_tag = listener.fromTranslationRotation(trans_space_to_tag, rot_space_to_tag)
            T_head_to_tag = listener.fromTranslationRotation(trans_head_to_tag, rot_head_to_tag)

            # Compute inverse of head_camera -> ar_tag
            T_tag_to_space = inverse_matrix(T_space_to_tag)

            # Compute space_camera -> head_camera
            T_head_to_space = concatenate_matrices(T_head_to_tag, T_tag_to_space)

            # T_head_to_space = inverse_matrix(T_space_to_head)

            return T_head_to_space

        except Exception as e:
            rospy.logerr("Error computing transform: %s", str(e))
        
    def pixel_to_world(self, u, v, Z):
        """
        Converts pixel coordinates (u, v) with depth Z into world coordinates
        using the precomputed transform from `usb_cam` to `head_camera`.
        """

        # Step 1: Compute the camera coordinates
        X_camera = Z * (u - self.cx_usb) / self.fx_usb
        Y_camera = Z * (v - self.cy_usb) / self.fy_usb
        Z_camera = Z

        # Step 2: Convert the camera coordinates to homogeneous form
        camera_point = np.array([X_camera, Y_camera, Z_camera, 1.0])  # Homogeneous coordinates

        # Step 3: Apply the precomputed transformation matrix
        T = self.compute_camera_transform()

        head_frame_point = np.dot(T, camera_point)

        # Step 4: Construct the PoseStamped message in the head frame
        ball_in_head_frame = PoseStamped()
        ball_in_head_frame.header.stamp = rospy.get_rostime()
        ball_in_head_frame.header.frame_id = "head_camera"
        ball_in_head_frame.pose.position.x = head_frame_point[0]
        ball_in_head_frame.pose.position.y = head_frame_point[1]
        ball_in_head_frame.pose.position.z = head_frame_point[2]
        
        #return ball_in_head_frame
        try:
        # Step 5: Transform the point from head_camera to base frame using tfBuffer
            #transform_to_base = self.tfBuffer.lookup_transform(ball_in_head_frame, "base", rospy.Time(0), rospy.Duration(1.0))
            #ball_in_base_frame = tf2_geometry_msgs.do_transform_pose(ball_in_head_frame, transform_to_base)
            ball_in_base_frame = self.tfBuffer.transform(ball_in_head_frame, "base", rospy.Duration(1.0))

            return ball_in_base_frame
        
        except Exception as e:
            rospy.logerr(f"Failed to transform to base frame: {e}")
            return None

    def detect_balls(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to reduce noise and improve circle detection
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)

        # Detect circles using HoughCircles
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.3, minDist=30,
                                    param1=100, param2=30, minRadius=0, maxRadius=20)

        # If some circles are detected
        if circles is not None:
            # Round the circle values and convert to integer
            circles = np.round(circles[0, :]).astype("int")
            ball_dict = {}

            # Loop through the circles and process each one
            for i, (x, y, r) in enumerate(circles):
                #print(r)
                # Draw the circle in green
                cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
                # Draw the center of the circle in red
                cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)

                cv2.putText(
                    frame,
                    f"ball_{i}",
                    (x, y - r - 10),  # Position above the circle
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2
                )
                ball_dict[i] = (x, y) # populate ball dict with (x, y) values

            cv2.imshow("Circle Detection", frame)
            cv2.waitKey(1)
            return ball_dict


    def get_rot_and_translation(self, ar_tag):
        trans = self.tf_buffer.lookup_transform()



if __name__ == '__main__':
    w = ObjectDetector()
    #w.get_table_height()
