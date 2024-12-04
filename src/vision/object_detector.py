#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo # For camera intrinsic parameters
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
import os
import time
import tf
from geometry_msgs.msg import Point, PointStamped
from std_msgs.msg import Header


PLOTS_DIR = os.path.join(os.getcwd(), 'plots')

class ObjectDetector:
    def __init__(self):
        rospy.init_node('object_detector', anonymous=True)

        self.bridge = CvBridge()

        self.cv_color_image = None

        self.color_image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.color_image_callback)

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.camera_info_sub = rospy.Subscriber("/usb_cam/camera_info", CameraInfo, self.camera_info_callback)

        self.tf_listener = tf.TransformListener()  # Create a TransformListener object

        self.point_pub = rospy.Publisher("goal_point", Point, queue_size=10)
        self.image_pub = rospy.Publisher('detected_cup', Image, queue_size=10)

        rospy.spin()

    def camera_info_callback(self, msg):
        # TODO: Extract the intrinsic parameters from the CameraInfo message (look this message type up online)
        self.fx = msg.K[0]
        self.fy = msg.K[4]
        self.cx = msg.K[2]
        self.cy = msg.K[5]


    def color_image_callback(self, msg):
        try:
            # Convert the ROS Image message to an OpenCV image (BGR8 format)
            self.cv_color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.process_images()

        except Exception as e:
            print("Error:", e)

    def get_table_height(self):
        """
        Get the height of the table using the AR tag transform.
        """
        # Create a tf2 buffer and listener
        tfBuffer = tf.Buffer()
        
        try:
            # Lookup the transform for the AR tag
            transform = tfBuffer.lookup_transform("base", "ar_marker_0", rospy.Time(0), rospy.Duration(10.0))
            
            # Extract the z-coordinate (height)
            table_height = transform.transform.translation.z
            rospy.loginfo(f"Table height (z): {table_height} meters")
            return table_height
        
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr(f"Failed to get table height: {e}")
            return None



    def process_images(self):
        # Convert the color image to HSV color space
        hsv = cv2.cvtColor(self.cv_color_image, cv2.COLOR_BGR2HSV)

        # Define color ranges for different balls
        ball_numbers = {
            'cue' : (np.array([0, 0, 110]), np.array([80, 102, 255])),
            'yellow' : (np.array([26, 237, 122]), np.array([63, 255, 255])),
            'blue' : (np.array([78, 103, 28]), np.array([119, 216, 121])),
            'red' : (np.array([0, 220, 113]), np.array([9, 248, 231])),
            'purple' : (np.array([115, 0, 0]), np.array([175, 222, 99])),
            'orange' : (np.array([0, 246, 100]), np.array([25, 255, 255])),
            'maroon' : (np.array([4, 120, 31]), np.array([13, 237, 180])),
            'black' : (np.array([0, 0, 30]), np.array([179, 255, 54])),
            'green' : (np.array([31, 102, 0]), np.array([105, 255, 123]))
        }

      # Create a copy of the original image for drawing
        display_image = self.cv_color_image.copy()

        #z = self.get_table_height()

        # Process each ball color
        for color_name, (lower_hsv, upper_hsv) in ball_numbers.items():
            # Create mask for this color
            mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process each contour
            for contour in contours:
                # Filter by area to remove very small or very large objects
                if 100 < cv2.contourArea(contour) < 5000:
                    # Calculate contour perimeter and area
                    perimeter = cv2.arcLength(contour, True)
                    area = cv2.contourArea(contour)
                    
                    # Compute circularity
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter ** 2)
                        
                        # Check if object is sufficiently circular (close to 1)
                        if circularity > 0.4:
                            # Additional shape verification using minimum enclosing circle
                            (x, y), radius = cv2.minEnclosingCircle(contour)
                            
                            # Verify the object is roughly circular
                            if radius > 10:  # Minimum ball size
                                # Existing ball detection and TF broadcasting logic
                                center_x = int(x)
                                center_y = int(y)
                                
                                # Draw and label circular objects
                                cv2.drawContours(display_image, [contour], 0, (0, 255, 0), 2)
                                cv2.putText(
                                    display_image, 
                                    f"{color_name} Ball (Circularity: {circularity:.2f})", 
                                    (center_x, center_y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.5, 
                                    (255, 0, 0), 
                                    2
                                )

        # Display the image with detected balls
        cv2.imshow('Detected Balls', display_image)
        cv2.waitKey(1)

if __name__ == '__main__':
    ObjectDetector()
