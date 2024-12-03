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

        self.color_image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.color_image_callback)

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.camera_info_sub = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.camera_info_callback)

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

    def process_images(self):
        # Convert the color image to HSV color space
        hsv = cv2.cvtColor(self.cv_color_image, cv2.COLOR_BGR2HSV)

        # Define color ranges for different balls
        #TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
        ball_numbers = {
            # TODO manually plug in values from step 1 in vision.py
            '0' : (np.array([56, 37, 98]), np.array([85, 255, 255])),
            '1' : (np.array([56, 37, 98]), np.array([85, 255, 255])),
            '2' : (np.array([56, 37, 98]), np.array([85, 255, 255])),
            '3' : (np.array([56, 37, 98]), np.array([85, 255, 255])),
            '4' : (np.array([56, 37, 98]), np.array([85, 255, 255])),
            '5' : (np.array([56, 37, 98]), np.array([85, 255, 255])),
            '6' : (np.array([56, 37, 98]), np.array([85, 255, 255])),
            '7' : (np.array([56, 37, 98]), np.array([85, 255, 255])),
            '8' : (np.array([56, 37, 98]), np.array([85, 255, 255])),
            '9' : (np.array([56, 37, 98]), np.array([85, 255, 255])),
            '10' : (np.array([56, 37, 98]), np.array([85, 255, 255])),
            '11' : (np.array([56, 37, 98]), np.array([85, 255, 255])),
            '12' : (np.array([56, 37, 98]), np.array([85, 255, 255])),
            '13' : (np.array([56, 37, 98]), np.array([85, 255, 255])),
            '14' : (np.array([56, 37, 98]), np.array([85, 255, 255])),
            '15' : (np.array([56, 37, 98]), np.array([85, 255, 255])),

            #TODO Remove below values
            'green': (np.array([56, 37, 98]), np.array([85, 255, 255])),
            'red': (np.array([0, 100, 100]), np.array([10, 255, 255])),
            'blue': (np.array([100, 100, 100]), np.array([140, 255, 255]))
        }

        # Process each ball color
        for color_name, (lower_hsv, upper_hsv) in ball_numbers.items():
            # Create mask for this color
            mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process each contour (ball) of this color
            for contour in contours:
                if cv2.contourArea(contour) > 100:
                    # Calculate contour center
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        center_x = int(M["m10"] / M["m00"])
                        center_y = int(M["m01"] / M["m00"])

                        # Broadcast tf frame with 2D pixel coordinates
                        ball_tf_broadcaster = tf.TransformBroadcaster()
                        ball_tf_broadcaster.sendTransform(
                            # TODO TODO TODO TODO TODO
                            #TODO Get depth of the board and replace pix coord z w/ that instead of 0
                            (center_x, center_y, 0),  # Use pixel coordinates
                            tf.transformations.quaternion_from_euler(0, 0, 0),
                            rospy.Time.now(),
                            f"{color_name}_ball_2d",
                            "camera_frame"
                        )

if __name__ == '__main__':
    ObjectDetector()
