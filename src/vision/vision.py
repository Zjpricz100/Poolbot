#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo # For camera intrinsic parameters
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
import os
import time
import tf
from geometry_msgs.msg import Point, PointStamped
from std_msgs.msg import Header


from hsv_color_thresholder import ImageSubscriber
from object_detector import ObjectDetector


"""
1. Scan above the table, record HSV values for the balls present
2. Create a tf for each object present.
3. Move robot into position to hit the cue ball into the target ball to get into hole. 
"""


def main():

    """
    Step 1. Scan above the table record thresholding values for each ball present.
    """

    # Initialize the ROS node
    rospy.init_node('hsv_color_thresholder', anonymous=True)
    img_sub = ImageSubscriber() #Stores hsv values of the balls. 
    # Main loop keeps updating the namedwindow
    while not rospy.is_shutdown():
        img_sub.process_image() #Will print the hsv values when done.

    # Close all OpenCV windows after exiting
    cv2.destroyAllWindows()

    """
    Step 2 Create tf objects for cue ball
    """
    #TODO Manually input the values after recording them in step 1
    # into the process image function so that it has access to them.

    #Create an object detector
    detect_balls = ObjectDetector()
    #Creates frames for each ball present on the board
    detect_balls.process_images()


    """
    Step 3 Move robot into position to hit the cue ball into the target ball to get into hole. 
    """




if __name__ == '__main__':
    main()
