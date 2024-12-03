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
    img_sub = ImageSubscriber()
    # Main loop keeps updating the namedwindow
    while not rospy.is_shutdown():
        img_sub.process_image()

    # Close all OpenCV windows after exiting
    cv2.destroyAllWindows()


    """
    Step 2 Create tf objects for cue ball
    

    """




if __name__ == '__main__':
    main()
