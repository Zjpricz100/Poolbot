
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo  # For camera intrinsic parameters
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
import os
import time
import tf
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import Header
class TableLog:
    def __init__(self):
        self.message_type = PoseStamped
        self.pubs = {
            "white" : rospy.Publisher("white_ball", self.message_type, queue_size=10),
            "yellow" : rospy.Publisher("yellow_ball", self.message_type, queue_size=10),
            "blue" : rospy.Publisher("blue_ball", self.message_type, queue_size=10),
            "red" : rospy.Publisher("red_ball", self.message_type, queue_size=10),
            "purple" : rospy.Publisher("purple_ball", self.message_type, queue_size=10),
            "orange" : rospy.Publisher("orange_ball", self.message_type, queue_size=10),
            "maroon" : rospy.Publisher("maroon_ball", self.message_type, queue_size=10),
            "black" : rospy.Publisher("black_ball", self.message_type, queue_size=10),
            "green" : rospy.Publisher("green_ball", self.message_type, queue_size=10)
        }
    
    