#!/usr/bin/env python

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

PLOTS_DIR = os.path.join(os.getcwd(), 'plots')

class ObjectDetector:
    def __init__(self):
        self.cv_color_image = None
        self.ball_numbers = {
        # 'cue': (np.array([0, 0, 110]), np.array([80, 102, 255])),
        # 'yellow': (np.array([26, 237, 122]), np.array([63, 255, 255])),
        # 'blue': (np.array([78, 103, 28]), np.array([119, 216, 121])),
        # 'red': (np.array([0, 220, 113]), np.array([9, 248, 231])),
        # 'purple': (np.array([115, 0, 0]), np.array([175, 222, 99])),
        # 'orange': (np.array([0, 246, 100]), np.array([25, 255, 255])),
        # 'maroon': (np.array([4, 120, 31]), np.array([13, 237, 180])),
        # 'black': (np.array([0, 0, 30]), np.array([179, 255, 54])),
        # 'green': (np.array([31, 102, 0]), np.array([105, 255, 123])),
        'ball': (np.array([0, 0, 0]), np.array([255, 255, 255]))
    }

    def camera_info_callback(self, msg):
        # Extract the intrinsic parameters from the CameraInfo message
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

    def camera_loop(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            self.cv_color_image = frame
            ball_dict = self.detect_balls(frame)
            print(ball_dict)
    
    def detect_balls(self, frame):    
        if len(frame.shape) > 2:
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
            #print("a", ball_dict)
            return ball_dict

        # Display the output frame with circles drawn

    def detect_balls_static(self, frame):    
        if len(frame.shape) > 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # Apply GaussianBlur to reduce noise and improve circle detection
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)

        # Detect circles using HoughCircles
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2,
                minDist=30,
                param1=100, 
                param2=30,
                minRadius=10,  # Adjust based on your image size
                maxRadius=30   # Adjust based on your image size
            )

        #print("c", circles)
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
            #cv2.waitKey(0)
            return ball_dict

        # Display the output frame with circles drawn


if __name__ == '__main__':
    detector = ObjectDetector()
    #detector.camera_loop()
    frame = cv2.imread("/Users/maxortega/Downloads/Screen Shot 2024-12-10 at 6.32.30 PM.png")
    #gray_image = cv2.imread("src/vision/Screen_Shot_2024-12-10_at_6.32.30_PM.png", cv2.IMREAD_GRAYSCALE)
    detector.detect_balls_static(frame)
