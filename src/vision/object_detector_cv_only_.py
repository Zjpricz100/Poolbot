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
            grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # Apply GaussianBlur to reduce noise and improve circle detection
        blurred = cv2.GaussianBlur(grey_frame, (5, 5), 0)

        # Detect circles using HoughCircles
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.3, minDist=30,
                                    param1=100, param2=30, minRadius=0, maxRadius=20)
    
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
        ball_dict = {}
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

        # Display the output frame with circles drawn


if __name__ == '__main__':
    detector = ObjectDetector()
    #detector.camera_loop()
    frame = cv2.imread("/Users/maxortega/Downloads/Screen Shot 2024-12-10 at 6.32.30 PM.png")
    #gray_image = cv2.imread("src/vision/Screen_Shot_2024-12-10_at_6.32.30_PM.png", cv2.IMREAD_GRAYSCALE)
    detector.detect_balls_static(frame)
