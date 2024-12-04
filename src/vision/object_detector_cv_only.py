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
        'cue': (np.array([0, 0, 110]), np.array([80, 102, 255])),
        'yellow': (np.array([26, 237, 122]), np.array([63, 255, 255])),
        'blue': (np.array([78, 103, 28]), np.array([119, 216, 121])),
        'red': (np.array([0, 220, 113]), np.array([9, 248, 231])),
        'purple': (np.array([115, 0, 0]), np.array([175, 222, 99])),
        'orange': (np.array([0, 246, 100]), np.array([25, 255, 255])),
        'maroon': (np.array([4, 120, 31]), np.array([13, 237, 180])),
        'black': (np.array([0, 0, 30]), np.array([179, 255, 54])),
        'green': (np.array([31, 102, 0]), np.array([105, 255, 123])),
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
            self.detect_balls(frame)
    
    def detect_balls(self, frame):
        # Convert frame to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Loop through all color ranges in the dictionary
        for color_name, (lower_hsv, upper_hsv) in self.ball_numbers.items():
            # Create mask for the specific color range
            mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

            # Mask the frame to isolate the color
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

            # Convert the masked image to grayscale
            gray_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to reduce noise and improve circle detection
            blurred_frame = cv2.GaussianBlur(gray_frame, (25, 25), 0)
            edges = cv2.Canny(blurred_frame, 50, 150)
            # Use HoughCircles to detect circles in the blurred grayscale image
            circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.5, minDist=20,
                            param1=200, param2=35, minRadius=15, maxRadius=60)

            # If circles are detected, draw them
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    # Draw the outer circle
                    cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
                    # Draw the center of the circle
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
                    # Add label for the color of the ball
                    cv2.putText(frame, color_name, (x - r, y - r - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display the final image with detected balls
        cv2.imshow("Ball Detection", frame)
        cv2.waitKey(1)

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

        # Process each ball color
        for color_name, (lower_hsv, upper_hsv) in ball_numbers.items():
            # Create mask for this color
            mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Initialize a list to store circularity and contour information
            ball_candidates = []

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
                        if circularity > 0.6:
                            # Additional shape verification using minimum enclosing circle
                            (x, y), radius = cv2.minEnclosingCircle(contour)
                            
                            # Verify the object is roughly circular
                            if radius > 0.5:  # Minimum ball size
                                ball_candidates.append((color_name, circularity, contour))

            # Sort the candidates by circularity and select the two most circular ones
            ball_candidates = sorted(ball_candidates, key=lambda x: x[1], reverse=True)[:2]

            # Draw and label the most circular balls
            for candidate in ball_candidates:
                color_name, circularity, contour = candidate
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center_x = int(x)
                center_y = int(y)
                
                # Draw the contour and label the ball
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
    detector = ObjectDetector()
    detector.camera_loop()

