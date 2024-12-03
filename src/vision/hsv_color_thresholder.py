#!/usr/bin/env python
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

cv2.namedWindow('image')
cv2.createTrackbar('HMin', 'image', 0, 179, lambda x: None)  # Hue is from 0-179 for OpenCV
cv2.createTrackbar('SMin', 'image', 0, 255, lambda x: None)
cv2.createTrackbar('VMin', 'image', 0, 255, lambda x: None)
cv2.createTrackbar('HMax', 'image', 0, 179, lambda x: None)
cv2.createTrackbar('SMax', 'image', 0, 255, lambda x: None)
cv2.createTrackbar('VMax', 'image', 0, 255, lambda x: None)

# Set default value for MAX HSV trackbars.
cv2.setTrackbarPos('HMax', 'image', 179)
cv2.setTrackbarPos('SMax', 'image', 255)
cv2.setTrackbarPos('VMax', 'image', 255)

# Initialize variables
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

# subscribes to the image topic that has the rgb images from the realsense
class ImageSubscriber:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/io/internal_camera/right_hand_camera/image_raw", Image, self.image_callback)
        self.img = None
        self.hsv_values = {} #Ball # (Doesn't match with actual ball # label): HSV_Values 
        self.num_balls = 0 #Number of balls

    def save_hsv_values(self, name, hsv_values):
        # Get current trackbar positions
        self.hsv_values[name] = hsv_values


    def image_callback(self, msg):
        try:
            # Convert the ROS image to OpenCV format
            self.img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)

    def process_image(self):
        print("pro")
        if self.img is None:
            return
        print("cess")
        
        
        # Get current positions of all trackbars
        hMin = cv2.getTrackbarPos('HMin', 'image')
        sMin = cv2.getTrackbarPos('SMin', 'image')
        vMin = cv2.getTrackbarPos('VMin', 'image')
        hMax = cv2.getTrackbarPos('HMax', 'image')
        sMax = cv2.getTrackbarPos('SMax', 'image')
        vMax = cv2.getTrackbarPos('VMax', 'image')
        # Set minimum and maximum HSV values for thresholding
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])
        # Convert the image to HSV and apply the threshold
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        output = cv2.bitwise_and(self.img, self.img, mask=mask)

      
        cv2.imshow('thresholded_img', output)

        # Capture key press
        key = cv2.waitKey(1) & 0xFF

        #Allows us to capture and save multiple hsv values for multiple balls
        #Make sure that when you use this you save the cue ball first for simplicity!
        
        # Save HSV values when 's' is pressed
        if key == ord('s'):
            hsv_values = {
                'hMin': hMin,
                'sMin': sMin,
                'vMin': vMin,
                'hMax': hMax,
                'sMax': sMax,
                'vMax': vMax
            }
            #Set name to be num balls and save values of the ball.
            name = str(self.num_balls)
            self.save_hsv_values(name, hsv_values)
            
            print(f"HSV values saved. Total saved: {len(self.saved_hsv_values)}")
        
        # Exit when 'q' is pressed
        if key == ord('q'):
            #Print all 16 values. 1 cue ball, balls 1-15
            for i in range(16):
                print(str(i) + ": " + self.hsv_values[str(i)]) #print the recorded hsv values.
            rospy.signal_shutdown('User requested shutdown.')
            

def main():
    """
    # Initialize the ROS node
    rospy.init_node('hsv_color_thresholder', anonymous=True)
    # Create our subscriber
    img_subscriber = ImageSubscriber()

    # Main loop keeps updating the namedwindow
    while not rospy.is_shutdown():
        img_subscriber.process_image()

    # Close all OpenCV windows after exiting
    cv2.destroyAllWindows()
    """
    # Initialize the ROS node
    rospy.init_node('hsv_color_thresholder', anonymous=True)
    img_sub = ImageSubscriber() 
    # Main loop keeps updating the namedwindow
    while not rospy.is_shutdown():
        img_sub.process_image()

    # Close all OpenCV windows after exiting
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
