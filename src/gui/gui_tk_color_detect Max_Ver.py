import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np

###ROS
import rospy
from sensor_msgs.msg import Image as Image2
from cv_bridge import CvBridge
####


import sys
import os

# Get the path to the src directory
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_dir)

from vision.object_detector_cv_only_ import ObjectDetector


class PoolRobotController:
    def __init__(self):
        rospy.init_node('pool_robot_gui', anonymous=True)
        self.root = tk.Tk()
        self.root.title("Pool Robot Controller")
        self.root.geometry("800x800")

        # Use system color names or hex color codes to change color
        self.root.configure(bg='#E6E6FA')  # Lavender color using hex code

        # Image and Ball Detection
        self.object_detector = ObjectDetector()
        self.detected_balls = {}

        # Image Display Area
        self.image_label = tk.Label(self.root)
        self.image_label.pack(padx=10, pady=10)
        self.image_label.bind("<Button-1>", self.on_image_click)

        # Load Image Button
        load_button = tk.Button(
            self.root, 
            text="Load Table Image", 
            command=self.load_and_detect_balls
        )
        load_button.pack(pady=10)

        # Selected Ball Display
        self.selected_ball = tk.StringVar(value="No Ball Selected")
        tk.Label(self.root, textvariable=self.selected_ball, font=("Arial", 16)).pack()

        #self.ball_selected = "None" #No ball will originally be selected.

        # Shot Controls
        self.setup_shot_controls()

        # Setup for streaming. 
        self.is_streaming = False
        self.cap = None
        self.screenshot = False

        
        # Add video stream button
        stream_button = tk.Button(
            self.root,
            text="Toggle Video Stream",
            command=self.toggle_video_stream
        )
        stream_button.pack(pady=10)

        # Add after existing button declarations
        screenshot_button = tk.Button(
            self.root,
            text="Take Screenshot",
            command=self.take_screenshot
        )
        screenshot_button.pack(pady=10)

        
    
        ##Integration with ROS
        self.bridge = CvBridge()
        # ROS Image Subscriber
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image2, self.ros_image_callback)

    def toggle_video_stream(self):
        self.screenshot = False
        if not self.is_streaming:
            # Start video stream
            self.cap = cv2.VideoCapture(0)
            #TODO self.cap = cv2.VideoCapture("/dev/video0")  # Linux, insert device path
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam")
                return
            self.is_streaming = True
            self.update_video_stream()
        else:
            # Stop video stream
            if self.cap:
                self.cap.release()
            self.is_streaming = False
            self.image_label.config(image='')
    
    def update_video_stream(self):
        if self.is_streaming and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # Detect balls in current frame
                self.detected_balls = self.object_detector.detect_balls(frame)
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize frame
                height, width = frame_rgb.shape[:2]
                max_width = 600
                scale = max_width / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                resized_frame = cv2.resize(frame_rgb, (new_width, new_height))
                
                # Update display
                photo = ImageTk.PhotoImage(Image.fromarray(resized_frame))
                self.image_label.config(image=photo)
                self.image_label.image = photo
                
                # Schedule next update if still streaming
                if self.is_streaming:
                    self.root.after(10, self.update_video_stream)

    def ros_image_callback(self, msg):
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.cv_image = cv_image #Record ros image in cv format
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # Store original image for resizing
            self.original_image = rgb_image
            
            # Resize and display image
            self.display_image(rgb_image)
            
            # Detect balls
            self.detected_balls = self.object_detector.detect_balls(cv_image)
        
        except Exception as e:
            rospy.logerr(f"Image conversion error: {e}")

    #ROS VIDEO STREAM 
    def display_image(self, image):
        self.is_streaming = True
        # Resize image to fit window
        height, width = image.shape[:2]
        max_width = self.root.winfo_width()
        max_height = self.root.winfo_height()
        
        scale = min(max_width/width, max_height/height)
        
        resized_image = cv2.resize(
            image, 
            (int(width * scale), int(height * scale)), 
            interpolation=cv2.INTER_AREA
        )
        
        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(Image.fromarray(resized_image))


        if not self.screenshot:
            # Update label
            self.image_label.config(image=self.photo)
            self.image_label.image = self.photo

    #Used to cv an image and label balls present in the image.
    def load_and_detect_balls(self):
        # Stop video streaming if active
        if self.is_streaming:
            self.toggle_video_stream()

        # Open file dialog to select image
        file_path = filedialog.askopenfilename(
            title="Select Pool Table Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        
        if file_path:
            # Read image
            frame = cv2.imread(file_path)
                        
            # Detect balls using original ObjectDetector method
            self.detected_balls = self.object_detector.detect_balls_static(frame)

            if self.detected_balls is None:
                print("No Ball Detected")
                return
        
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize image to fit GUI
            height, width = frame_rgb.shape[:2]
            max_width = 600
            scale = max_width / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            resized_image = cv2.resize(
                frame_rgb, 
                (new_width, new_height), 
                interpolation=cv2.INTER_AREA
            )

            self.current_image = resized_image #For access in the draw_arrows functions.

            # Store PhotoImage as instance variable
            self.photo = ImageTk.PhotoImage(Image.fromarray(resized_image))
            self.image_label.config(image=self.photo)
            
            # Scale ball coordinates to match resized image
            scaled_balls = {}
            for color, (x, y) in self.detected_balls.items():
                scaled_x = int(x * (new_width / width))
                scaled_y = int(y * (new_height / height))
                scaled_balls[color] = (scaled_x, scaled_y)
            
            # Update detected balls with scaled coordinates
            self.detected_balls = scaled_balls
            
            # Bind angle and power sliders to redraw arrow
            self.angle_slider.bind("<Motion>", self.draw_shot_arrow)
            self.power_slider.bind("<Motion>", self.draw_shot_arrow)

    #Used to click on ball of choice to choose trajectory.
    def on_image_click(self, event):
        #TODO Debug print to understand click coordinates
        print(f"Click coordinates: x={event.x}, y={event.y}")
        print("Detected Balls:", self.detected_balls)
        print(f"Image label width: {self.image_label.winfo_width()}")
        print(f"Image label height: {self.image_label.winfo_height()}")

        # Ensure image is loaded
        # Check if we have a photo reference
        if not hasattr(self, 'screenshot'):
            print("No image loaded")
            return
        

        #Get the screenshot from ros and display
        #self.photo = ImageTk.PhotoImage(Image.fromarray(self.cv_image))
        #self.image_label.config(image=self.photo)

        # Get image dimensions from the PhotoImage object
        image_width = self.photo.width()
        image_height = self.photo.height()

        # Get label dimensions
        label_width = self.image_label.winfo_width()
        label_height = self.image_label.winfo_height()

        # Calculate scaling factors for width and height separately
        width_scale = image_width / label_width
        height_scale = image_height / label_height

        # Convert click coordinates
        click_x = int(event.x * width_scale)
        click_y = int(event.y * height_scale)
        
         # Convert detected ball coordinates
        scaled_balls = {
            color: (int(x / width_scale), int(y / height_scale)) 
            for color, (x, y) in self.detected_balls.items()
        }

        # Find closest ball
        closest_ball = None
        min_distance = float('inf')

        
        for color, (ball_x, ball_y) in scaled_balls.items():
            distance = np.sqrt((click_x - ball_x)**2 + (click_y - ball_y)**2)
            if distance < min_distance:
                min_distance = distance
                closest_ball = color
    
        # Update selection if close enough
        if min_distance < 15:
            self.selected_ball.set(f"Selected: {closest_ball} Ball")
            self.ball_selected = closest_ball

        #print("ball selected: ", self.ball_selected)

    def draw_shot_arrow(self, event=None):
        # Check if we have an image and selected ball
        if not hasattr(self, 'current_image') or not hasattr(self, 'ball_selected') or self.detected_balls is None:
            return
            
        # Create a copy of the current image to draw on
        draw_image = self.current_image.copy()
        
        # Get selected ball coordinates
        if self.ball_selected in self.detected_balls:
            ball_x, ball_y = self.detected_balls[self.ball_selected]
            
            # Get angle and power from sliders
            angle = self.angle_slider.get()
            power = self.power_slider.get()
            
            # Calculate arrow endpoint
            arrow_length = power * 2  # Scale power to arrow length
            radians = np.radians(270 + angle)
            end_x = int(ball_x + arrow_length * np.cos(radians))
            end_y = int(ball_y + arrow_length * np.sin(radians))
            
            # Draw arrow on image
            cv2.arrowedLine(
                draw_image,
                (ball_x, ball_y),
                (end_x, end_y),
                (255, 0, 0),  # Red arrow
                2  # Line thickness
            )
            
            # Convert to PhotoImage and update display
            arrow_photo = ImageTk.PhotoImage(Image.fromarray(draw_image))
            self.image_label.config(image=arrow_photo)
            self.image_label.image = arrow_photo
            # Store PhotoImage as instance variable
            self.photo = arrow_photo
            self.image_label.config(image=self.photo)

    def take_screenshot(self):
        if not self.is_streaming:
            messagebox.showerror("Error", "Video stream must be active to take screenshot")
            return
        self.screenshot = True
            
        # Capture current frame
        frame = self.cv_image
            
        # Process captured frame
        self.detected_balls = self.object_detector.detect_balls_static(frame)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize image to fit GUI
        height, width = frame_rgb.shape[:2]
        max_width = 600
        scale = max_width / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        resized_image = cv2.resize(
            frame_rgb,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )
        
        # Store the current image for arrow drawing
        self.current_image = resized_image
        
        # Scale ball coordinates
        scaled_balls = {}
        for color, (x, y) in self.detected_balls.items():
            scaled_x = int(x * (new_width / width))
            scaled_y = int(y * (new_height / height))
            scaled_balls[color] = (scaled_x, scaled_y)
        
        self.detected_balls = scaled_balls
        
        # Create and store PhotoImage
        self.photo = ImageTk.PhotoImage(Image.fromarray(resized_image))
        self.image_label.config(image=self.photo)
        
        # Enable shot controls
        self.angle_slider.bind("<Motion>", self.draw_shot_arrow)
        self.power_slider.bind("<Motion>", self.draw_shot_arrow)


    def setup_shot_controls(self):
        shot_frame = tk.LabelFrame(self.root, text="Shot Controls")
        shot_frame.pack(padx=10, pady=10, fill="x")

        # Angle Slider
        tk.Label(shot_frame, text="Shot Angle").pack()
        self.angle_slider = tk.Scale(
            shot_frame, 
            from_=0, 
            to=360, 
            orient=tk.HORIZONTAL, 
            length=400
        )
        self.angle_slider.pack()

        # Power Slider
        tk.Label(shot_frame, text="Shot Power").pack()
        self.power_slider = tk.Scale(
            shot_frame, 
            from_=0, 
            to=100, 
            orient=tk.HORIZONTAL, 
            length=400
        )
        self.power_slider.pack()

        # Execute Shot Button
        execute_button = tk.Button(
            shot_frame, 
            text="Execute Shot", 
            command=self.execute_shot
        )
        execute_button.pack(pady=10)

    def execute_shot(self):
        # Gather shot parameters
        angle = self.angle_slider.get()
        power = self.power_slider.get()
        ball = self.selected_ball.get().replace("Selected: ", "").replace(" Ball", "").lower()

        # Confirmation dialog
        confirm = messagebox.askyesno(
            "Confirm Shot", 
            f"Execute shot for {ball}?\nAngle: {angle}°\nPower: {power}%"
        )

        if confirm:
            #TODO Interface with ROS CODE
            # ROS publishers to send shot parameters
            """
            # Initialize ROS node
            #rospy.init_node('pool_robot_gui', anonymous=True)
            angle_pub = rospy.Publisher('/shot/angle', Float64, queue_size=10)
            power_pub = rospy.Publisher('/shot/power', Float64, queue_size=10)
            ball_pub = rospy.Publisher('/selected/ball', String, queue_size=10)

            angle_pub.publish(angle)
            power_pub.publish(power)
            ball_pub.publish(ball)
            """

            messagebox.showinfo("Shot Executed", f"Shot for {ball} executed!")

    def run(self):
        try:
            self.root.mainloop()
        finally:
            if self.cap:
                self.cap.release()

# Run the application
if __name__ == '__main__':
    app = PoolRobotController()
    app.run()