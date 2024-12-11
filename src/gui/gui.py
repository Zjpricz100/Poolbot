import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np


import sys
import os

# Get the path to the src directory
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_dir)

from vision.object_detector_cv_only import ObjectDetector


class PoolRobotController:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Pool Robot Controller")
        self.root.geometry("800x800")


        # Method 3: Use system color names or hex color codes
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

        # Shot Controls
        self.setup_shot_controls()

    


    #Used to cv an image and label balls present in the image.
    def load_and_detect_balls(self):
        # Open file dialog to select image
        file_path = filedialog.askopenfilename(
            title="Select Pool Table Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        
        if file_path:
            # Read image
            frame = cv2.imread(file_path)
            
            # Detect balls using original ObjectDetector method
            self.detected_balls = self.object_detector.detect_balls(frame)
            
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
            
            # Scale ball coordinates to match resized image
            scaled_balls = {}
            for color, (x, y) in self.detected_balls.items():
                scaled_x = int(x * (new_width / width))
                scaled_y = int(y * (new_height / height))
                scaled_balls[color] = (scaled_x, scaled_y)
            
            # Update detected balls with scaled coordinates
            self.detected_balls = scaled_balls
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(Image.fromarray(resized_image))
            
            # Update label
            self.image_label.config(image=photo)
            self.image_label.image = photo  # Keep a reference


            ########## Arrow code implementation ############
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(Image.fromarray(resized_image))
            
            # Update label
            self.image_label.config(image=photo)
            self.image_label.image = photo  # Keep a reference

            # Add method to draw arrow
            def draw_shot_arrow(event):
                # Create a copy of the image to draw on
                draw_image = resized_image.copy()
                
                # Get selected ball coordinates
                if hasattr(self, 'ball_selected'):
                    ball_x, ball_y = self.detected_balls[self.ball_selected]
                    
                    # Get angle and power from sliders
                    angle = self.angle_slider.get()
                    power = self.power_slider.get()
                    
                    # Calculate arrow endpoint
                    arrow_length = power * 2  # Scale power to arrow length
                    radians = np.radians(270 - angle)
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
                    
                    # Convert back to PhotoImage
                    arrow_photo = ImageTk.PhotoImage(Image.fromarray(draw_image))
                    self.image_label.config(image=arrow_photo)
                    self.image_label.image = arrow_photo

            # Bind angle and power sliders to redraw arrow
            self.angle_slider.bind("<Motion>", draw_shot_arrow)
            self.power_slider.bind("<Motion>", draw_shot_arrow)

            # Update label
            self.image_label.config(image=photo)
            self.image_label.image = photo  # Keep a reference


    #Used to click on ball of choice to choose trajectory.
    def on_image_click(self, event):
        #TODO Debug print to understand click coordinates
        print(f"Click coordinates: x={event.x}, y={event.y}")
        print("Detected Balls:", self.detected_balls)
        print(f"Image label width: {self.image_label.winfo_width()}")
        print(f"Image label height: {self.image_label.winfo_height()}")

        # Ensure image is loaded
        if not hasattr(self.image_label, 'image'):
            return

        # Get original image dimensions
        image_width = self.image_label.image.width()
        image_height = self.image_label.image.height()

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
            self.selected_ball.set(f"Selected: {closest_ball.capitalize()} Ball")
            self.ball_selected = closest_ball

        print("ball selected: ", self.selected_ball)

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
            angle_pub = rospy.Publisher('/shot/angle', Float64, queue_size=10)
            power_pub = rospy.Publisher('/shot/power', Float64, queue_size=10)
            ball_pub = rospy.Publisher('/selected/ball', String, queue_size=10)

            angle_pub.publish(angle)
            power_pub.publish(power)
            ball_pub.publish(ball)
            """

            messagebox.showinfo("Shot Executed", f"Shot for {ball} executed!")

    def run(self):
        self.root.mainloop()

# Run the application
if __name__ == '__main__':
    app = PoolRobotController()
    app.run()
