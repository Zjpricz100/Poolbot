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
        self.root.geometry("800x700")

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
        tk.Label(self.root, textvariable=self.selected_ball, font=("Arial", 12)).pack()

        # Shot Controls
        self.setup_shot_controls()

    def load_and_detect_balls(self):
        # Open file dialog to select image
        file_path = filedialog.askopenfilename(
            title="Select Pool Table Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        
        if file_path:
            # Read image
            frame = cv2.imread(file_path)
            
            # Detect balls
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
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(Image.fromarray(resized_image))
            
            # Update label
            self.image_label.config(image=photo)
            self.image_label.image = photo  # Keep a reference

    def on_image_click(self, event):
        # Debug print to understand click coordinates
        print(f"Click coordinates: x={event.x}, y={event.y}")
        print(f"Image label width: {self.image_label.winfo_width()}")
        print(f"Image width: {self.image_label.image.width}")

        # Scale factor to convert GUI coordinates back to original image
        scale_factor = self.image_label.winfo_width() / self.image_label.image.width()
        
        # Convert click coordinates
        click_x = int(event.x / scale_factor)
        click_y = int(event.y / scale_factor)

        # Find closest ball
        closest_ball = None
        min_distance = float('inf')
        
        for color, (ball_x, ball_y) in self.detected_balls.items():
            distance = np.sqrt((click_x - ball_x)**2 + (click_y - ball_y)**2)
            if distance < min_distance:
                min_distance = distance
                closest_ball = color

        # Update selected ball if within reasonable proximity
        if min_distance < 50:  # Adjust threshold as needed
            self.selected_ball.set(f"Selected: {closest_ball.capitalize()} Ball")
            self.ball_selected = closest_ball

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
            messagebox.showinfo("Shot Executed", f"Shot for {ball} executed!")

    def run(self):
        self.root.mainloop()

# Run the application
if __name__ == '__main__':
    app = PoolRobotController()
    app.run()
