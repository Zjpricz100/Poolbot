import wx
import cv2
import numpy as np
import os
import sys

# Add src directory to path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_dir)
from vision.object_detector_cv_only import ObjectDetector

class PoolGameFrame(wx.Frame):
    def __init__(self):
        super().__init__(parent=None, title='Pool Game Interface', size=(1200, 800))
        self.detector = ObjectDetector()
        self.detected_balls = {}
        self.selected_ball = None
        self.current_image = None
        self.init_ui()

    def init_ui(self):
        panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Left side - Image and controls
        left_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Create an initial blank bitmap
        blank_bitmap = wx.Bitmap(640, 480)
        self.image_display = wx.StaticBitmap(panel, wx.ID_ANY, blank_bitmap, size=(640, 480))
        left_sizer.Add(self.image_display, 0, wx.ALL, 5)
        
        # Load image button
        load_btn = wx.Button(panel, label='Load Image')
        load_btn.Bind(wx.EVT_BUTTON, self.on_load_image)
        left_sizer.Add(load_btn, 0, wx.ALL, 5)
        
        # Right side - Controls
        right_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Ball selection
        ball_label = wx.StaticText(panel, label="Select Ball:")
        self.ball_choice = wx.Choice(panel, choices=[])
        self.ball_choice.Bind(wx.EVT_CHOICE, self.on_ball_select)
        right_sizer.Add(ball_label, 0, wx.ALL, 5)
        right_sizer.Add(self.ball_choice, 0, wx.EXPAND|wx.ALL, 5)
        
        # Angle control
        angle_label = wx.StaticText(panel, label="Angle (degrees):")
        self.angle_slider = wx.Slider(panel, value=0, minValue=0, maxValue=359)
        self.angle_text = wx.StaticText(panel, label="0°")
        self.angle_slider.Bind(wx.EVT_SLIDER, self.on_angle_change)
        right_sizer.Add(angle_label, 0, wx.ALL, 5)
        right_sizer.Add(self.angle_slider, 0, wx.EXPAND|wx.ALL, 5)
        right_sizer.Add(self.angle_text, 0, wx.ALL, 5)
        
        # Power control
        power_label = wx.StaticText(panel, label="Power:")
        self.power_slider = wx.Slider(panel, value=50, minValue=0, maxValue=100)
        self.power_text = wx.StaticText(panel, label="50%")
        self.power_slider.Bind(wx.EVT_SLIDER, self.on_power_change)
        right_sizer.Add(power_label, 0, wx.ALL, 5)
        right_sizer.Add(self.power_slider, 0, wx.EXPAND|wx.ALL, 5)
        right_sizer.Add(self.power_text, 0, wx.ALL, 5)
        
        # Execute shot button
        shot_btn = wx.Button(panel, label='Execute Shot')
        shot_btn.Bind(wx.EVT_BUTTON, self.on_execute_shot)
        right_sizer.Add(shot_btn, 0, wx.ALL, 5)
        
        main_sizer.Add(left_sizer, 0, wx.ALL, 5)
        main_sizer.Add(right_sizer, 0, wx.ALL, 5)
        
        panel.SetSizer(main_sizer)
        self.Centre()

    def on_load_image(self, event):
        with wx.FileDialog(self, "Choose an image", wildcard="Image files (*.png;*.jpg)|*.png;*.jpg",
                         style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return
            
            path = fileDialog.GetPath()
            self.process_image(path)

    def process_image(self, path):
        print(f"Loading image from path: {path}")
        
        # Load and process image
        self.current_image = cv2.imread(path)
        if self.current_image is None:
            wx.MessageBox('Failed to load image!', 'Error', wx.OK | wx.ICON_ERROR)
            return
        
        print(f"Image shape: {self.current_image.shape}")
        
        try:
            # Create a copy for ball detection
            gray_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            self.detected_balls = self.detector.detect_balls(gray_image)
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            
            # Get dimensions
            height, width = rgb_image.shape[:2]
            print(f"Converting image of size: {width}x{height}")
            
            # Create wx.Bitmap directly from numpy array
            wx_bitmap = wx.Bitmap.FromBuffer(width, height, rgb_image)
            
            if not wx_bitmap.IsOk():
                print("Failed to create valid bitmap")
                return
                
            print("Setting bitmap to display")
            self.image_display.SetBitmap(wx_bitmap)
            
            # Update ball choices
            ball_choices = [f"Ball {i}" for i in self.detected_balls.keys()]
            self.ball_choice.Clear()
            self.ball_choice.AppendItems(ball_choices)
            
            # Force a refresh
            self.image_display.Refresh()
            self.Layout()
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            wx.MessageBox(f'Error processing image: {str(e)}', 'Error', wx.OK | wx.ICON_ERROR)


    def on_ball_select(self, event):
        selection = self.ball_choice.GetSelection()
        if selection != wx.NOT_FOUND:
            self.selected_ball = list(self.detected_balls.keys())[selection]
            # Draw circle around selected ball
            if self.current_image is not None and self.selected_ball is not None:
                img_copy = self.current_image.copy()
                ball_pos = self.detected_balls[self.selected_ball]
                cv2.circle(img_copy, ball_pos, 30, (0, 255, 0), 2)
                
                # Convert and display the updated image
                color_cv = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
                height, width = color_cv.shape[:2]
                wx_image = wx.Bitmap.FromBuffer(width, height, color_cv)
                self.image_display.SetBitmap(wx_image)

    def on_angle_change(self, event):
        value = self.angle_slider.GetValue()
        self.angle_text.SetLabel(f"{value}°")
        
        # Update image with angle indicator if a ball is selected
        if self.current_image is not None and self.selected_ball is not None:
            img_copy = self.current_image.copy()
            ball_pos = self.detected_balls[self.selected_ball]
            angle_rad = np.radians(value)
            end_point = (
                int(ball_pos[0] + 50 * np.cos(angle_rad)),
                int(ball_pos[1] - 50 * np.sin(angle_rad))
            )
            cv2.line(img_copy, ball_pos, end_point, (255, 0, 0), 2)
            
            # Convert and display the updated image
            color_cv = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
            height, width = color_cv.shape[:2]
            wx_image = wx.Bitmap.FromBuffer(width, height, color_cv)
            self.image_display.SetBitmap(wx_image)

    def on_power_change(self, event):
        value = self.power_slider.GetValue()
        self.power_text.SetLabel(f"{value}%")

    def on_execute_shot(self, event):
        if self.selected_ball is None:
            wx.MessageBox('Please select a ball first!', 'Error', wx.OK | wx.ICON_ERROR)
            return
            
        angle = self.angle_slider.GetValue()
        power = self.power_slider.GetValue()
        ball_pos = self.detected_balls[self.selected_ball]
        print(f"Executing shot on ball {self.selected_ball} at position {ball_pos} with angle {angle}° and power {power}%")

def main():
    app = wx.App()
    frame = PoolGameFrame()
    frame.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()