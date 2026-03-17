from OpencvImage import OpencvImage
from kivy.properties import NumericProperty
import cv2
from kivy.clock import Clock
from SelfDrivingCar import SelfDrivingCar
import os.path

class VideoSelfDrivingImage(OpencvImage):
    # Define skip_frames as a NumericProperty
    skip_frames = NumericProperty(0)

    def __init__(self, video_path: str, self_driving_car: SelfDrivingCar, **kwargs):
        super().__init__(**kwargs)
        if not os.path.exists(video_path):
            raise FileNotFoundError(video_path)

        self.video_path = video_path
        self.self_driving_car = self_driving_car
        self.capture = cv2.VideoCapture(self.video_path)
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open video: {self.video_path}")

        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

        # Retrieve the frame rate of the video
        self.frame_rate = self.capture.get(cv2.CAP_PROP_FPS)
        if self.frame_rate == 0:  # Prevent division by zero if frame rate is not available
            self.frame_rate = 30  # Default to 30 FPS if unable to retrieve actual frame rate
        
        # Calculate the interval for scheduling updates based on the frame rate
        self.update_interval = 1 / self.frame_rate

        # Skip the requested number of frames, but stay within the actual video.
        if self.frame_count > 0:
            self.skip_frames = min(self.skip_frames, self.frame_count - 1)
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, int(self.skip_frames))
        
        # Schedule the frame updates with the calculated interval
        Clock.schedule_interval(self.update_frame, self.update_interval)

    def update_frame(self, dt):
        ret, frame = self.capture.read()
        if ret:
            self.put_image(frame)
        else:
            Clock.unschedule(self.update_frame)
            self.capture.release()
    
    def _process_image(self, frame):
        # Now using the actual frame rate for processing
        return self.self_driving_car.process_from_image(frame, self.update_interval)
