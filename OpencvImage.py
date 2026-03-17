from kivy.uix.image import Image
import cv2
from kivy.graphics.texture import Texture
import numpy as np

class OpencvImage(Image):
    def put_image(self, image_input):
        """
        Public function to accept an OpenCV image, process it, and display in the widget.
        """
        # Check if the input is a Kivy Texture and convert to OpenCV image if so
        if isinstance(image_input, Texture):
            cv_image = self._texture_to_cv(image_input)
        else:
            cv_image = image_input

        # Process the image
        processed_image = self._process_image(cv_image)
        
        # Convert the processed image to texture and display it
        self._display_image(processed_image)

    def _process_image(self, frame):
        """
        Private function to process the OpenCV image. 
        This function should be overridden to perform custom processing.
        """
        # Example processing: convert to grayscale
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return processed_frame

    def _display_image(self, cv_image):
        """
        Convert the OpenCV image to a Kivy Texture and display it in the widget.
        """
        if len(cv_image.shape) == 2:  # Grayscale image
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
        buffer = cv2.flip(cv_image, 0).tobytes()

        # if we don't have a texture, or the existing texture has different dimensions
        if self.texture is None or self.texture.size != (cv_image.shape[1], cv_image.shape[0]):
            # Create new texture and assign it
            texture = Texture.create(size=(cv_image.shape[1], cv_image.shape[0]), colorfmt='bgr')
            self.texture = texture

        self.texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')

        self.canvas.ask_update()

    def _texture_to_cv(self, texture):
        """
        Convert a Kivy Texture to an OpenCV Image.
        """
        size = texture.size
        data = texture.pixels
        # Create an array from the pixel data
        arr = np.frombuffer(data, dtype=np.uint8).reshape(size[1], size[0], 4)
        # Convert from RGBA to BGR format which OpenCV uses
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        return arr

# Usage example (not executable here since it requires an actual OpenCV image):
# opencv_img_widget = OpencvImage()
# cv_image = cv2.imread('path_to_image.jpg')
# opencv_img_widget.put_image(cv_image)
