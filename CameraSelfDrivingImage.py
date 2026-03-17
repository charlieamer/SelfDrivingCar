from OpencvCamera import OpencvCamera
from SelfDrivingCar import SelfDrivingCar

class CameraSelfDrivingImage(OpencvCamera):
    def __init__(self, self_driving_car: SelfDrivingCar, **kwargs):
        super().__init__(**kwargs, play=True)
        self.self_driving_car = self_driving_car
    
    def _process_image(self, frame):
        return self.self_driving_car.process_from_image(frame)
