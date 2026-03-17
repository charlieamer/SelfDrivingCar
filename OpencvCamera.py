from kivy.core.camera import Camera as CoreCamera
from kivy.properties import NumericProperty, ListProperty, BooleanProperty
from OpencvImage import OpencvImage

class OpencvCamera(OpencvImage):
    play = BooleanProperty(False)
    index = NumericProperty(-1)
    resolution = ListProperty([-1, -1])

    def __init__(self, **kwargs):
        self._camera = None
        super(OpencvCamera, self).__init__(**kwargs)
        if self.index == -1:
            self.index = 0
        on_index = self._on_index
        fbind = self.fbind
        fbind('index', on_index)
        fbind('resolution', on_index)
        on_index()

    def on_tex(self, camera):
        self.put_image(camera.texture)

    def _on_index(self, *largs):
        self._camera = None
        if self.index < 0:
            return
        if self.resolution[0] < 0 or self.resolution[1] < 0:
            self._camera = CoreCamera(index=self.index, stopped=True)
        else:
            self._camera = CoreCamera(index=self.index,
                                      resolution=self.resolution, stopped=True)
        if self.play:
            self._camera.start()

        self._camera.bind(on_texture=self.on_tex)

    def set_is_playing(self, value):
        if not self._camera:
            return

        self.play = value
        if value:
            self._camera.start()
        else:
            self._camera.stop()
