import argparse
import os

os.environ.setdefault("KIVY_NO_ARGS", "1")

from kivy.app import App
from CameraSelfDrivingImage import CameraSelfDrivingImage
from SelfDrivingCar import SelfDrivingCar
from VideoSelfDrivingImage import VideoSelfDrivingImage


class CamApp(App):
    def __init__(
        self,
        video_path=None,
        detector_rows=None,
        center_x=None,
        threshold_offset=45,
        min_bright_width=12,
        smoothing_window=15,
        min_dark_light_jump=25,
        point_move_speed=12,
        stale_distance=25,
        search_half_width=None,
        lowest_detector_offset=0,
        highest_detector_offset=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.self_driving_car = SelfDrivingCar(
            detector_rows=detector_rows,
            ideal_center_x=center_x,
            threshold_offset=threshold_offset,
            min_bright_width=min_bright_width,
            smoothing_window=smoothing_window,
            min_dark_light_jump=min_dark_light_jump,
            point_move_speed=point_move_speed,
            stale_distance=stale_distance,
            search_half_width=search_half_width,
            lowest_detector_offset=lowest_detector_offset,
            highest_detector_offset=highest_detector_offset,
        )
        self.video_path = video_path

    def _get_camera_widget(self):
        return CameraSelfDrivingImage(self.self_driving_car)

    def _get_static_video_widget(self):
        return VideoSelfDrivingImage(
            self.video_path,
            self.self_driving_car,
            skip_frames=0,
        )

    def build(self):
        return self._get_static_video_widget()


def parse_detector_rows(value: str) -> list[int]:
    rows: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        rows.append(int(part))
    if not rows:
        raise argparse.ArgumentTypeError("At least one detector row is required")
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument(
        "--detectors",
        type=parse_detector_rows,
        default=[650],
        help="Comma-separated list of y coordinates for scanline detectors",
    )
    parser.add_argument(
        "--center-x",
        type=int,
        default=None,
        help="Ideal lane center x coordinate in pixels. Defaults to image center.",
    )
    parser.add_argument(
        "--threshold-offset",
        type=int,
        default=45,
        help="Brightness above the row median required to count as a bright lane segment.",
    )
    parser.add_argument(
        "--min-bright-width",
        type=int,
        default=12,
        help="Minimum width in pixels for a bright segment to count as a lane marking.",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=15,
        help="Odd-sized moving-average window used to smooth each detector row.",
    )
    parser.add_argument(
        "--min-dark-light-jump",
        type=int,
        default=25,
        help="Minimum brightness jump from surrounding dark road into the bright segment.",
    )
    parser.add_argument(
        "--point-move-speed",
        type=int,
        default=12,
        help="Maximum pixels a tracked point may move toward the new detection per frame.",
    )
    parser.add_argument(
        "--stale-distance",
        type=int,
        default=25,
        help="If the tracked point is farther than this from the raw detection, ignore it in center calculations and draw it pink.",
    )
    parser.add_argument(
        "--search-half-width",
        type=int,
        default=None,
        help="Half-width of each search band around the interpolated detector offset.",
    )
    parser.add_argument(
        "--lowest-detector-offset",
        type=int,
        default=0,
        help="Offset from center for the lowest detector search location.",
    )
    parser.add_argument(
        "--highest-detector-offset",
        type=int,
        default=0,
        help="Offset from center for the highest detector search location.",
    )
    args = parser.parse_args()
    CamApp(
        video_path=args.video_path,
        detector_rows=args.detectors,
        center_x=args.center_x,
        threshold_offset=args.threshold_offset,
        min_bright_width=args.min_bright_width,
        smoothing_window=args.smoothing_window,
        min_dark_light_jump=args.min_dark_light_jump,
        point_move_speed=args.point_move_speed,
        stale_distance=args.stale_distance,
        search_half_width=args.search_half_width,
        lowest_detector_offset=args.lowest_detector_offset,
        highest_detector_offset=args.highest_detector_offset,
    ).run()
