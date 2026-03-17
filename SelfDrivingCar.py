import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ScanlineDetection:
    y: int
    bright_segments: list[tuple[int, int]]
    left_lane_x: int | None
    right_lane_x: int | None
    raw_left_lane_x: int | None
    raw_right_lane_x: int | None
    left_usable: bool = False
    right_usable: bool = False

    @property
    def lane_center_x(self) -> int | None:
        if not self.left_usable or not self.right_usable:
            return None
        if self.left_lane_x is None or self.right_lane_x is None:
            return None
        return (self.left_lane_x + self.right_lane_x) // 2


class ScanlineDetector:
    def __init__(
        self,
        y: int,
        threshold_offset: int = 45,
        min_bright_width: int = 12,
        smoothing_window: int = 15,
        min_dark_light_jump: int = 25,
        search_half_width: int | None = None,
    ):
        self.y = y
        self.threshold_offset = threshold_offset
        self.min_bright_width = min_bright_width
        self.smoothing_window = smoothing_window
        self.min_dark_light_jump = min_dark_light_jump
        self.search_half_width = search_half_width

    def detect(
        self,
        gray_image: np.ndarray,
        ideal_center_x: int,
        search_center_offset: int,
    ) -> ScanlineDetection:
        height, width = gray_image.shape
        y = min(max(self.y, 0), height - 1)
        row = gray_image[y].astype(np.float32)
        smoothed = self._smooth(row)
        threshold = min(255.0, float(np.median(smoothed) + self.threshold_offset))
        bright_mask = smoothed >= threshold
        segments = self._find_bright_segments(bright_mask, smoothed)
        min_offset, max_offset = self._search_offsets(width, search_center_offset)

        left_segments = [
            segment
            for segment in segments
            if ideal_center_x - max_offset <= self._segment_center(segment) <= ideal_center_x - min_offset
        ]
        right_segments = [
            segment
            for segment in segments
            if ideal_center_x + min_offset <= self._segment_center(segment) <= ideal_center_x + max_offset
        ]

        left_lane = max(left_segments, key=self._segment_center, default=None)
        right_lane = min(right_segments, key=self._segment_center, default=None)
        raw_left_lane_x = None if left_lane is None else self._segment_center(left_lane)
        raw_right_lane_x = None if right_lane is None else self._segment_center(right_lane)

        return ScanlineDetection(
            y=y,
            bright_segments=segments,
            left_lane_x=raw_left_lane_x,
            right_lane_x=raw_right_lane_x,
            raw_left_lane_x=raw_left_lane_x,
            raw_right_lane_x=raw_right_lane_x,
        )

    def _search_offsets(self, image_width: int, search_center_offset: int) -> tuple[int, int]:
        default_max = image_width // 4
        half_width = default_max if self.search_half_width is None else max(0, self.search_half_width)
        max_offset = min(default_max, search_center_offset + half_width)
        min_offset = max(0, search_center_offset - half_width)
        min_offset = min(min_offset, max_offset)
        return min_offset, max_offset

    def _smooth(self, row: np.ndarray) -> np.ndarray:
        kernel_size = max(1, self.smoothing_window)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
        return np.convolve(row, kernel, mode="same")

    def _find_bright_segments(
        self,
        bright_mask: np.ndarray,
        smoothed_row: np.ndarray,
    ) -> list[tuple[int, int]]:
        transitions = np.diff(bright_mask.astype(np.int8))
        starts = list(np.where(transitions == 1)[0] + 1)
        ends = list(np.where(transitions == -1)[0] + 1)

        if bright_mask[0]:
            starts.insert(0, 0)
        if bright_mask[-1]:
            ends.append(len(bright_mask))

        segments: list[tuple[int, int]] = []
        for start, end in zip(starts, ends):
            if end - start < self.min_bright_width:
                continue

            segment = (int(start), int(end - 1))
            if not self._has_strong_contrast(segment, smoothed_row):
                continue

            segments.append(segment)
        return segments

    def _has_strong_contrast(self, segment: tuple[int, int], smoothed_row: np.ndarray) -> bool:
        start, end = segment
        lane_slice = smoothed_row[start:end + 1]
        shoulder = max(3, self.smoothing_window // 2)

        left_outer_start = max(0, start - shoulder)
        left_outer_end = start
        right_outer_start = end + 1
        right_outer_end = min(len(smoothed_row), end + 1 + shoulder)

        if left_outer_end <= left_outer_start or right_outer_end <= right_outer_start:
            return False

        lane_mean = float(np.mean(lane_slice))
        left_mean = float(np.mean(smoothed_row[left_outer_start:left_outer_end]))
        right_mean = float(np.mean(smoothed_row[right_outer_start:right_outer_end]))

        return (
            lane_mean - left_mean >= self.min_dark_light_jump
            and lane_mean - right_mean >= self.min_dark_light_jump
        )

    def _segment_center(self, segment: tuple[int, int]) -> int:
        start, end = segment
        return (start + end) // 2


class SelfDrivingCar:
    def __init__(
        self,
        detector_rows: list[int] | None = None,
        ideal_center_x: int | None = None,
        threshold_offset: int = 45,
        min_bright_width: int = 12,
        smoothing_window: int = 15,
        min_dark_light_jump: int = 25,
        point_move_speed: int = 12,
        stale_distance: int = 25,
        search_half_width: int | None = None,
        lowest_detector_offset: int = 0,
        highest_detector_offset: int = 0,
    ):
        self.previous_frame_timestamp: float | None = None
        self.dt = 1 / 30
        self.ideal_center_x = ideal_center_x
        self.point_move_speed = point_move_speed
        self.stale_distance = stale_distance
        self.lowest_detector_offset = lowest_detector_offset
        self.highest_detector_offset = highest_detector_offset
        self.detectors = [
            ScanlineDetector(
                y,
                threshold_offset=threshold_offset,
                min_bright_width=min_bright_width,
                smoothing_window=smoothing_window,
                min_dark_light_jump=min_dark_light_jump,
                search_half_width=search_half_width,
            )
            for y in (detector_rows or [700])
        ]
        self.smoothed_points: dict[tuple[int, str], int | None] = {}
        self._reset_print()

    def process_from_image(self, image, dt: float | None = None):
        self._reset_print()
        self._update_dt(dt)

        output = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        frame_center_x = image.shape[1] // 2
        ideal_center_x = self._resolve_ideal_center(image.shape[1], frame_center_x)
        detector_offsets = self._detector_offsets()

        detections = [
            self._apply_point_smoothing(
                detector.detect(gray, ideal_center_x, offset),
                ideal_center_x,
                image.shape[1],
            )
            for detector, offset in zip(self.detectors, detector_offsets)
        ]
        output = self._draw_visualization(output, detections, ideal_center_x, detector_offsets)

        valid_centers = [detection.lane_center_x for detection in detections if detection.lane_center_x is not None]
        position_offset = None
        if valid_centers:
            current_center_x = int(round(sum(valid_centers) / len(valid_centers)))
            position_offset = current_center_x - ideal_center_x

        self._print(output, "Frame: ", f"{image.shape[1]}x{image.shape[0]}")
        self._print(output, "Detectors: ", len(self.detectors))
        self._print(output, "Ideal center: ", ideal_center_x)
        if position_offset is None:
            self._print(output, "Current center: ", "n/a")
        else:
            self._print(output, "Offset: ", position_offset, "pixels")

        return output

    def _resolve_ideal_center(self, image_width: int, fallback_center_x: int) -> int:
        if self.ideal_center_x is None:
            return fallback_center_x
        return min(max(self.ideal_center_x, 0), image_width - 1)

    def _draw_visualization(
        self,
        image: np.ndarray,
        detections: list[ScanlineDetection],
        ideal_center_x: int,
        detector_offsets: list[int],
    ) -> np.ndarray:
        height, width = image.shape[:2]

        for detector, detection, search_center_offset in zip(self.detectors, detections, detector_offsets):
            y = detection.y
            min_offset, max_offset = detector._search_offsets(width, search_center_offset)
            left_search_start = max(0, ideal_center_x - max_offset)
            left_search_end = max(0, ideal_center_x - min_offset)
            right_search_start = min(width - 1, ideal_center_x + min_offset)
            right_search_end = min(width - 1, ideal_center_x + max_offset)

            cv2.line(image, (left_search_start, y), (left_search_end, y), (0, 0, 0), 2)
            cv2.line(image, (right_search_start, y), (right_search_end, y), (0, 0, 0), 2)
            cv2.line(image, (ideal_center_x, max(0, y - 8)), (ideal_center_x, min(height - 1, y + 8)), (0, 0, 0), 2)

            left_fallback_x = left_search_end
            right_fallback_x = right_search_start
            left_x = detection.left_lane_x if detection.left_lane_x is not None else left_fallback_x
            right_x = detection.right_lane_x if detection.right_lane_x is not None else right_fallback_x
            left_color = self._point_color(detection.left_lane_x, detection.raw_left_lane_x)
            right_color = self._point_color(detection.right_lane_x, detection.raw_right_lane_x)

            cv2.circle(image, (left_x, y), 8, left_color, -1)
            cv2.circle(image, (right_x, y), 8, right_color, -1)

            tracked_center_x = self._tracked_center_x(detection)
            if tracked_center_x is not None:
                center_color = (255, 0, 0) if detection.lane_center_x is not None else (140, 0, 0)
                cv2.circle(image, (tracked_center_x, y), 6, center_color, -1)

        return image

    def _detector_offsets(self) -> list[int]:
        if not self.detectors:
            return []
        if len(self.detectors) == 1:
            return [self.lowest_detector_offset]

        ys = [detector.y for detector in self.detectors]
        lowest_y = max(ys)
        highest_y = min(ys)
        if lowest_y == highest_y:
            return [self.lowest_detector_offset for _ in self.detectors]

        offsets: list[int] = []
        for detector in self.detectors:
            progress = (lowest_y - detector.y) / (lowest_y - highest_y)
            interpolated = self.lowest_detector_offset + progress * (
                self.highest_detector_offset - self.lowest_detector_offset
            )
            offsets.append(int(round(interpolated)))
        return offsets

    def _apply_point_smoothing(
        self,
        detection: ScanlineDetection,
        ideal_center_x: int,
        image_width: int,
    ) -> ScanlineDetection:
        left_start_x = max(0, ideal_center_x - image_width // 8)
        right_start_x = min(image_width - 1, ideal_center_x + image_width // 8)
        left_lane_x = self._smooth_point(("left", detection.y), detection.raw_left_lane_x, left_start_x)
        right_lane_x = self._smooth_point(("right", detection.y), detection.raw_right_lane_x, right_start_x)
        left_usable = self._is_point_usable(left_lane_x, detection.raw_left_lane_x)
        right_usable = self._is_point_usable(right_lane_x, detection.raw_right_lane_x)

        return ScanlineDetection(
            y=detection.y,
            bright_segments=detection.bright_segments,
            left_lane_x=left_lane_x,
            right_lane_x=right_lane_x,
            raw_left_lane_x=detection.raw_left_lane_x,
            raw_right_lane_x=detection.raw_right_lane_x,
            left_usable=left_usable,
            right_usable=right_usable,
        )

    def _smooth_point(self, key: tuple[str, int], raw_x: int | None, default_x: int) -> int | None:
        previous_x = self.smoothed_points.get(key)
        if raw_x is None:
            if previous_x is None:
                self.smoothed_points[key] = default_x
                return default_x
            self.smoothed_points[key] = previous_x
            return previous_x
        if previous_x is None:
            previous_x = default_x

        delta = raw_x - previous_x
        if abs(delta) <= self.point_move_speed:
            new_x = previous_x + delta
        else:
            step = self.point_move_speed if delta > 0 else -self.point_move_speed
            new_x = previous_x + step
        self.smoothed_points[key] = new_x
        return new_x

    def _is_point_usable(self, smoothed_x: int | None, raw_x: int | None) -> bool:
        if smoothed_x is None or raw_x is None:
            return False
        return abs(smoothed_x - raw_x) <= self.stale_distance

    def _point_color(self, smoothed_x: int | None, raw_x: int | None) -> tuple[int, int, int]:
        if smoothed_x is None:
            return (0, 0, 255)
        if raw_x is None:
            return (0, 0, 255)
        if not self._is_point_usable(smoothed_x, raw_x):
            return (255, 0, 255)
        return (0, 255, 0)

    def _tracked_center_x(self, detection: ScanlineDetection) -> int | None:
        if detection.left_lane_x is None or detection.right_lane_x is None:
            return None
        return (detection.left_lane_x + detection.right_lane_x) // 2

    def _update_dt(self, dt: float | None):
        if dt is None:
            if self.previous_frame_timestamp is None:
                self.previous_frame_timestamp = time.time() - 1 / 30
            self.dt = time.time() - self.previous_frame_timestamp
        else:
            self.dt = dt
        self.previous_frame_timestamp = time.time()

    def _reset_print(self):
        self.current_print_line = 0

    def _print(self, frame, *args):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        line_type = 2
        line_height = 24
        x = 10
        for word in args:
            if isinstance(word, float):
                word = "%.1f" % word
                font_color = (200, 50, 50)
            elif isinstance(word, int):
                word = str(word)
                font_color = (50, 200, 50)
            else:
                font_color = (255, 255, 255)
                word = str(word)
            position = (x, 30 + self.current_print_line * line_height)
            x += (len(word) + 1) * 12
            cv2.putText(frame, word, position, font, font_scale, font_color, line_type)
        self.current_print_line += 1
