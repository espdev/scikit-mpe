# -*- coding: utf-8 -*-

from typing import List

import numpy as np

from ._base import PointType, FloatPointType


class MPEError(Exception):
    pass


class ComputeTravelTimeError(MPEError):
    pass


class ExtractPathError(MPEError):
    def __init__(self, *args,
                 travel_time: np.ndarray,
                 start_point: PointType,
                 end_point: PointType) -> None:
        super().__init__(*args)

        self.travel_time = travel_time
        self.start_point = start_point
        self.end_point = end_point


class EndPointNotReachedError(ExtractPathError):
    def __init__(self, *args,
                 travel_time: np.ndarray,
                 start_point: PointType,
                 end_point: PointType,
                 extracted_points: List[FloatPointType],
                 last_distance: float) -> None:
        super().__init__(*args, travel_time=travel_time, start_point=start_point, end_point=end_point)

        self.extracted_points = extracted_points
        self.last_distance = last_distance
