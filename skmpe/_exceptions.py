# -*- coding: utf-8 -*-

from typing import List

import numpy as np

from ._base import mpe_module, PointType, FloatPointType


@mpe_module
class MPEError(Exception):
    """Base exception class for all MPE errors"""


@mpe_module
class ComputeTravelTimeError(MPEError):
    """The exception occurs when computing travel time has failed"""


@mpe_module
class PathExtractionError(MPEError):
    """Base exception class for all extracting path errors"""

    def __init__(self, *args,
                 travel_time: np.ndarray,
                 start_point: PointType,
                 end_point: PointType) -> None:
        super().__init__(*args)

        self._travel_time = travel_time
        self._start_point = start_point
        self._end_point = end_point

    @property
    def travel_time(self) -> np.ndarray:
        """Computed travel time data"""
        return self._travel_time

    @property
    def start_point(self) -> PointType:
        """Starting point"""
        return self._start_point

    @property
    def end_point(self) -> PointType:
        """Ending point"""
        return self._end_point


@mpe_module
class EndPointNotReachedError(PathExtractionError):
    """The exception occurs when the ending point is not reached"""

    def __init__(self, *args,
                 travel_time: np.ndarray,
                 start_point: PointType,
                 end_point: PointType,
                 extracted_points: List[FloatPointType],
                 last_distance: float,
                 reason: str) -> None:
        super().__init__(*args, travel_time=travel_time, start_point=start_point, end_point=end_point)

        self._extracted_points = extracted_points
        self._last_distance = last_distance
        self._reason = reason

    @property
    def extracted_points(self) -> List[FloatPointType]:
        """The list of extracted path points"""
        return self._extracted_points

    @property
    def last_distance(self) -> float:
        """The last distance to the ending point from the last path point"""
        return self._last_distance

    @property
    def reason(self) -> str:
        """The reason of extracting path termination"""
        return self._reason
