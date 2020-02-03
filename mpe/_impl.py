# -*- coding: utf-8 -*-

import abc
import itertools
from typing import Optional, Union, List

import numpy as np
import skfmm as fmm
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.distance import euclidean

from ._base import mpe as mpe_impl_dispatch

from ._base import (
    PointType,
    FloatPointType,
    WayPointsType,
    InitialInfo,
    PathInfo,
    ResultPathInfo,
    logger,
)

from ._parameters import (
    ExtractPointUpdateMethod,
    Parameters,
)

from ._exceptions import (
    ComputeTravelTimeError,
    EndPointNotReachedError,
)


class PathPointCalculatorBase(abc.ABC):

    def __init__(self, travel_time: np.ndarray, grid_spacing: float):
        grid_coords = [np.arange(n) for n in travel_time.shape]

        gradients = np.gradient(travel_time, grid_spacing)
        gradient_interpolants = []

        for gradient in gradients:
            if isinstance(gradient, np.ma.MaskedArray):
                # unmasking masked gradient values (we assume the mask is not hard)
                gradient[gradient.mask] = 0.0

            interpolant = RegularGridInterpolator(
                grid_coords, gradient, method='linear', bounds_error=True)

            gradient_interpolants.append(interpolant)

        self.gradient_interpolants = gradient_interpolants
        self.grid_spacing = grid_spacing

    def __call__(self, point: Union[PointType, FloatPointType]) -> FloatPointType:
        return tuple(self.update(np.array(point, dtype=np.float_)))

    def normalized_velocity(self, point: np.ndarray) -> np.ndarray:
        velocity = np.array([gi(point).squeeze() for gi in self.gradient_interpolants])
        return velocity / np.linalg.norm(velocity)

    @abc.abstractmethod
    def update(self, point: np.ndarray) -> np.ndarray:
        pass


class EulerPathPointCalculator(PathPointCalculatorBase):
    """First order method (Euler's method)
    """

    def update(self, point: np.ndarray) -> np.ndarray:
        return point - self.normalized_velocity(point) * self.grid_spacing


class RungeKuttaPathPointCalculator(PathPointCalculatorBase):
    """Fourth order Runge-Kutta method
    """

    def update(self, point: np.ndarray) -> np.ndarray:
        grid_spacing = self.grid_spacing
        normalized_velocity = self.normalized_velocity

        k1 = grid_spacing * normalized_velocity(point)
        k2 = grid_spacing * normalized_velocity(point - k1 / 2.0)
        k3 = grid_spacing * normalized_velocity(point - k2 / 2.0)
        k4 = grid_spacing * normalized_velocity(point - k3)

        return point - (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


point_update_methods = {
    ExtractPointUpdateMethod.euler: EulerPathPointCalculator,
    ExtractPointUpdateMethod.runge_kutta: RungeKuttaPathPointCalculator,
}


def compute_travel_time(speed_data: np.ndarray,
                        end_point: PointType,
                        parameters: Parameters):
    # define zero contour and set the wave source
    phi = np.ones_like(speed_data)
    phi[end_point] = -1

    try:
        return fmm.travel_time(phi, speed_data,
                               dx=parameters.fmm_grid_spacing,
                               order=parameters.fmm_order)
    except Exception as err:
        raise ComputeTravelTimeError from err


def extract_path(travel_time: np.ndarray,
                 start_point: PointType,
                 end_point: PointType,
                 parameters: Parameters) -> np.ndarray:
    point_update_method = parameters.extract_point_update_method
    grid_spacing = parameters.extract_grid_spacing
    max_iterations = parameters.extract_max_iterations

    update = point_update_methods[point_update_method](travel_time, grid_spacing)

    end_point_reached = False
    current_point = start_point
    path_points = [current_point]

    for i in range(max_iterations):
        # on each iteration we compute the next path point: compute the path curve evolution
        current_point = update(current_point)
        distance = euclidean(current_point, end_point)

        logger.debug(
            'iteration %d: current_point=%s, distance=%.2f', i + 1, current_point, distance)

        if distance > grid_spacing:
            path_points.append(current_point)
        else:
            path_points.append(end_point)
            end_point_reached = True
            logger.debug(
                'The minimal path has been extracted in %d iterations', i + 1)
            break

    if not end_point_reached:
        last_distance = euclidean(current_point, end_point)

        err_msg = (
            f'The extracted path from the start point {start_point} '
            f'did not reach the end point {end_point} in {max_iterations} iterations '
            f'with distance {last_distance}.'
        )

        raise EndPointNotReachedError(
            err_msg,
            travel_time=travel_time,
            start_point=start_point,
            end_point=end_point,
            extracted_points=path_points,
            last_distance=last_distance,
        )

    return np.array(path_points, dtype=np.float_)


def extract_path_without_way_points(init_info: InitialInfo,
                                    parameters: Parameters) -> ResultPathInfo:
    travel_time = compute_travel_time(
        init_info.speed_data, init_info.end_point, parameters)

    path = extract_path(
        travel_time, init_info.start_point, init_info.end_point, parameters)

    path_info = PathInfo(
        path=path,
        start_point=init_info.start_point,
        end_point=init_info.end_point,
        travel_time=travel_time,
        reversed=False,
    )

    return ResultPathInfo(path=path, pieces=[path_info])


def make_whole_path_from_pieces(path_pieces_info: List[PathInfo]) -> ResultPathInfo:
    path_pieces = [path_pieces_info[0].path]

    for path_info in path_pieces_info[1:]:
        path = path_info.path
        if path_info.reversed:
            path = np.flipud(path)
        path_pieces.append(path[1:])

    return ResultPathInfo(
        path=np.vstack(path_pieces),
        pieces=path_pieces_info,
    )


def extract_path_with_way_points(init_info: InitialInfo,
                                 parameters: Parameters) -> ResultPathInfo:
    speed_data = init_info.speed_data
    path_pieces_info = []

    if parameters.reuse_computed_travel_time:
        compute_ttime = [True, False]
        last_travel_time = None

        for (start_point, end_point), compute_tt in zip(
                init_info.point_intervals(), itertools.cycle(compute_ttime)):
            if compute_tt:
                travel_time = compute_travel_time(speed_data, end_point, parameters)
                last_travel_time = travel_time
                is_reversed = False
            else:
                travel_time = last_travel_time
                start_point, end_point = end_point, start_point
                is_reversed = True

            path = extract_path(travel_time, start_point, end_point, parameters)

            path_pieces_info.append(PathInfo(
                path=path,
                start_point=start_point,
                end_point=end_point,
                travel_time=travel_time,
                reversed=is_reversed
            ))
    else:
        for start_point, end_point in init_info.point_intervals():
            travel_time = compute_travel_time(speed_data, end_point, parameters)
            path = extract_path(travel_time, start_point, end_point, parameters)

            path_pieces_info.append(PathInfo(
                path=path,
                start_point=start_point,
                end_point=end_point,
                travel_time=travel_time,
                reversed=False
            ))

    return make_whole_path_from_pieces(path_pieces_info)


@mpe_impl_dispatch.register(np.ndarray)
def mpe(speed_data: np.ndarray, *,
        start_point: PointType,
        end_point: PointType,
        way_points: WayPointsType = (),
        parameters: Optional[Parameters] = None) -> ResultPathInfo:

    init_info = InitialInfo(
        speed_data=speed_data,
        start_point=start_point,
        end_point=end_point,
        way_points=way_points
    )

    return mpe(init_info, parameters=parameters)


@mpe_impl_dispatch.register(InitialInfo)
def mpe(init_info: InitialInfo, *,
        parameters: Optional[Parameters] = None) -> ResultPathInfo:

    if parameters is None:
        parameters = Parameters()

    if init_info.way_points:
        return extract_path_with_way_points(init_info, parameters)
    else:
        return extract_path_without_way_points(init_info, parameters)
