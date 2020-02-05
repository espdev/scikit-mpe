# -*- coding: utf-8 -*-

import itertools
from typing import List, Optional

import numpy as np
import skfmm as fmm

from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import RK23, RK45, DOP853, Radau, BDF, LSODA
from scipy.spatial.distance import euclidean

from ._base import PointType, InitialInfo, PathInfo, ResultPathInfo, logger
from ._parameters import Parameters, OdeSolverMethod, default_parameters
from ._exceptions import ComputeTravelTimeError, PathExtractionError, EndPointNotReachedError


def make_interpolator(coords, values, fill_value: float = 0.0):
    return RegularGridInterpolator(
        coords, values, method='linear', bounds_error=False, fill_value=fill_value)


ODE_SOLVER_METHODS = {
    OdeSolverMethod.RK23: RK23,
    OdeSolverMethod.RK45: RK45,
    OdeSolverMethod.DOP853: DOP853,
    OdeSolverMethod.Radau: Radau,
    OdeSolverMethod.BDF: BDF,
    OdeSolverMethod.LSODA: LSODA,
}


class MinimalPathExtractor:
    """Minimal path extractor

    Minimal path extractor based on the fast marching method and ODE solver.

    """

    def __init__(self, speed_data: np.ndarray, source_point: PointType,
                 parameters: Optional[Parameters] = None) -> None:

        if parameters is None:
            parameters = default_parameters()

        travel_time, phi = self._compute_travel_time(speed_data, source_point, parameters)
        grad_interpolants, tt_interpolant, phi_interpolant = self._compute_interpolants(travel_time, phi, parameters)

        self.speed_data = speed_data
        self.travel_time = travel_time
        self.phi = phi

        self.source_point = source_point

        self.travel_time_interpolant = tt_interpolant
        self.phi_interpolant = phi_interpolant
        self.gradient_interpolants = grad_interpolants

        self.parameters = parameters

        # output after compute ODE solution
        self.integrate_times = []
        self.path_points = []
        self.path_travel_times = []
        self.steps = 0
        self.func_eval_count = 0

    @staticmethod
    def _compute_travel_time(speed_data: np.ndarray,
                             source_point: PointType,
                             parameters: Parameters):
        # define the zero contour and set the wave source
        phi = np.ones_like(speed_data)
        phi[source_point] = -1

        try:
            travel_time = fmm.travel_time(phi, speed_data,
                                          dx=parameters.travel_time_spacing,
                                          order=parameters.travel_time_order)
        except Exception as err:
            raise ComputeTravelTimeError from err

        return travel_time, phi

    @staticmethod
    def _compute_interpolants(travel_time, phi, parameters):
        grid_coords = [np.arange(n) for n in travel_time.shape]

        gradients = np.gradient(travel_time, parameters.travel_time_spacing)
        gradient_interpolants = []

        for gradient in gradients:
            interpolant = make_interpolator(grid_coords, gradient, fill_value=0.0)
            gradient_interpolants.append(interpolant)

        tt_interpolant = make_interpolator(grid_coords, travel_time, fill_value=0.0)
        phi_interpolant = make_interpolator(grid_coords, phi, fill_value=1.0)

        return gradient_interpolants, tt_interpolant, phi_interpolant

    def __call__(self, start_point: PointType) -> np.ndarray:
        gradient_interpolants = self.gradient_interpolants
        travel_time_interpolant = self.travel_time_interpolant

        def right_hand_func(time: float, point: np.ndarray) -> np.ndarray:  # noqa
            velocity = np.array([gi(point).item() for gi in gradient_interpolants])

            if np.any(np.isclose(velocity, 0.0)):
                # zero-velocity most often means masked travel time data
                return velocity

            return -velocity / np.linalg.norm(velocity)

        solver_cls = ODE_SOLVER_METHODS[self.parameters.ode_solver_method]
        logger.debug("ODE solver '%s' will be used.", solver_cls.__name__)

        solver = solver_cls(
            right_hand_func,
            t0=0.0,
            t_bound=self.parameters.integrate_time_bound,
            y0=start_point,
            max_step=self.parameters.integrate_max_step,
            first_step=None,
        )

        end_point = self.source_point

        self.integrate_times = []
        self.path_travel_times = []
        self.path_points = []
        self.steps = 0

        while True:
            self.steps += 1
            message = solver.step()

            if solver.status == 'failed':
                raise PathExtractionError(
                    f"ODE solver '{solver_cls.__name__}' has failed: {message}",
                    travel_time=self.travel_time, start_point=start_point, end_point=end_point)

            t = solver.t
            y = solver.y
            tt = travel_time_interpolant(y).item()

            self.integrate_times.append(t)
            self.path_points.append(y)
            self.path_travel_times.append(tt)
            self.func_eval_count = solver.nfev

            step_size = solver.step_size
            dist_to_end = euclidean(y, end_point)

            logger.debug('step: %d, time: %.2f, point: %s, step: %.2f, nfev: %d, dist: %.2f, message: "%s"',
                         self.steps, t, y, step_size, solver.nfev, dist_to_end, message)

            if dist_to_end < step_size:
                logger.debug(
                    'The minimal path has been extracted (time: %.2f, steps: %d, nfev: %d, dist_to_end: %.2f)',
                    t, self.steps, solver.nfev, dist_to_end)
                break

            if solver.status == 'finished':
                err_msg = (
                    f'The extracted path from the start point {start_point} '
                    f'did not reach the end point {end_point} in {t} time and {self.steps} steps '
                    f'with distance {dist_to_end:.2f} to the end point.'
                )

                raise EndPointNotReachedError(
                    err_msg,
                    travel_time=self.travel_time,
                    start_point=start_point,
                    end_point=end_point,
                    extracted_points=self.path_points,
                    last_distance=dist_to_end,
                )

        return np.array(self.path_points)


def extract_path_without_way_points(init_info: InitialInfo,
                                    parameters: Parameters) -> ResultPathInfo:
    extractor = MinimalPathExtractor(init_info.speed_data, init_info.end_point, parameters)
    path = extractor(init_info.start_point)

    path_info = PathInfo(
        path=path,
        start_point=init_info.start_point,
        end_point=init_info.end_point,
        travel_time=extractor.travel_time,
        path_travel_times=np.asarray(extractor.path_travel_times),
        reversed=False,
    )

    return ResultPathInfo(path=path, pieces=[path_info])


def make_whole_path_from_pieces(path_pieces_info: List[PathInfo]) -> ResultPathInfo:
    path_pieces = [path_pieces_info[0].path]

    for path_info in path_pieces_info[1:]:
        path = path_info.path
        if path_info.reversed:
            path = np.flipud(path)
        path_pieces.append(path)

    return ResultPathInfo(
        path=np.vstack(path_pieces),
        pieces=path_pieces_info,
    )


def extract_path_with_way_points(init_info: InitialInfo,
                                 parameters: Parameters) -> ResultPathInfo:
    speed_data = init_info.speed_data
    path_pieces_info = []

    if parameters.travel_time_cache:
        compute_ttime = [True, False]
        last_extractor = None

        for (start_point, end_point), compute_tt in zip(
                init_info.point_intervals(), itertools.cycle(compute_ttime)):
            if compute_tt:
                extractor = MinimalPathExtractor(speed_data, end_point, parameters)
                last_extractor = extractor
                is_reversed = False
            else:
                extractor = last_extractor
                start_point, end_point = end_point, start_point
                is_reversed = True

            path = extractor(start_point)

            path_pieces_info.append(PathInfo(
                path=path,
                start_point=start_point,
                end_point=end_point,
                travel_time=extractor.travel_time,
                path_travel_times=np.asarray(extractor.path_travel_times),
                reversed=is_reversed
            ))
    else:
        for start_point, end_point in init_info.point_intervals():
            extractor = MinimalPathExtractor(speed_data, end_point, parameters)
            path = extractor(start_point)

            path_pieces_info.append(PathInfo(
                path=path,
                start_point=start_point,
                end_point=end_point,
                travel_time=extractor.travel_time,
                path_travel_times=np.asarray(extractor.path_travel_times),
                reversed=False
            ))

    return make_whole_path_from_pieces(path_pieces_info)


def extract_path(init_info: InitialInfo,
                 parameters: Optional[Parameters] = None) -> ResultPathInfo:

    if parameters is None:
        parameters = default_parameters()

    if init_info.way_points:
        return extract_path_with_way_points(init_info, parameters)
    else:
        return extract_path_without_way_points(init_info, parameters)
