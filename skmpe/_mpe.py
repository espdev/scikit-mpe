# -*- coding: utf-8 -*-

import itertools
import warnings
from typing import List, Optional, NamedTuple

import numpy as np
import skfmm as fmm

from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import RK23, RK45, DOP853, Radau, BDF, LSODA
from scipy.spatial.distance import euclidean

from ._base import mpe_module, PointType, InitialInfo, PathInfo, PathInfoResult, logger
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


@mpe_module
class PathExtractionResult(NamedTuple):
    """The named tuple with info about extracted path

    Notes
    -----

    The instance of the class is returned from :func:`MinimalPathExtractor.__call__`.

    .. py:attribute:: path_points

        The extracted path points in the list

    .. py:attribute:: path_integrate_times

        The list of integrate times for every path point

    .. py:attribute:: path_travel_times

        The list of travel time values for every path point

    .. py:attribute:: step_count

        The number of integration steps

    .. py:attribute:: func_eval_count

        The number of evaluations of the right hand function

    """

    path_points: List[PointType]
    path_integrate_times: List[float]
    path_travel_times: List[float]
    step_count: int
    func_eval_count: int


@mpe_module
class MinimalPathExtractor:
    """Minimal path extractor

    Minimal path extractor based on the fast marching method and ODE solver.

    Parameters
    ----------

    speed_data : np.ndarray
        The speed data (n-d numpy array)

    end_point : Sequence[int]
        The ending point (a.k.a. "source point")

    parameters : class:`Parameters`
        The parameters

    Examples
    --------

    .. code-block:: python

        from skmpe import MinimalPathExtractor

        # some function for computing speed data
        speed_data_2d = compute_speed_data_2d()

        mpe = MinimalPathExtractor(speed_data_2d, end_point=(10, 25))
        path = mpe((123, 34))

    Raises
    ------
    ComputeTravelTimeError : Computing travel time has failed

    """

    def __init__(self, speed_data: np.ndarray, end_point: PointType,
                 parameters: Optional[Parameters] = None) -> None:

        if parameters is None:  # pragma: no cover
            parameters = default_parameters()

        travel_time, phi = self._compute_travel_time(speed_data, end_point, parameters)
        gradients = np.gradient(travel_time, parameters.travel_time_spacing)

        grad_interpolants, tt_interpolant, phi_interpolant = self._compute_interpolants(gradients, travel_time, phi)

        self._travel_time = travel_time
        self._phi = phi

        self._end_point = end_point

        self._travel_time_interpolant = tt_interpolant
        self._phi_interpolant = phi_interpolant
        self._gradient_interpolants = grad_interpolants

        self._parameters = parameters

        # the output when computing ODE solution is finished
        self._path_points = []
        self._path_integrate_times = []
        self._path_travel_times = []
        self._step_count = 0
        self._func_eval_count = 0

    @property
    def travel_time(self) -> np.ndarray:
        """Returns the computed travel time for given speed data
        """
        return self._travel_time

    @property
    def phi(self) -> np.ndarray:
        """Returns the computed phi (zero contour) for given source point
        """
        return self._phi

    @property
    def parameters(self) -> Parameters:
        """Returns the parameters
        """
        return self._parameters

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
        except Exception as err:  # pragma: no cover
            raise ComputeTravelTimeError from err

        return travel_time, phi

    @staticmethod
    def _compute_interpolants(gradients, travel_time, phi):
        grid_coords = [np.arange(n) for n in travel_time.shape]

        gradient_interpolants = []
        for gradient in gradients:
            interpolant = make_interpolator(grid_coords, gradient, fill_value=0.0)
            gradient_interpolants.append(interpolant)

        tt_interpolant = make_interpolator(grid_coords, travel_time, fill_value=0.0)
        phi_interpolant = make_interpolator(grid_coords, phi, fill_value=1.0)

        return gradient_interpolants, tt_interpolant, phi_interpolant

    def __call__(self, start_point: PointType) -> PathExtractionResult:
        """Extract path from start point to source point (ending point)

        Parameters
        ----------
        start_point : Sequence[int]
            The starting point

        Returns
        -------
        path_extraction_result : :class:`PathExtractionResult`
            The path extraction result

        Raises
        ------
        PathExtractionError : Extracting path has failed
        EndPointNotReachedError : The extracted path is not reached the ending point

        """

        gradient_interpolants = self._gradient_interpolants
        travel_time_interpolant = self._travel_time_interpolant

        def right_hand_func(time: float, point: np.ndarray) -> np.ndarray:  # noqa
            velocity = np.array([gi(point).item() for gi in gradient_interpolants])

            if np.any(np.isclose(velocity, 0.0)):
                # zero-velocity most often means masked travel time data
                return velocity

            return -velocity / np.linalg.norm(velocity)

        solver_cls = ODE_SOLVER_METHODS[self.parameters.ode_solver_method]
        logger.debug("ODE solver '%s' will be used.", solver_cls.__name__)

        with warnings.catch_warnings():
            # filter warn "extraneous arguments"
            warnings.simplefilter('ignore', category=UserWarning)

            solver = solver_cls(
                right_hand_func,
                t0=0.0,
                t_bound=self.parameters.integrate_time_bound,
                y0=start_point,
                min_step=self.parameters.integrate_min_step,
                max_step=self.parameters.integrate_max_step,
                first_step=None,
            )

        self._path_points = []
        self._path_integrate_times = []
        self._path_travel_times = []
        self._step_count = 0

        end_point = self._end_point
        dist_tol = self.parameters.dist_tol

        y = None
        y_old = start_point
        small_dist_steps_left = self.parameters.max_small_dist_steps

        while True:
            self._step_count += 1
            message = solver.step()

            if solver.status == 'failed':  # pragma: no cover
                raise PathExtractionError(
                    f"ODE solver '{solver_cls.__name__}' has failed: {message}",
                    travel_time=self.travel_time, start_point=start_point, end_point=end_point)

            if y is not None:
                y_old = y

            y = solver.y
            t = solver.t
            tt = travel_time_interpolant(y).item()

            add_point = True
            y_dist = euclidean(y, y_old)

            if y_dist < dist_tol:
                logger.warning('step: %d, the distance between old and current extracted point (%f) is '
                               'too small (less than dist_tol=%f)', self._step_count, y_dist, dist_tol)
                add_point = False
                small_dist_steps_left -= 1

            if add_point:
                small_dist_steps_left = self.parameters.max_small_dist_steps

                self._path_points.append(y)
                self._path_integrate_times.append(t)
                self._path_travel_times.append(tt)

            self._func_eval_count = solver.nfev

            step_size = solver.step_size
            dist_to_end = euclidean(y, end_point)

            logger.debug('step: %d, time: %.2f, point: %s, step_size: %.2f, nfev: %d, dist: %.2f, message: "%s"',
                         self._step_count, t, y, step_size, solver.nfev, dist_to_end, message)

            if dist_to_end < step_size:
                logger.debug(
                    'The minimal path has been extracted (time: %.2f, _step_count: %d, nfev: %d, dist_to_end: %.2f)',
                    t, self._step_count, solver.nfev, dist_to_end)
                break

            if solver.status == 'finished' or small_dist_steps_left == 0:
                if small_dist_steps_left == 0:
                    reason = f'the distance between old and current point stay too small ' \
                             f'for {self.parameters.max_small_dist_steps} _step_count'
                else:
                    reason = f'time bound {self.parameters.integrate_time_bound} is reached, solver was finished.'

                err_msg = (
                    f'The extracted path from the start point {start_point} '
                    f'did not reach the end point {end_point} in {t} time and {self._step_count} _step_count '
                    f'with distance {dist_to_end:.2f} to the end point. Reason: {reason}'
                )

                raise EndPointNotReachedError(
                    err_msg,
                    travel_time=self.travel_time,
                    start_point=start_point,
                    end_point=end_point,
                    extracted_points=self._path_points,
                    last_distance=dist_to_end,
                    reason=reason,
                )

        return PathExtractionResult(
            path_points=self._path_points,
            path_integrate_times=self._path_integrate_times,
            path_travel_times=self._path_travel_times,
            step_count=self._step_count,
            func_eval_count=self._func_eval_count,
        )


def extract_path_without_way_points(init_info: InitialInfo,
                                    parameters: Parameters) -> PathInfoResult:
    extractor = MinimalPathExtractor(init_info.speed_data, init_info.end_point, parameters)
    result = extractor(init_info.start_point)

    path_info = PathInfo(
        path=np.asarray(result.path_points),
        start_point=init_info.start_point,
        end_point=init_info.end_point,
        travel_time=extractor.travel_time,
        extraction_result=result,
        reversed=False,
    )

    return PathInfoResult(path=path_info.path, pieces=(path_info,))


def make_whole_path_from_pieces(path_pieces_info: List[PathInfo]) -> PathInfoResult:
    path_pieces = [path_pieces_info[0].path]

    for path_info in path_pieces_info[1:]:
        path = path_info.path
        if path_info.reversed:
            path = np.flipud(path)
        path_pieces.append(path)

    return PathInfoResult(
        path=np.vstack(path_pieces),
        pieces=tuple(path_pieces_info),
    )


def extract_path_with_way_points(init_info: InitialInfo,
                                 parameters: Parameters) -> PathInfoResult:
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

            result = extractor(start_point)

            path_pieces_info.append(PathInfo(
                path=np.asarray(result.path_points),
                start_point=start_point,
                end_point=end_point,
                travel_time=extractor.travel_time,
                extraction_result=result,
                reversed=is_reversed
            ))
    else:
        for start_point, end_point in init_info.point_intervals():
            extractor = MinimalPathExtractor(speed_data, end_point, parameters)
            result = extractor(start_point)

            path_piece_info = PathInfo(
                path=np.asarray(result.path_points),
                start_point=start_point,
                end_point=end_point,
                travel_time=extractor.travel_time,
                extraction_result=result,
                reversed=False
            )

            path_pieces_info.append(path_piece_info)

    return make_whole_path_from_pieces(path_pieces_info)


def extract_path(init_info: InitialInfo,
                 parameters: Optional[Parameters] = None) -> PathInfoResult:

    if parameters is None:
        parameters = default_parameters()

    if init_info.way_points:
        return extract_path_with_way_points(init_info, parameters)
    else:
        return extract_path_without_way_points(init_info, parameters)
