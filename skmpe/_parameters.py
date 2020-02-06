# -*- coding: utf-8 -*-

import contextlib
import enum

from pydantic import confloat, conint, validator

from ._base import MPE_MODULE, ImmutableDataObject
from ._helpers import set_module


@set_module(MPE_MODULE)
class TravelTimeOrder(enum.IntEnum):
    first = 1
    second = 2


@set_module(MPE_MODULE)
class OdeSolverMethod(str, enum.Enum):
    RK23 = 'RK23'
    RK45 = 'RK45'
    DOP853 = 'DOP853'
    Radau = 'Radau'
    BDF = 'BDF'
    LSODA = 'LSODA'


@set_module(MPE_MODULE)
class Parameters(ImmutableDataObject):
    """MPE algorithm parameters
    """

    travel_time_spacing: confloat(gt=0.0) = 1.0
    travel_time_order: TravelTimeOrder = TravelTimeOrder.first
    travel_time_cache: bool = False

    ode_solver_method: OdeSolverMethod = OdeSolverMethod.RK45

    integrate_time_bound: confloat(gt=0.0) = 10000.0
    integrate_min_step: confloat(ge=0.0) = 0.0
    integrate_max_step: confloat(gt=0.0) = 4.0

    dist_tol: confloat(ge=0.0) = 1e-3
    max_small_dist_steps: conint(strict=True, gt=1) = 100

    @validator('travel_time_order')
    def _check_travel_time_order(cls, v):
        if v == TravelTimeOrder.second:
            raise ValueError(
                'Currently the second order for computing travel time does not work properly.'
                '\nSee the following issue for details: https://github.com/scikit-fmm/scikit-fmm/issues/28'
            )
        return v


_default_parameters = Parameters()


@contextlib.contextmanager
def parameters(**kwargs):
    """Context manager for using specified parameters

    Parameters
    ----------

    kwargs : mapping
        The parameters

            - **travel_time_spacing** --
            - **travel_time_order** --
            - **travel_time_cache** --
            - **ode_solver_method** --
            - **integrate_time_bound** --
            - **integrate_min_step** --
            - **dist_tol** --
            - **max_small_dist_steps** --

    """

    global _default_parameters
    prev_default_parameters = _default_parameters

    _default_parameters = Parameters(**kwargs)

    try:
        yield _default_parameters
    finally:
        _default_parameters = prev_default_parameters


def default_parameters() -> Parameters:
    """Returns the default parameters

    Returns
    -------
    parameters : Parameters
        Default parameters

    """

    return _default_parameters
