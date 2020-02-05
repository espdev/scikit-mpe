# -*- coding: utf-8 -*-

import contextlib
import enum

from pydantic import confloat

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
    integrate_max_step: confloat(gt=0.0) = 4.0


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
            - **gradient_spacing** --
            - **extract_max_iterations** --
            - **extract_method** --
            - **travel_time_cache** --

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