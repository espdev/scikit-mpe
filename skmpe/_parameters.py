# -*- coding: utf-8 -*-

import contextlib
import enum

from pydantic import confloat, conint, validator

from ._base import mpe_module, ImmutableDataObject


@mpe_module
class TravelTimeOrder(enum.IntEnum):
    """The enumeration of travel time computation orders

    Orders:

        - **first** -- the first ordered travel time computation
        - **second** -- the second ordered travel time computation

    """

    first = 1
    second = 2


@mpe_module
class OdeSolverMethod(str, enum.Enum):
    """The enumeration of ODE solver methods
    """

    RK23 = 'RK23'
    RK45 = 'RK45'
    DOP853 = 'DOP853'
    Radau = 'Radau'
    BDF = 'BDF'
    LSODA = 'LSODA'


@mpe_module
class Parameters(ImmutableDataObject):
    """MPE algorithm parameters model

    .. py:attribute:: travel_time_spacing

        The travel time computation spacing

        | default: 1.0

    .. py:attribute:: travel_time_order

        The travel time computation order

        | default: ``TravelTimeOrder.first``

    .. py:attribute:: travel_time_cache

        Use or not travel time computation cache for extracting paths with way points

        | default: True

    .. py:attribute:: ode_solver_method

        ODE solver method

        | default: 'RK45'

    .. py:attribute:: integrate_time_bound

        Integration time bound

        | default: 10000

    .. py:attribute:: integrate_min_step

        Integration minimum step

        | default: 0.0

    .. py:attribute:: integrate_max_step

        Integration maximum step

        | default: 4.0

    .. py:attribute:: dist_tol

        Distance tolerance for control path evolution

        | default: 1e-03

    .. py:attribute:: max_small_dist_steps

        The max number of small distance steps while path evolution

        | default: 100

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


@mpe_module
@contextlib.contextmanager
def parameters(**kwargs):
    """Context manager for using specified parameters

    Parameters
    ----------

    kwargs : mapping
        The parameters

    Examples
    --------

    .. code-block:: python

        >>> from skmpe import parameters
        >>> with parameters(integrate_time_bound=200000) as params:
        >>>     print(params.__repr__())

        Parameters(
            travel_time_spacing=1.0,
            travel_time_order=<TravelTimeOrder.first: 1>,
            travel_time_cache=False,
            ode_solver_method=<OdeSolverMethod.RK45: 'RK45'>,
            integrate_time_bound=200000.0,
            integrate_min_step=0.0,
            integrate_max_step=4.0,
            dist_tol=0.001,
            max_small_dist_steps=100
        )

    .. code-block:: python

        from skmpe import parameters, mpe

        ...

        with parameters(integrate_time_bound=200000):
            path_result = mpe(start_point, end_point)

    """

    global _default_parameters
    prev_default_parameters = _default_parameters

    _default_parameters = Parameters(**kwargs)

    try:
        yield _default_parameters
    finally:
        _default_parameters = prev_default_parameters


@mpe_module
def default_parameters() -> Parameters:
    """Returns the default parameters

    Returns
    -------
    parameters : Parameters
        Default parameters

    Examples
    --------

    .. code-block:: python

        >>> from skmpe import default_parameters
        >>> print(default_parameters().__repr__())

        Parameters(
            travel_time_spacing=1.0,
            travel_time_order=<TravelTimeOrder.first: 1>,
            travel_time_cache=False,
            ode_solver_method=<OdeSolverMethod.RK45: 'RK45'>,
            integrate_time_bound=10000.0,
            integrate_min_step=0.0,
            integrate_max_step=4.0,
            dist_tol=0.001,
            max_small_dist_steps=100
        )



    """

    return _default_parameters
