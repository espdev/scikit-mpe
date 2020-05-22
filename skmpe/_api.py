# -*- coding: utf-8 -*-

from typing import Optional, Union, overload

import numpy as np

from ._base import mpe_module, singledispatch, PointType, PointSequenceType, InitialInfo, PathInfoResult
from ._parameters import Parameters
from ._mpe import extract_path


@overload
def mpe(speed_data: np.ndarray,
        start_point: Union[PointType, np.ndarray],
        end_point: Union[PointType, np.ndarray],
        way_points: Union[PointSequenceType, np.ndarray] = (),
        *,
        parameters: Optional['Parameters'] = None) -> PathInfoResult:
    pass  # pragma: no cover


@overload
def mpe(init_info: InitialInfo,
        *,
        parameters: Optional['Parameters'] = None) -> PathInfoResult:
    pass  # pragma: no cover


@mpe_module
@singledispatch
def mpe(*args, **kwargs) -> PathInfoResult:  # noqa
    """Extracts a minimal path by start/end and optionally way points

    The function is high level API for extracting paths.

    Parameters
    ----------
    init_info : :class:`InitialInfo`
        (sign 1) The initial info
    start_point : Sequence[int]
        (sign 2) The starting point
    end_point : Sequence[int]
        (sign 2) The ending point
    way_points : Sequence[Sequence[int]]
        (sign 2) The way points
    parameters : :class:`Parameters`
        The parameters

    Notes
    -----

    There are two signatures of `mpe` function.

    Use :class:`InitialInfo` for init data:

    .. code-block:: python

        mpe(init_info: InitialInfo, *,
            parameters: Optional[Parameters] = None) -> ResultPathInfo

    Set init data directly:

    .. code-block:: python

        mpe(speed_data: np.ndarray, *,
            start_point: Sequence[int],
            end_point: Sequence[int],
            way_points: Sequence[Sequence[int]] = (),
            parameters: Optional[Parameters] = None) -> ResultPathInfo

    Returns
    -------
    path_info : :class:`PathInfoResult`
        Extracted path info

    See Also
    --------
    InitialInfo, Parameters, MinimalPathExtractor

    """


@mpe.register(np.ndarray)  # noqa
def mpe_1(speed_data: np.ndarray,
          start_point: Union[PointType, np.ndarray],
          end_point: Union[PointType, np.ndarray],
          way_points: Union[PointSequenceType, np.ndarray] = (),
          *,
          parameters: Optional[Parameters] = None) -> PathInfoResult:

    init_info = InitialInfo(
        speed_data=speed_data,
        start_point=start_point,
        end_point=end_point,
        way_points=way_points
    )

    return mpe(init_info, parameters=parameters)


@mpe.register(InitialInfo)  # noqa
def mpe_2(init_info: InitialInfo,
          *,
          parameters: Optional[Parameters] = None) -> PathInfoResult:
    return extract_path(init_info, parameters)
