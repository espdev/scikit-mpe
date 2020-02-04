# -*- coding: utf-8 -*-

from typing import Optional, Union

import numpy as np

from ._base import PointType, PointSequenceType, InitialInfo, ResultPathInfo
from ._base import mpe as api_dispatch
from ._parameters import Parameters, default_parameters
from ._mpe import extract_path_with_way_points, extract_path_without_way_points


@api_dispatch.register(np.ndarray)
def mpe(speed_data: np.ndarray, *,
        start_point: Union[PointType, np.ndarray],
        end_point: Union[PointType, np.ndarray],
        way_points: Union[PointSequenceType, np.ndarray] = (),
        parameters: Optional[Parameters] = None) -> ResultPathInfo:

    init_info = InitialInfo(
        speed_data=speed_data,
        start_point=start_point,
        end_point=end_point,
        way_points=way_points
    )

    return mpe(init_info, parameters=parameters)


@api_dispatch.register(InitialInfo)
def mpe(init_info: InitialInfo, *,
        parameters: Optional[Parameters] = None) -> ResultPathInfo:

    if parameters is None:
        parameters = default_parameters()

    if init_info.way_points:
        return extract_path_with_way_points(init_info, parameters)
    else:
        return extract_path_without_way_points(init_info, parameters)
