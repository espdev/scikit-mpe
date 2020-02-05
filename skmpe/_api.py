# -*- coding: utf-8 -*-

from typing import Optional, Union

import numpy as np

from ._base import mpe as api_dispatch
from ._base import PointType, PointSequenceType, InitialInfo, ResultPathInfo
from ._parameters import Parameters
from ._mpe import extract_path


@api_dispatch.register(np.ndarray)
def mpe(speed_data: np.ndarray,
        start_point: Union[PointType, np.ndarray],
        end_point: Union[PointType, np.ndarray],
        way_points: Union[PointSequenceType, np.ndarray] = (),
        *,
        parameters: Optional[Parameters] = None) -> ResultPathInfo:

    init_info = InitialInfo(
        speed_data=speed_data,
        start_point=start_point,
        end_point=end_point,
        way_points=way_points
    )

    return mpe(init_info, parameters=parameters)


@api_dispatch.register(InitialInfo)
def mpe(init_info: InitialInfo,
        *,
        parameters: Optional[Parameters] = None) -> ResultPathInfo:

    return extract_path(init_info, parameters)
