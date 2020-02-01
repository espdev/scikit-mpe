# -*- coding: utf-8 -*-

import pytest
import numpy as np
from pydantic import ValidationError

from mpe import InitialInfo


@pytest.mark.parametrize('speed_data, start_point, end_point, way_points', [
    (np.zeros((10, 15)), (3, 4), (5, 6), None),
    (np.zeros((10, 15)), (3, 4), (5, 6), ()),
    (np.zeros((10, 15)), (3, 4), (5, 6), ((1, 2), (7, 8))),
])
def test_initial_info(speed_data, start_point, end_point, way_points):
    info = InitialInfo(speed_data=speed_data, start_point=start_point, end_point=end_point, way_points=way_points)

    assert info.speed_data == pytest.approx(speed_data)
    assert info.start_point == start_point
    assert info.end_point == end_point
    assert info.way_points == way_points


def test_invalid_initial_info():
    with pytest.raises(ValidationError):
        InitialInfo(speed_data=np.zeros((10,)), start_point=(1,), end_point=(2,))

    with pytest.raises(ValidationError):
        InitialInfo(speed_data=np.zeros((10, 15)), start_point=(1, 2), end_point=(3, 4, 5))

    with pytest.raises(ValidationError):
        InitialInfo(speed_data=np.zeros((10, 15)), start_point=(1, 2, 3), end_point=(3, 4))

    with pytest.raises(ValidationError):
        InitialInfo(speed_data=np.zeros((10, 15)), start_point=(1, 2, 3), end_point=(3, 4, 5))

    with pytest.raises(ValidationError):
        InitialInfo(speed_data=np.zeros((10, 15)), start_point=(1, 2), end_point=(3, 4),
                    way_points=((5, 6, 7), (7, 8, 9)))

    with pytest.raises(ValidationError):
        InitialInfo(speed_data=np.zeros((10, 15)), start_point=(1, 2), end_point=(3, 4),
                    way_points=((5, 6), (7, 8, 9)))

    with pytest.raises(ValidationError):
        InitialInfo(speed_data=np.zeros((10, 15)), start_point=(10, 2), end_point=(3, 4))

    with pytest.raises(ValidationError):
        InitialInfo(speed_data=np.zeros((10, 15)), start_point=(1, 2), end_point=(3, 40))

    mask = np.zeros((10, 15), dtype=np.bool_)
    mask[(1, 5), (2, 6)] = True
    speed_data = np.ma.masked_array(np.zeros((10, 15)), mask=mask)

    with pytest.raises(ValidationError):
        InitialInfo(speed_data=speed_data, start_point=(1, 2), end_point=(3, 4))

    with pytest.raises(ValidationError):
        InitialInfo(speed_data=speed_data, start_point=(3, 4), end_point=(5, 6))

    with pytest.raises(ValidationError):
        InitialInfo(speed_data=speed_data, start_point=(3, 4), end_point=(2, 3),
                    way_points=((1, 2), (4, 4)))

    with pytest.raises(ValidationError):
        InitialInfo(speed_data=speed_data, start_point=(3, 4), end_point=(2, 3),
                    way_points=((2, 2), (5, 6)))
