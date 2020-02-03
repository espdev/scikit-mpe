# -*- coding: utf-8 -*-

import pytest
import numpy as np
from pydantic import ValidationError

from mpe import InitialInfo


@pytest.mark.parametrize('shape, start_point, end_point, way_points', [
    ((10, 15), (3, 4), (5, 6), ()),
    ((10, 15, 12), (3, 4, 5), (5, 6, 7), ()),
    ((10, 15), (3, 4), (5, 6), ((1, 2), (7, 8))),
    ((10, 15, 12), (3, 4, 5), (5, 6, 7), ((1, 2, 3), (7, 8, 9))),
])
def test_basic(shape, start_point, end_point, way_points):
    speed_data = np.zeros(shape)

    info = InitialInfo(
        speed_data=speed_data,
        start_point=start_point,
        end_point=end_point,
        way_points=way_points
    )

    assert info.speed_data == pytest.approx(speed_data)
    assert info.start_point == start_point
    assert info.end_point == end_point
    assert info.way_points == way_points


def test_invalid_ndim():
    with pytest.raises(ValidationError, match='minimum dimension must be'):
        InitialInfo(
            speed_data=np.zeros((10,)),
            start_point=(1,),
            end_point=(2,)
        )


@pytest.mark.parametrize('shape, start_point, end_point, way_points', [
    ((10, 15), (1, 2), (3, 4, 5), ()),
    ((10, 15), (1, 2, 3), (3, 4, 5), ()),
    ((10, 15), (1, 2), (3, 4), ((5, 6), (7, 8, 9))),
    ((10, 15), (1, 2), (3, 4), ((5, 6, 7), (7, 8, 9))),
    ((10, 15, 12), (1, 2, 3), (3, 4), ()),
    ((10, 15, 12), (1, 2), (3, 4), ()),
    ((10, 15, 12), (1, 2, 3), (3, 4, 5), ((1, 2), (3, 4, 5))),
    ((10, 15, 12), (1, 2, 3), (3, 4, 5), ((1, 2), (3, 4))),
])
def test_mismatch_ndim(shape, start_point, end_point, way_points):
    with pytest.raises(ValidationError, match='must have dimension'):
        InitialInfo(
            speed_data=np.zeros(shape),
            start_point=start_point,
            end_point=end_point,
            way_points=way_points
        )


@pytest.mark.parametrize('shape, start_point, end_point, way_points', [
    ((10, 15), (10, 2), (3, 4), ()),
    ((10, 15), (1, 2), (3, 45), ()),
    ((10, 15), (1, 2), (-3, 4), ()),
    ((10, 15), (1, 2), (3, 4), ((10, 6), (7, 8))),
    ((10, 15), (1, 2), (3, 4), ((5, 6), (7, 15))),
    ((10, 15, 12), (1, 2, 30), (3, 4, 5), ()),
    ((10, 15, 12), (1, 2, 10), (3, -4, 50), ()),
    ((10, 15, 12), (1, 2, 3), (3, 4, 5), ((1, 2, 12), (3, 4, 5))),
    ((10, 15, 12), (1, 2, 3), (3, 4, 5), ((1, 2, 3), (3, 4, -1))),
])
def test_out_of_bounds(shape, start_point, end_point, way_points):
    with pytest.raises(ValidationError, match="is out of 'speed_data' bounds"):
        InitialInfo(
            speed_data=np.zeros(shape),
            start_point=start_point,
            end_point=end_point,
            way_points=way_points
        )


@pytest.mark.parametrize('shape, start_point, end_point, way_points', [
    ((10, 15), (1, 2), (1, 2), ()),
    ((10, 15), (1, 2), (3, 4), ((3, 4), (5, 6))),
    ((10, 15), (1, 2), (3, 4), ((5, 6), (5, 6))),
    ((10, 15, 12), (1, 2, 3), (1, 2, 3), ()),
    ((10, 15, 12), (1, 2, 3), (3, 4, 5), ((1, 2, 3), (3, 4, 7))),
    ((10, 15, 12), (1, 2, 3), (3, 4, 5), ((1, 5, 2), (1, 5, 2))),
])
def test_duplicates(shape, start_point, end_point, way_points):
    with pytest.raises(ValidationError, match='the points must not be duplicated'):
        InitialInfo(
            speed_data=np.zeros(shape),
            start_point=start_point,
            end_point=end_point,
            way_points=way_points
        )


@pytest.mark.parametrize('shape, mask, start_point, end_point, way_points', [
    ((10, 15), ((1, 5), (2, 6)), (1, 2), (3, 4), ()),
    ((10, 15), ((1, 5), (2, 6)), (3, 4), (5, 6), ()),
    ((10, 15), ((1, 5), (2, 6)), (3, 4), (2, 3), ((1, 2), (4, 4))),
    ((10, 15), ((1, 5), (2, 6)), (3, 4), (2, 3), ((2, 2), (5, 6))),
    ((10, 15, 12), ((1, 5), (2, 6), (7, 8)), (1, 2, 7), (2, 3, 4), ()),
    ((10, 15, 12), ((1, 5), (2, 6), (7, 8)), (3, 4, 7), (5, 6, 8), ()),
])
def test_inside_masked_area(shape, mask, start_point, end_point, way_points):
    mask_data = np.zeros(shape, dtype=np.bool_)
    mask_data[mask] = True
    speed_data = np.ma.masked_array(np.zeros(shape), mask=mask_data)

    with pytest.raises(ValidationError, match="inside 'speed_data' masked area"):
        InitialInfo(
            speed_data=speed_data,
            start_point=start_point,
            end_point=end_point,
            way_points=way_points,
        )
