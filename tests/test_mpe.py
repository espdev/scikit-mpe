# -*- coding: utf-8 -*-

import collections

import pytest
import numpy as np

from skmpe import mpe, parameters, EndPointNotReachedError


travel_time_order_param = pytest.mark.parametrize('travel_time_order', [
    pytest.param(1),
    pytest.param(2, marks=pytest.mark.skip('https://github.com/scikit-fmm/scikit-fmm/issues/28')),
])


@pytest.mark.parametrize('start_point, end_point, point_count', [
    ((37, 255), (172, 112), 79),
    ((37, 255), (484, 300), 189),
])
@travel_time_order_param
def test_extract_without_waypoints(retina_speed_image, travel_time_order, start_point, end_point, point_count):
    with parameters(travel_time_order=travel_time_order):
        path_info = mpe(retina_speed_image, start_point, end_point)

    assert path_info.point_count == point_count


@pytest.mark.parametrize('start_point, end_point, way_points, ttime_cache, point_count, ttime_count, reversed_count', [
    ((37, 255), (484, 300), ((172, 112), (236, 98), (420, 153)), True, 200, 2, 2),
    ((37, 255), (484, 300), ((172, 112), (236, 98), (420, 153)), False, 199, 4, 0),
])
@travel_time_order_param
def test_extract_with_waypoints(retina_speed_image, travel_time_order,
                                start_point, end_point, way_points, ttime_cache,
                                point_count, ttime_count, reversed_count):
    with parameters(travel_time_order=travel_time_order, travel_time_cache=ttime_cache):
        path_info = mpe(retina_speed_image, start_point, end_point, way_points)

    assert path_info.point_count == point_count

    ttime_counter = collections.Counter(id(piece.travel_time) for piece in path_info.pieces)
    assert len(ttime_counter) == ttime_count

    assert list(piece.reversed for piece in path_info.pieces).count(True) == reversed_count


@pytest.mark.parametrize('start_point, end_point, time_bound', [
    ((37, 255), (484, 300), 500),
])
def test_end_point_not_reached(retina_speed_image, start_point, end_point, time_bound):
    with pytest.raises(EndPointNotReachedError):
        with parameters(integrate_time_bound=time_bound):
            mpe(retina_speed_image, start_point, end_point)


@pytest.mark.parametrize('start_point, end_point, time_bound, wall', [
    ((37, 255), (484, 300), 2000.0, (slice(245, 247), slice(0, None))),
])
def test_unreachable_end_point(retina_speed_image, start_point, end_point, time_bound, wall):
    mask = np.zeros_like(retina_speed_image, dtype=np.bool_)
    mask[wall] = True
    retina_speed_image = np.ma.masked_array(retina_speed_image, mask=mask)

    with pytest.raises(EndPointNotReachedError):
        with parameters(integrate_time_bound=time_bound):
            mpe(retina_speed_image, start_point, end_point)
