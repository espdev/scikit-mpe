# -*- coding: utf-8 -*-

import collections

import pytest
import numpy as np

from skmpe import mpe, parameters, OdeSolverMethod, EndPointNotReachedError


TRAVEL_TIME_ABS_TOL = 100

travel_time_order_param = pytest.mark.parametrize('travel_time_order', [
    pytest.param(1),
    pytest.param(2, marks=pytest.mark.skip('https://github.com/scikit-fmm/scikit-fmm/issues/28')),
])


@pytest.mark.parametrize('ode_method, start_point, end_point', [
    (OdeSolverMethod.RK23, (37, 255), (172, 112)),
    (OdeSolverMethod.RK45, (37, 255), (172, 112)),
    (OdeSolverMethod.DOP853, (37, 255), (172, 112)),
    (OdeSolverMethod.Radau, (37, 255), (172, 112)),
    (OdeSolverMethod.BDF, (37, 255), (172, 112)),
    (OdeSolverMethod.LSODA, (37, 255), (172, 112)),

    (OdeSolverMethod.RK45, (37, 255), (484, 300)),
])
@travel_time_order_param
def test_extract_path_without_waypoints(retina_speed_image, travel_time_order, ode_method, start_point, end_point):
    with parameters(ode_solver_method=ode_method, travel_time_order=travel_time_order):
        path_info = mpe(retina_speed_image, start_point, end_point)

    assert path_info.point_count > 0

    path_piece_info = path_info.pieces[0]
    start_travel_time = path_piece_info.travel_time[start_point[0], start_point[1]]
    end_travel_time = path_piece_info.travel_time[end_point[0], end_point[1]]
    path_start_travel_time = path_piece_info.extraction_result.path_travel_times[0]
    path_end_travel_time = path_piece_info.extraction_result.path_travel_times[-1]

    assert path_start_travel_time == pytest.approx(start_travel_time, abs=TRAVEL_TIME_ABS_TOL)
    assert path_end_travel_time == pytest.approx(end_travel_time, abs=TRAVEL_TIME_ABS_TOL)


@pytest.mark.parametrize('start_point, end_point, way_points, ttime_cache, ttime_count, reversed_count', [
    ((37, 255), (484, 300), ((172, 112), (236, 98), (420, 153)), True, 2, 2),
    ((37, 255), (484, 300), ((172, 112), (236, 98), (420, 153)), False, 4, 0),
])
@travel_time_order_param
def test_extract_path_with_waypoints(retina_speed_image, travel_time_order,
                                     start_point, end_point, way_points, ttime_cache,
                                     ttime_count, reversed_count):
    with parameters(travel_time_order=travel_time_order, travel_time_cache=ttime_cache):
        path_info = mpe(retina_speed_image, start_point, end_point, way_points)

    assert path_info.point_count > 0

    for path_piece_info in path_info.pieces:
        start_pt = path_piece_info.start_point
        end_pt = path_piece_info.end_point

        start_travel_time = path_piece_info.travel_time[start_pt[0], start_pt[1]]
        end_travel_time = path_piece_info.travel_time[end_pt[0], end_pt[1]]
        path_start_travel_time = path_piece_info.extraction_result.path_travel_times[0]
        path_end_travel_time = path_piece_info.extraction_result.path_travel_times[-1]

        assert path_start_travel_time == pytest.approx(start_travel_time, abs=TRAVEL_TIME_ABS_TOL)
        assert path_end_travel_time == pytest.approx(end_travel_time, abs=TRAVEL_TIME_ABS_TOL)

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
