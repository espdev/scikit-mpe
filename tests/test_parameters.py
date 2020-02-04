# -*- coding: utf-8 -*-

import pytest
from pydantic import ValidationError

from mpe import Parameters, parameters, default_parameters


def test_forbid_extra():
    with pytest.raises(ValidationError):
        Parameters(foo=1)


def tets_immutable():
    p = Parameters()
    with pytest.raises(TypeError):
        p.fmm_grid_spacing = 2.0


def test_context():
    with parameters(fmm_grid_spacing=2.0):
        assert default_parameters().fmm_grid_spacing == pytest.approx(2.0)
    assert default_parameters().fmm_grid_spacing == pytest.approx(1.0)
