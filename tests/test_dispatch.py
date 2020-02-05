# -*- coding: utf-8 -*-

import pytest

from skmpe import mpe


@pytest.mark.parametrize('args, kwargs', [
    ((10, 20), dict(start_point=(1, 2), end_point=(5, 5))),
    ((10.0,), dict(start_point=(1, 2), end_point=(5, 5))),
])
def test_invalid_signatures(args, kwargs):
    with pytest.raises(TypeError, match='invalid signature'):
        mpe(*args, **kwargs)
