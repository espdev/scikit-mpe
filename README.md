# scikit-mpe

[![PyPI version](https://img.shields.io/pypi/v/scikit-mpe.svg)](https://pypi.python.org/pypi/scikit-mpe)
[![Build status](https://travis-ci.org/espdev/scikit-mpe.svg?branch=master)](https://travis-ci.org/espdev/scikit-mpe)
[![Documentation Status](https://readthedocs.org/projects/scikit-mpe/badge/?version=latest)](https://scikit-mpe.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/espdev/scikit-mpe/badge.svg?branch=master)](https://coveralls.io/github/espdev/scikit-mpe?branch=master)
![Supported Python versions](https://img.shields.io/pypi/pyversions/scikit-mpe.svg)
[![License](https://img.shields.io/pypi/l/scikit-mpe.svg)](LICENSE)

**scikit-mpe** is a package for extracting a minimal path in N-dimensional Euclidean space (on regular Cartesian grids) 
using [the fast marching method](https://math.berkeley.edu/~sethian/2006/Explanations/fast_marching_explain.html).


## Quickstart

### Installing

```
pip install -U scikit-mpe
```

### Usage

Here is a simple example that demonstrates how to extract the path passing through the retina vessels.

```python
from matplotlib import pyplot as plt

from skimage.data import retina
from skimage.color import rgb2gray
from skimage.transform import rescale
from skimage.filters import sato

from skmpe import mpe

image = rescale(rgb2gray(retina()), 0.5)
speed_image = sato(image)

start_point = (76, 388)
end_point = (611, 442)
way_points = [(330, 98), (554, 203)]

path_info = mpe(speed_image, start_point, end_point, way_points)

px, py = path_info.path[:, 1], path_info.path[:, 0]

plt.imshow(image, cmap='gray')
plt.plot(px, py, '-r')

plt.plot(*start_point[::-1], 'oy')
plt.plot(*end_point[::-1], 'og')
for p in way_points:
    plt.plot(*p[::-1], 'ob')

plt.show()
```

![retina_vessel_path](https://user-images.githubusercontent.com/1299189/73838143-0d74c380-4824-11ea-946a-667c8236ed75.png)

## Documentation

The full documentation can be found at [scikit-mpe.readthedocs.io](https://scikit-mpe.readthedocs.io/en/latest)

(The documentation is being written)

## References

- [Fast Marching Methods: A boundary value formulation](https://math.berkeley.edu/~sethian/2006/Explanations/fast_marching_explain.html)
- [Level Set Methods and Fast Marching Methods](https://math.berkeley.edu/~sethian/2006/History/Menu_Expanded_History.html)
- [scikit-fmm](https://github.com/scikit-fmm/scikit-fmm) - Python implementation of the fast marching method
- [ITKMinimalPathExtraction](https://github.com/InsightSoftwareConsortium/ITKMinimalPathExtraction) - ITK based C++ implementation of MPE

## License

[MIT](https://choosealicense.com/licenses/mit/)
