# scikit-mpe

**scikit-mpe** is a package for extracting a minimal path in n-dimensional Euclidean space (on regular Cartesian grids) 
using the fast marching method.


## Quickstart

Here is a simple example that demonstrates how to extract the path passing through the retina vessel.

```python
from matplotlib import pyplot as plt

from skimage.data import retina
from skimage.color import rgb2gray
from skimage.transform import rescale
from skimage.filters import sato

from skmpe import mpe


speed_image = sato(rescale(rgb2gray(retina()), 0.5))

start_point = (76, 388)
end_point = (611, 442)
way_points = [(330, 98), (554, 203)]

path_info = mpe(speed_image, start_point, end_point, way_points)

px, py = path_info.path[:, 1], path_info.path[:, 0]

plt.imshow(speed_image, cmap='gray')
plt.plot(px, py, '-r')

plt.plot(*start_point[::-1], 'oy')
plt.plot(*end_point[::-1], 'og')
for p in way_points:
    plt.plot(*p[::-1], 'ob')

plt.show()
```

![retina_vessel_path](https://user-images.githubusercontent.com/1299189/73837796-482a2c00-4823-11ea-9273-3e551f4bbdd7.png)
