.. _tutorial:

********
Tutorial
********

Overview
========

**scikit-mpe** package allows you to extract N-dimensional minimal paths
using existing speed data and starting/ending and optionally way points initial data.

.. note::

    The package does not compute any speed data (a.k.a speed function). It is expected that the
    speed data was previously obtained/computed in some way.

The package can be useful for various engineering and image processing tasks.
For example, the package can be used for extracting paths through tubular structures
on 2-d and 3-d images, or shortest paths on a terrain map.

The package uses `the fast marching method <https://scikit-fmm.readthedocs.io/en/latest/>`_ and
`ODE solver <https://docs.scipy.org/doc/scipy/reference/integrate.html#solving-initial-value-problems-for-ode-systems>`_ for extracting minimal paths.

Algorithm
=========

The algorithm contains two main steps:

    - First, the travel time is computing from the given ending point (zero contour) to every speed data point
      using the `fast marching method <https://en.wikipedia.org/wiki/Fast_marching_method>`_.
    - Second, the minimal path (travel time is minimizing) is extracting from the starting point to the ending point
      using ODE solver (`Runge-Kutta <https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods>`_ for example)
      for solving the differential equation :math:`x_t = - \nabla (t) / | \nabla (t)`

If we have way points we need to perform these two steps for every interval between the starting point, the set of the
way points and the ending point and concatenate the path pieces to the full path.

Quickstart
==========

Let's look at a simple example of how the algorithm works.

.. note::

    We will use `retina test image <https://scikit-image.org/docs/dev/api/skimage.data.html#skimage.data.retina>`_ from
    `scikit-image <https://scikit-image.org/>`_ package as the test data for all examples.

First, we need a speed data (speed function). We can use one of the tubeness filters for computing speed data for
our test data, `sato filter <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.sato>`_ for example.

.. plot::
    :context:

    from skimage.data import retina
    from skimage.color import rgb2gray
    from skimage.transform import rescale
    from skimage.filters import sato

    image_data = rescale(rgb2gray(retina()), 0.5)
    speed_data = sato(image_data) + 0.05
    speed_data[speed_data > 1.0] = 1.0

    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(image_data, cmap='gray')
    ax1.set_title('source data')
    ax1.axis('off')
    ax2.imshow(speed_data, cmap='gray')
    ax2.set_title('speed data')
    ax2.axis('off')

The speed data values must be in range [0.0, 1.0] and can be `masked <https://numpy.org/devdocs/reference/maskedarray.generic.html>`_ also.

where:

    - 0.0 -- zero speed (impassable)
    - 1.0 -- max speed
    - masked -- impassable

Second, let's try to extract the minimal path for some starting and ending points
using **scikit-mpe** package and plot it. Also we can plot travel time contours.

.. plot::
    :context: close-figs

    from skmpe import mpe

    # define starting and ending points
    start_point = (165, 280)
    end_point = (611, 442)

    path_info = mpe(speed_data, start_point, end_point)

    # get computed travel time for given ending point and extracted path
    travel_time = path_info.pieces[0].travel_time
    path = path_info.path

    nrows, ncols = speed_data.shape
    xx, yy = np.meshgrid(np.arange(ncols), np.arange(nrows))

    fig, ax = plt.subplots(1, 1)
    ax.imshow(speed_data, cmap='gray', alpha=0.9)
    ax.plot(path[:,1], path[:,0], '-', color=[0, 1, 0], linewidth=2)
    ax.plot(start_point[1], start_point[0], 'or')
    ax.plot(end_point[1], end_point[0], 'o', color=[1, 1, 0])
    tt_c = ax.contour(xx, yy, travel_time, 20, cmap='plasma', linewidths=1.5)
    ax.clabel(tt_c, inline=1, fontsize=9, fmt='%d')
    ax.set_title('travel time contours and minimal path')
    ax.axis('off')
    cb = fig.colorbar(tt_c)
    cb.ax.set_ylabel('travel time')

Advanced Usage
==============
