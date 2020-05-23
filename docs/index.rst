.. _index:

scikit-mpe
==========

**scikit-mpe** is a package for extracting a minimal path in n-dimensional Euclidean space (on regular Cartesian grids)
using `the fast marching method <https://math.berkeley.edu/~sethian/2006/Explanations/fast_marching_explain.html>`_.

The package can be used in various engineering and image processing tasks.
For example, it can be used for extracting paths through tubular structures on 2-d and 3-d images,
or shortest paths on terrain maps.

Installing
----------

Python 3.6 or above is supported. You can install the package using pip::

    pip install -U scikit-mpe


A Simple Example
----------------

Here is the simple example: how to extract 2-d minimal path using some speed data.

.. code-block:: python
    :linenos:

    from skmpe import mpe

    # Somehow speed data is calculating
    speed_data = get_speed_data()

    # Extracting minimal path from the starting point to the ending point
    path_info = mpe(speed_data, start_point=(10, 20), end_point=(120, 45))

    # Getting the path data in numpy ndarray
    path = path_info.path


Contents
--------

.. toctree::
    :maxdepth: 2

    tutorial
    api
    changelog


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
