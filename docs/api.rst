API reference
=============

Runtime
-------

.. autofunction:: hpxpy.init
.. autofunction:: hpxpy.finalize
.. autofunction:: hpxpy.num_worker_threads
.. autofunction:: hpxpy.hpx_version

Array
-----

.. autoclass:: hpxpy.Array
   :members:
   :special-members: __len__, __getitem__, __setitem__

Constructors
------------

.. autofunction:: hpxpy.zeros
.. autofunction:: hpxpy.full
.. autofunction:: hpxpy.arange

Reductions & element-wise
-------------------------

.. autofunction:: hpxpy.sum
.. autofunction:: hpxpy.min
.. autofunction:: hpxpy.max
.. autofunction:: hpxpy.dot
