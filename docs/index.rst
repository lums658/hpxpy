hpxpy
=====

A thin Python wrapper over the `HPX <https://hpx.stellar-group.org/>`_ C++ runtime:
a NUMA-aware :class:`~hpxpy.Array` whose operations *are* HPX parallel algorithms,
with a measured abstraction penalty of ~1.0 vs hand-written C++ HPX.

Phase 1 (current) wraps HPX with no NumPy in the data path; NumPy compatibility is a
separate, later phase. See the project plan and benchmark notes in the repository.

.. code-block:: python

   import hpxpy as hpx

   hpx.init()                 # start the HPX runtime (all cores)
   a = hpx.arange(1_000_000)  # NUMA-aware Array [0, 1, ..., n-1]
   b = hpx.full(1_000_000, 2.0)
   total = a.sum()            # parallel hpx::reduce
   d = a.dot(b)               # fused transform_reduce
   c = a * b                  # element-wise -> new Array

.. toctree::
   :maxdepth: 2
   :caption: Contents

   api
