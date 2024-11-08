..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

=====
Cache
=====

.. code-block:: python

    from numpy import array

    from gemseo import create_discipline
    from gemseo.caches.hdf5_cache import HDF5Cache

Create a discipline

.. code-block:: python

    discipline = create_discipline("AnalyticDiscipline", expressions={"z": "x+y"})

Set the cache policy to store all executions:

.. code-block:: python

    discipline.set_cache("HDF5Cache", hdf_file_path="file.h5")  # on the disk
    discipline.set_cache("MemoryFullCache")  # in memory

Set the simple cache policy to store the last execution in memory:

.. code-block:: python

    discipline.set_cache("SimpleCache")  # default option

Export cache to dataset:

.. code-block:: python

    input_data = {"x": array([1.0]), "y": array([2.0])}
    discipline.execute(input_data)
    dataset = discipline.cache.to_dataset()

Cache inputs and outputs in an HDF5 file:

.. code-block:: python

    input_data = {"x": array([1.0]), "y": array([2.0])}
    output_data = {"z": array([3.0])}
    cache = HDF5Cache(hdf_file_path="file.h5", hdf_node_path="node")
    cache.cache_outputs(input_data, output_data)

Get cached data:

.. code-block:: python

    last_entry = cache.last_entry
    last_cached_input_data = last_entry.inputs
    last_cached_output_data = last_entry.outputs
    n_entries = len(cache)

Get outputs and jacobian if data are cached, else ``None``:

.. code-block:: python

    _, output_data, jac_data = cache[input_data]
