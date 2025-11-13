<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Caching and recording discipline data

## Introduction

There are several reasons to store the evaluations (input, output and Jacobian values) of a discipline:

- avoid evaluating a discipline at an input value for which it has already been evaluated,
- save data for post-processing purposes, e.g. visualization, statistics, machine learning, debugging, etc,
- save the current state in memory to restart a crashed sequential disciplinary process from the iteration preceding the unfortunate event,
- \...

Some of these reasons are all the more important as the discipline triggers a simulation which can be costly. Caching disciplinary data helps to avoid wasting computing resources.

Examples can be found [here][cache-examples].

## The basics

In GEMSEO, a [Discipline][gemseo.core.discipline.discipline.Discipline] is composed of a [cache][gemseo.core.discipline.discipline.Discipline.cache] to store these evaluations expressed in terms of input, output and Jacobian data.

### The caching mechanism

When the user passes an input value to the method [execute()][gemseo.core.discipline.discipline.Discipline.execute], the [Discipline][gemseo.core.discipline.discipline.Discipline] looks in its [cache][gemseo.core.discipline.discipline.Discipline.cache] if there is an output value associated with this input value. If so, it returns it to the user. Otherwise, it computes it, stores it in the cache and returns it to the user.

### Define a tolerance for caching

The user can pass a tolerance below which two input arrays are considered equal: `numpy.linalg.norm(user_array-cached_array)/(1+norm(cached_array)) <= tolerance`. This tolerance could be useful to optimize CPU time. It could be something like `2 * numpy.finfo(float).eps`.

### Export to another format

The [cache][gemseo.core.discipline.discipline.Discipline.cache] can be converted to a [Dataset][gemseo.datasets.dataset.Dataset] for post-processing purposes using its method [to_dataset()][gemseo.caches.base_cache.BaseCache.to_dataset]. It can also be saved into an XML file to be read by [ggobi](http://ggobi.org/) using its method [to_ggobi()][gemseo.caches.base_full_cache.BaseFullCache.to_ggobi].

!!! note
    For the sake of performance, the input value of type `Mapping[str, ndarray | int | float]` is flatten to a NumPy array, hashed using the algorithm XXH64 of the [xxHash library](https://cyan4973.github.io/xxHash/) and the hashed value compare to the ones stored in the [cache][gemseo.core.discipline.discipline.Discipline.cache].

### Set the cache policy of a discipline

The data can be cached either:

- in memory:
    - the [SimpleCache][gemseo.caches.simple_cache.SimpleCache] (default policy) only stores in memory the data associated with the last call to [execute()][gemseo.core.discipline.discipline.Discipline.execute],
    - the [MemoryFullCache][gemseo.caches.memory_full_cache.MemoryFullCache] stores in memory the data associated with all the calls to [execute()][gemseo.core.discipline.discipline.Discipline.execute],
- on the disk:
    - the [HDF5Cache][gemseo.caches.hdf5_cache.HDF5Cache] stores in a node of an HDF file the data associated with all the calls to [execute()][gemseo.core.discipline.discipline.Discipline.execute].

The cache strategy of an [Discipline][gemseo.core.discipline.discipline.Discipline] can be changed with the method [set_cache()][gemseo.core.discipline.discipline.Discipline.set_cache] by passing as first argument the name of the cache class, e.g. `"MemoryFullCache"`.

!!! note
    The types of cache can be extended by subclassing [BaseFullCache][gemseo.caches.base_full_cache.BaseFullCache] or [MemoryFullCache][gemseo.caches.memory_full_cache.MemoryFullCache].
    The [set_cache()][gemseo.core.discipline.discipline.Discipline.set_cache] method will find the new types automatically because it is based on a [CacheFactory][gemseo.caches.factory.CacheFactory].

## Advanced use

### Get metadata

You can easily get:

- the number of entries: `n_entries = len(cache)`,
- the names of the input variables: `input_names = cache.input_names`,
- the names of the output variables: `output_names = cache.output_names`,
- the size of the variables: `size = cache.names_to_sizes[variable_name]`.

### Get the last entry

Use `last_entry = cache.last_entry` to retrieve the last cached data.

`last_entry` is a [CacheEntry][gemseo.caches.cache_entry.CacheEntry] with fields `"inputs`", `"outputs"` and `"jacobian"`, to be used as `output_value = cache_entry.outputs`.

### Clear the cache

Use `cache.clear()` to remove all the entries.

### Handle the cache as a dictionary

A cache can be handled as a dictionary:

- store an output value: `cache[input_value] = (output_value, None)`
- store a Jacobian value: `cache[input_value] = (None, jacobian_value)`
- store both Jacobian and output values: `cache[input_value] = (output_value, jacobian_value)`
- retrieve an entry: `cache_entry = cache[input_value]`.

### Cache data in an HDF file

[HDF5](https://portal.hdfgroup.org) is a standard file format for storing simulation data:

> *"HDF5 is a data model, library, and file format for storing and managing data. It supports an unlimited variety of datatypes, and is designed for flexible and efficient I/O and for high volume and complex data. HDF5 is portable and is extensible, allowing applications to evolve in their use of HDF5. The HDF5 Technology suite includes tools and applications for managing, manipulating, viewing, and analyzing data in the HDF5 format."*

The HDFView application can be used to explore the data of the cache.

![HDFView application](../_images/HDFView_cache.png)
*HDFView of the cache generated by an MDF DOE scenario execution on the SSBJ test case*
