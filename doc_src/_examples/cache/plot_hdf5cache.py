# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This work is licensed under a BSD 0-Clause License.
#
# Permission to use, copy, modify, and/or distribute this software
# for any purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL
# WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL
# THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT,
# OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING
# FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
# NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
# WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
HDF5 cache
==========

In this example, we will see how to use :class:`.HDF5Cache`.
"""
from __future__ import annotations

from gemseo.api import configure_logger
from gemseo.caches.hdf5_cache import HDF5Cache
from numpy import array

configure_logger()

# %%
#
# Import
# ------
# Let's first import the :class:`array` and the :class:`.HDF5Cache` classes.


# %%
# Create
# ------
# An instance of :class:`.HDF5Cache` can be instantiated with the following
# statement.  The user has to provide the file path of the HDF5 file, as well
# as the node name, which usually is a discipline name.
#
# .. warning::
#     The :class:`.HDF5Cache` relies on some multiprocessing features. When working on
#     Windows, the execution of scripts containing instances of :class:`.HDF5Cache`
#     must be protected by an ``if __name__ == '__main__':`` statement.
#     Currently, the use of an HDF5Cache is not supported in parallel on Windows
#     platforms. This is due to the way subprocesses are forked in this architecture.
#     The method :meth:`.DOEScenario.set_optimization_history_backup`
#     is recommended as an alternative.

cache = HDF5Cache("my_cache.hdf5", "node1")

# %%
# It is possible to see the principal attributes of the cache by printing it,
# either using a print statement or using the logger:
print(cache)

# %%
# Cache
# -----
# In this example, we manually add data in the cache from the data dictionary
# to illustrate its use.  Yet, it has to be noted that a cache can be attached
# to an :class:`.MDODiscipline` instance, and the user does not have to feed the
# cache manually.
# Here, we provide to the cache the data dictionary, and we set `x` as input
# and `y` as output.

cache[{"x": array([1.0])}] = ({"y": array([2.0])}, None)
cache[{"x": array([2.0])}] = ({"y": array([3.0])}, None)
print(cache)

# %%
# Get all data
# ------------
# We can now print some information from the cache, such as its length. We can
# also display all the cached data so far.

print(len(cache))
for data in cache:
    print(data)

# %%
# Get last cached data
# --------------------
# It is also possible to display the last entry cached, for the inputs and the
# outputs.

last_entry = cache.last_entry
print(last_entry.inputs)
print(last_entry.outputs)

# %%
# Clear the cache
# ---------------
# It is also possible to clear the cache, which removes all the data which has
# been stored so far in the HDF5 file.

cache.clear()
print(cache)
