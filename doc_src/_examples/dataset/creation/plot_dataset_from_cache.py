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
Dataset from a cache
====================

In this example, we will see how to build a :class:`.Dataset` from objects
of an :class:`.AbstractFullCache`.
For that, we need to import this :class:`.Dataset` class:
"""
from __future__ import annotations

from gemseo.api import configure_logger
from gemseo.caches.memory_full_cache import MemoryFullCache
from numpy import array

configure_logger()


##############################################################################
# Synthetic data
# --------------
# Let us consider a :class:`.MemoryFullCache` storing two parameters:
#
# - x with dimension 1 which is a cache input,
# - y with dimension 2 which is a cache output.

cache = MemoryFullCache()
cache[{"x": array([1.0])}] = ({"y": array([2.0, 3.0])}, None)
cache[{"x": array([4.0])}] = ({"y": array([5.0, 6.0])}, None)

##############################################################################
# Create a dataset
# ----------------
# We can easily build a dataset from this :class:`.MemoryFullCache`,
# either by separating the inputs from the outputs (default option):
dataset = cache.export_to_dataset("toy_cache")
print(dataset)
##############################################################################
# or by considering all features as default parameters:
dataset = cache.export_to_dataset("toy_cache", categorize=False)
print(dataset)

##############################################################################
# Access properties
# -----------------
dataset = cache.export_to_dataset("toy_cache")
##############################################################################
# Variables names
# ~~~~~~~~~~~~~~~
# We can access the variables names:
print(dataset.variables)

##############################################################################
# Variables sizes
# ~~~~~~~~~~~~~~~
# We can access the variables sizes:
print(dataset.sizes)

##############################################################################
# Variables groups
# ~~~~~~~~~~~~~~~~
# We can access the variables groups:
print(dataset.groups)

##############################################################################
# Access data
# -----------
# Access by group
# ~~~~~~~~~~~~~~~
# We can get the data by group, either as an array (default option):
print(dataset.get_data_by_group("inputs"))
##############################################################################
# or as a dictionary indexed by the variables names:
print(dataset.get_data_by_group("inputs", True))

##############################################################################
# Access by variable name
# ~~~~~~~~~~~~~~~~~~~~~~~
# We can get the data by variables names,
# either as a dictionary indexed by the variables names (default option):
print(dataset.get_data_by_names(["x"]))
##############################################################################
# or as an array:
print(dataset.get_data_by_names(["x", "y"], False))

##############################################################################
# Access all data
# ~~~~~~~~~~~~~~~
# We can get all the data, either as a large array:
print(dataset.get_all_data())
##############################################################################
# or as a dictionary indexed by variables names:
print(dataset.get_all_data(as_dict=True))
##############################################################################
# We can get these data sorted by category, either with a large array for each
# category:
print(dataset.get_all_data(by_group=False))
##############################################################################
# or with a dictionary of variables names:
print(dataset.get_all_data(by_group=False, as_dict=True))
