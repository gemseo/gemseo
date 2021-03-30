# -*- coding: utf-8 -*-
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
DOE algorithms
==============

In this example, we will discover the different functions of the API
related to design of experiments.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

from future import standard_library

from gemseo.api import (
    configure_logger,
    get_algorithm_options_schema,
    get_available_doe_algorithms,
)

configure_logger()

standard_library.install_aliases()

##############################################################################
# Get available DOE algorithms
# ----------------------------
#
# The :meth:`~gemseo.api.get_available_doe_algorithms` function returns the list
# of optimization algorithms available in |g| or in external modules
print(get_available_doe_algorithms())

##########################################################################
# Get options schema
# ------------------
# For a given optimization algorithm, e.g. :code:`"DiagonalDOE"`,
# we can get the options; e.g.
print(get_algorithm_options_schema("DiagonalDOE"))
