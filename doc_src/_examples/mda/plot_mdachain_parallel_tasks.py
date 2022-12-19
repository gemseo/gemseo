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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Jean-Christophe Giret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
MDAChain with independent parallel MDAs
=======================================

This example illustrates the possibility to parallelize MDAs in a MDAChain,
given that these MDA are independent and can be run in parallel.
"""
from __future__ import annotations

from gemseo.api import configure_logger
from gemseo.api import create_discipline
from gemseo.mda.mda_chain import MDAChain

configure_logger()

################################################
# Introduction
# ----------------------------------------------
#
# In a :class:`.MDAChain`,
# there may be an opportunity to parallelize the execution of :class:`.MDA`
# that can be executed independently.
# As an example,
# let us consider the following expressions,
# which will be used to instantiate analytic disciplines:

disciplines_expressions = [
    {"a": "x"},
    {"y1": "x1", "b": "a+1"},
    {"x1": "1.-0.3*y1"},
    {"y2": "x2", "c": "a+2"},
    {"x2": "1.-0.3*y2"},
    {"obj1": "x1+x2"},
    {"obj2": "b+c"},
    {"obj": "obj1+obj2"},
]

# %%
# We can easily observe in these disciplines,
# that the :math:`x_1` and :math:`y_1` variables are strongly coupled.
# It follows that the second and third disciplines are strongly coupled and
# constitute a :class:`.MDA`.
#
#
# The same statement can be done for the disciplines that provide the output variables
# :math:`x_2` and :math:`y_2`,
# and the fourth and fifth disciplines which are also strongly coupled.
# These two MDAs are independent and only depend on the variable :math:`a` given by
# the first discipline.
#
#
# Thus,
# they can be run in parallel,
# hence reducing the overall :class:`.MDAChain` execution provided that enough
# resources are available on the computing node (in our case, at least two CPUs).
# By default, the parallel execution of the independent :class:`.MDA` are deactivated,
# meaning that the execution of the two independent :class:`.MDA` will remain sequential.
# Yet, a parallel execution of the two :class:`.MDA` can be  activated using the
# `mdachain_parallelize_task` boolean option.
#
#
# If activated,
# the user has also the possibility to provide parallelization options,
# such as using either threads or processes to perform the parallelization,
# or the number of processes or threads to use.
# By default, as more lightweight, threading is used but on some specific case,
# where for instance race conditions may occur,
# multiprocessing can be employed.

####################################################
# Example of :class:`.MDAChain` with parallelization
# --------------------------------------------------
#
# We are here using the disciplines previously defined by their analytical expressions,
# and we are going to explicitly ask for a parallelization of the execution of the
# two independent :class:`.MDA`.

disciplines = []
for expr in disciplines_expressions:
    disciplines.append(create_discipline("AnalyticDiscipline", expressions=expr))

mdo_parallel_chain_options = {"use_threading": True, "n_processes": 2}
mdachain = MDAChain(
    disciplines,
    name="mdachain_lower",
    mdachain_parallelize_tasks=True,
    mdachain_parallel_options=mdo_parallel_chain_options,
)
res = mdachain.execute()
