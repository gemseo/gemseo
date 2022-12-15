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


#############################################################################
# Create, execute and post-process MDA
# ------------------------------------

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
