# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""
Scalable module from Tedford and Martins (2010)
***********************************************

The modules located in the **scalable_tm** directory offer a set of
classes relative to the scalable problem introduced in the paper:

    Tedford NP, Martins JRRA (2010), Benchmarking
    multidisciplinary design optimization algorithms,
    Optimization and Engineering, 11(1):159-183.

Overview
~~~~~~~~

This scalable problem aims to minimize an objective function
quadratically depending on shared design parameters and coupling variables,
under inequality constraints linearly depending on these coupling variables.

System discipline
-----------------

A system discipline computes the constraints and the objective
in function of the shared design parameters and coupling variables.

Strongly coupled disciplines
----------------------------

The coupling variables are the outputs of strongly coupled disciplines.

Each strongly coupled discipline computes a set of coupling variables
linearly depending on local design parameters, shared design parameters,
coupling variables from other strongly coupled disciplines,
and belonging to the unit hypercube.

Scalability
-----------

This problem is said "scalable"
because several sizing features can be chosen by the user:

- the number of local design parameters for each discipline,
- the number of shared design parameters,
- the number of coupling variables for each discipline,
- the number of disciplines.

A given sizing configuration is called "scaling strategy"
and this scalable module is particularly useful to compare different MDO
formulations with respect to the scaling strategy.

Implementation
~~~~~~~~~~~~~~

The scalable problem
--------------------

The :class:`.TMScalableProblem` class instantiates the disciplines of the
problem, that are :class:`.TMMainDiscipline` and several :class:`.TMSubDiscipline`,
as well as the  :class:`.DesignSpace`.
These instantiated objects can be used in a :class:`.Scenario`,
e.g. :class:`.MDOScenario` or :class:`.DOEScenario`.

The scalable study
------------------

The :class:`.TMScalableStudy` class instantiates a :class:`.TMScalableProblem`
for a particular scaling strategy
where the number of local design parameters is the same for all disciplines
as well as the number of coupling variables.
It provides a method to run MDO formulations and graphical capabilities
to analyze the results.

The parametric scalable study
-----------------------------

The :class:`.TMParamSS` class instantiates several :class:`.TMScalableStudy`
associated with different scaling strategies, e.g. different numbers of local
design parameters or different numbers of coupling variables.
It provides a method to run MDO formulations and save results on the disk.
The :class:`.TMParamSSPost` provides graphical capabilities to post-process
these saved results.
"""
from __future__ import annotations
