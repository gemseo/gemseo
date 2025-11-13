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
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Benchmark MDO formulations based on scalable disciplines.

The
[gemseo.problems.mdo.scalable.data_driven.study][gemseo.problems.mdo.scalable.data_driven.study]
package implements several classes
to benchmark MDO formulations based on scalable disciplines.

The
[ScalableDiagonalModel][gemseo.problems.mdo.scalable.data_driven.diagonal.ScalableDiagonalModel]
class implements
the concept of scalability study:

1. By instantiating a
   [ScalableDiagonalModel][gemseo.problems.mdo.scalable.data_driven.diagonal.ScalableDiagonalModel],
   the user defines the MDO problem in terms of design parameters,
   objective function and constraints.
2. For each discipline, the user adds a dataset stored
   in a [BaseFullCache][gemseo.caches.base_full_cache.BaseFullCache]
   and select a type of
   [ScalableModel][gemseo.problems.mdo.scalable.data_driven.model.ScalableModel]
   to build the
   [DataDrivenScalableDiscipline][gemseo.problems.mdo.scalable.data_driven.discipline.DataDrivenScalableDiscipline]
   associated with this discipline.
3. The user adds different optimization strategies, defined in terms
   of both optimization algorithms and MDO formulation.
4. The user adds different scaling strategies, in terms of sizes of
   design parameters, coupling variables and equality and inequality
   constraints. The user can also define a scaling strategies according to
   particular parameters rather than groups of parameters.
5. Lastly, the user executes the
   [ScalableDiagonalModel][gemseo.problems.mdo.scalable.data_driven.diagonal.ScalableDiagonalModel]
   and the results are written in several files and stored into directories
   in a hierarchical way, where names depend on both MDO formulation,
   scaling strategy and replications when it is necessary. Different kinds
   of files are stored: optimization graphs, dependency matrix plots and
   of course, scalability results by means of a dedicated class:
   [ScalabilityResult][gemseo.problems.mdo.scalable.data_driven.study.result.ScalabilityResult].

The
[PostScalabilityStudy][gemseo.problems.mdo.scalable.data_driven.study.post.PostScalabilityStudy]
class implements the way as the set of
[ScalabilityResult][gemseo.problems.mdo.scalable.data_driven.study.result.ScalabilityResult]
-based result files
contained in the study directory are graphically post-processed. This class
provides several methods to easily change graphical properties, notably
the plot labels. It also makes it possible to define a cost function per
MDO formulation, converting the numbers of executions and linearizations
of the different disciplines required by an MDO process in an estimation
of the computational cost associated with what would be a scaled version
of the true problem.

Warning:
   Comparing MDO formulations in terms of estimated true computational time
   rather than CPU time of the
   [ScalableDiagonalModel][gemseo.problems.mdo.scalable.data_driven.diagonal.ScalableDiagonalModel]
   is highly recommended.
   Indeed, time is often an obviousness criterion to distinguish between
   MDO formulations having the same performance in terms of distance to the
   optimum: look at our calculation budget and choose the best formulation
   that satisfies this budget, or even saves us time. Thus, it is important
   to carefully define these cost functions.
"""

from __future__ import annotations
