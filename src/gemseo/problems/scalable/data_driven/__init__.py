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
#         documentation
#        :author:  Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""A scalable methodology to test MDO formulation on benchmark or real problems.

This API facilitates the use of the :mod:`gemseo.problems.scalable.data_driven.study`
package implementing classes to benchmark MDO formulations
based on scalable disciplines.

:class:`.ScalabilityStudy` class implements the concept of scalability study:

1. By instantiating a :class:`.ScalabilityStudy`, the user defines
   the MDO problem in terms of design parameters, objective function and
   constraints.
2. For each discipline, the user adds a dataset stored
   in a :class:`.Dataset` and select a type of
   :class:`.ScalableModel` to build the :class:`.ScalableDiscipline`
   associated with this discipline.
3. The user adds different optimization strategies, defined in terms
   of both optimization algorithms and MDO formulation.
4. The user adds different scaling strategies, in terms of sizes of
   design parameters, coupling variables and equality and inequality
   constraints. The user can also define a scaling strategies according to
   particular parameters rather than groups of parameters.
5. Lastly, the user executes the :class:`.ScalabilityStudy` and the results
   are written in several files and stored into directories
   in a hierarchical way, where names depend on both MDO formulation,
   scaling strategy and replications when it is necessary. Different kinds
   of files are stored: optimization graphs, dependency matrix plots and
   of course, scalability results by means of a dedicated class:
   :class:`.ScalabilityResult`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.problems.scalable.data_driven.study.post import PostScalabilityStudy
from gemseo.problems.scalable.data_driven.study.process import ScalabilityStudy

if TYPE_CHECKING:
    from collections.abc import Iterable


def create_scalability_study(
    objective: str,
    design_variables: Iterable[str],
    directory: str = "study",
    prefix: str = "",
    eq_constraints: Iterable[str] | None = None,
    ineq_constraints: Iterable[str] | None = None,
    maximize_objective: bool = False,
    fill_factor: float = 0.7,
    active_probability: float = 0.1,
    feasibility_level: float = 0.8,
    start_at_equilibrium: bool = True,
    early_stopping: bool = True,
    coupling_variables: Iterable[str] | None = None,
) -> ScalabilityStudy:
    """This method creates a :class:`.ScalabilityStudy`. It requires two mandatory
    arguments:

    - the ``'objective'`` name,
    - the list of ``'design_variables'`` names.

    Concerning output files, we can specify:

    - the ``directory`` which is ``'study'`` by default,
    - the prefix of output file names (default: no prefix).

    Regarding optimization parametrization, we can specify:

    - the list of equality constraints names (``eq_constraints``),
    - the list of inequality constraints names (``ineq_constraints``),
    - the choice of maximizing the objective function
      (``maximize_objective``).

    By default, the objective function is minimized and the MDO problem
    is unconstrained.

    Last but not least, with regard to the scalability methodology,
    we can overwrite:

    - the default fill factor of the input-output dependency matrix
      ``ineq_constraints``,
    - the probability to set the inequality constraints as active at
      initial step of the optimization ``active_probability``,
    - the offset of satisfaction for inequality constraints
      ``feasibility_level``,
    - the use of a preliminary MDA to start at equilibrium
      ``start_at_equilibrium``,
    - the post-processing of the optimization database to get results
      earlier than final step ``early_stopping``.

    Args:
        objective: The name of the objective.
        design_variables: The names of the design variables.
        directory: The working directory of the study.
        prefix: The prefix for the output filenames.
        eq_constraints: The names of the equality constraints, if any.
        ineq_constraints: The names of the inequality constraints, if any.
        maximize_objective: Whether to maximize the objective.
        fill_factor: The default fill factor
            of the input-output dependency matrix.
        active_probability: The probability to set the inequality
            constraints as active at initial step of the optimization.
        feasibility_level: The offset of satisfaction
            for the inequality constraints.
        start_at_equilibrium: Whether to start at equilibrium
            using a preliminary MDA.
        early_stopping: Whether to post-process the optimization database
            to get results earlier than final step.
        coupling_variables: The names of the coupling variables.
    """
    return ScalabilityStudy(
        objective,
        design_variables,
        directory,
        prefix,
        eq_constraints,
        ineq_constraints,
        maximize_objective,
        fill_factor,
        active_probability,
        feasibility_level,
        start_at_equilibrium,
        early_stopping,
        coupling_variables,
    )


def plot_scalability_results(study_directory: str) -> PostScalabilityStudy:
    """This method plots the set of :class:`.ScalabilityResult` generated by a
    :class:`.ScalabilityStudy` and located in the directory created by this study.

    Args:
        study_directory: The directory of the scalability study.
    """
    return PostScalabilityStudy(study_directory)
