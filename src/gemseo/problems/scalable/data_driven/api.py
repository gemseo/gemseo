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
"""
Scalability study - API
=======================

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

from gemseo.problems.scalable.data_driven.study.post import PostScalabilityStudy
from gemseo.problems.scalable.data_driven.study.process import ScalabilityStudy


def create_scalability_study(
    objective,
    design_variables,
    directory="study",
    prefix="",
    eq_constraints=None,
    ineq_constraints=None,
    maximize_objective=False,
    fill_factor=0.7,
    active_probability=0.1,
    feasibility_level=0.8,
    start_at_equilibrium=True,
    early_stopping=True,
    coupling_variables=None,
):
    """This method creates a :class:`.ScalabilityStudy`. It requires two mandatory
    arguments:

    - the :code:`'objective'` name,
    - the list of :code:`'design_variables'` names.

    Concerning output files, we can specify:

    - the :code:`directory` which is :code:`'study'` by default,
    - the prefix of output file names (default: no prefix).

    Regarding optimization parametrization, we can specify:

    - the list of equality constraints names (:code:`eq_constraints`),
    - the list of inequality constraints names (:code:`ineq_constraints`),
    - the choice of maximizing the objective function
      (:code:`maximize_objective`).

    By default, the objective function is minimized and the MDO problem
    is unconstrained.

    Last but not least, with regard to the scalability methodology,
    we can overwrite:

    - the default fill factor of the input-output dependency matrix
      :code:`ineq_constraints`,
    - the probability to set the inequality constraints as active at
      initial step of the optimization :code:`active_probability`,
    - the offset of satisfaction for inequality constraints
      :code:`feasibility_level`,
    - the use of a preliminary MDA to start at equilibrium
      :code:`start_at_equilibrium`,
    - the post-processing of the optimization database to get results
      earlier than final step :code:`early_stopping`.

    :param str objective: name of the objective
    :param list(str) design_variables: names of the design variables
    :param str directory: working directory of the study. Default: 'study'.
    :param str prefix: prefix for the output filenames. Default: ''.
    :param list(str) eq_constraints: names of the equality constraints.
        Default: None.
    :param list(str) ineq_constraints: names of the inequality constraints
        Default: None.
    :param bool maximize_objective: maximizing objective. Default: False.
    :param float fill_factor: default fill factor of the input-output
        dependency matrix. Default: 0.7.
    :param float active_probability: probability to set the inequality
        constraints as active at initial step of the optimization.
        Default: 0.1
    :param float feasibility_level: offset of satisfaction for inequality
        constraints. Default: 0.8.
    :param bool start_at_equilibrium: start at equilibrium
        using a preliminary MDA. Default: True.
    :param bool early_stopping: post-process the optimization database
        to get results earlier than final step.
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


def plot_scalability_results(study_directory):
    """This method plots the set of :class:`.ScalabilityResult` generated by a
    :class:`.ScalabilityStudy` and located in the directory created by this study.

    :param str study_directory: directory of the scalability study.
    """
    return PostScalabilityStudy(study_directory)
