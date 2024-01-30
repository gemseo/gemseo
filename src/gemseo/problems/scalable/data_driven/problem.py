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
"""Scalable MDO problem.

This module implements the concept of scalable problem by means of the
:class:`.ScalableProblem` class.

Given

- an MDO scenario based on a set of sampled disciplines
  with a particular problem dimension,
- a new problem dimension (= number of inputs and outputs),

a scalable problem:

1. makes each discipline scalable based on the new problem dimension,
2. creates the corresponding MDO scenario.

Then, this MDO scenario can be executed and post-processed.

We can repeat this tasks for different sizes of variables
and compare the scalability, which is the dependence of the scenario results
on the problem dimension.

.. seealso:: :class:`.MDODiscipline`, :class:`.ScalableDiscipline`
   and :class:`.Scenario`
"""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

from numpy import array
from numpy import full
from numpy import ones
from numpy import where
from numpy import zeros
from numpy.random import default_rng

from gemseo import SEED
from gemseo import create_design_space
from gemseo import create_scenario
from gemseo import generate_coupling_graph
from gemseo import generate_n2_plot
from gemseo.algos.design_space import DesignSpace
from gemseo.core.coupling_structure import MDOCouplingStructure
from gemseo.disciplines.utils import get_all_inputs
from gemseo.mda.mda_factory import MDAFactory
from gemseo.problems.scalable.data_driven.discipline import ScalableDiscipline
from gemseo.utils.string_tools import MultiLineString

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from numpy._typing import NDArray

    from gemseo.core.discipline import MDODiscipline
    from gemseo.core.scenario import Scenario
    from gemseo.datasets.io_dataset import IODataset

LOGGER = logging.getLogger(__name__)


class ScalableProblem:
    """Scalable problem."""

    def __init__(
        self,
        datasets: Iterable[IODataset],
        design_variables: Iterable[str],
        objective_function: str,
        eq_constraints: Iterable[str] | None = None,
        ineq_constraints: Iterable[str] | None = None,
        maximize_objective: bool = False,
        sizes: Mapping[str, int] | None = None,
        **parameters: Any,
    ) -> None:
        """
        Args:
            datasets: One input-output dataset per discipline.
            design_variables: The names of the design variables.
            objective_function: The name of the objective.
            eq_constraints: The names of the equality constraints, if any.
            ineq_constraints: The names of the inequality constraints, if any.
            maximize_objective: Whether to maximize the objective.
            sizes: The sizes of the inputs and outputs.
                If ``None``, use the original sizes.
            **parameters: The optional parameters of the scalable model.
        """  # noqa: D205, D212, D415
        self.disciplines = [dataset.name for dataset in datasets]
        self.data = {dataset.name: dataset for dataset in datasets}
        self.inputs = {
            dataset.name: dataset.get_variable_names(dataset.INPUT_GROUP)
            for dataset in datasets
        }
        self.outputs = {
            dataset.name: dataset.get_variable_names(dataset.OUTPUT_GROUP)
            for dataset in datasets
        }
        self.varsizes = {}
        for dataset in datasets:
            self.varsizes.update(dataset.variable_names_to_n_components)
        self.design_variables = design_variables
        self.objective_function = objective_function
        self.ineq_constraints = ineq_constraints
        self.eq_constraints = eq_constraints
        self.maximize_objective = maximize_objective
        self.scaled_disciplines = []
        self.scaled_sizes = {}
        self._build_scalable_disciplines(sizes, **parameters)
        self.scenario = None

    def __str__(self) -> str:
        disciplines = ", ".join(self.disciplines)
        design_variables = None
        if self.design_variables is not None:
            design_variables = ", ".join(self.design_variables)
        ineq_constraints = None
        if self.ineq_constraints is not None:
            ineq_constraints = ", ".join(self.ineq_constraints)
        eq_constraints = None
        if self.eq_constraints is not None:
            eq_constraints = ", ".join(self.eq_constraints)
        sizes = [name + f" ({size})" for name, size in self.scaled_sizes.items()]
        sizes = ", ".join(sizes)
        optimize = "maximize" if self.maximize_objective else "minimize"
        msg = MultiLineString()
        msg.add("MDO problem")
        msg.indent()
        msg.add("Disciplines: {}", disciplines)
        msg.add("Design variables: {}", design_variables)
        msg.add("Objective function: {} (to {})", self.objective_function, optimize)
        msg.add("Inequality constraints: {}", ineq_constraints)
        msg.add("Equality constraints: {}", eq_constraints)
        msg.add("Sizes: {}", sizes)
        return str(msg)

    def plot_n2_chart(self, save: bool = True, show: bool = False) -> None:
        """Plot a N2 chart.

        Args:
            save: Whether to save the figure.
            show: Whether to display the figure.
        """
        generate_n2_plot(self.scaled_disciplines, save=save, show=show)

    def plot_coupling_graph(self) -> None:
        """Plot a coupling graph."""
        generate_coupling_graph(self.scaled_disciplines)

    def plot_1d_interpolations(
        self,
        save: bool = True,
        show: bool = False,
        step: float = 0.01,
        varnames: Sequence[str] | None = None,
        directory: Path | str = ".",
        png: bool = False,
    ):
        """Plot 1d interpolations.

        Args:
            save: Whether to save the figure.
            show: Whether to display the figure.
            step: The step to evaluate the 1d interpolation function.
            varnames: The names of the variable to plot.
                If ``None``, all the variables are plotted.
            directory: The directory path.
            png: Whether to use PNG file format instead of PDF.
        """
        directory = Path(directory)
        directory.mkdir(exist_ok=True)
        file_paths = []
        for scalable_discipline in self.scaled_disciplines:
            func = scalable_discipline.scalable_model.plot_1d_interpolations
            file_names = func(save, show, step, varnames, directory, png)
            file_paths += [directory / file_name for file_name in file_names]
        return file_paths

    def plot_dependencies(
        self, save: bool = True, show: bool = False, directory: str = "."
    ):
        """Plot dependency matrices.

        Args:
            save: Whether to save the figure.
            show: Whether to display the figure.
            directory: The directory path.
        """
        fnames = []
        for scalable_discipline in self.scaled_disciplines:
            scalable_model = scalable_discipline.scalable_model
            plot_dependency = scalable_model.plot_dependency
            fname = plot_dependency(
                add_levels=True, save=save, show=show, directory=directory
            )
            fnames.append(fname)
        return fnames

    def _build_scalable_disciplines(
        self, sizes: Mapping[str, int] | None = None, **parameters: Any
    ) -> None:
        """Build the scalable disciplines.

        Args:
            size: The sizes of the inputs and outputs.
            **parameters: The options of the scalable disciplines.
        """
        copied_parameters = deepcopy(parameters)
        for disc in self.disciplines:
            varnames = self.inputs[disc] + self.outputs[disc]
            sizes = sizes or {}
            new_varsizes = {
                varname: sizes.get(varname, self.varsizes[varname])
                for varname in varnames
            }
            if "group_dep" in parameters:
                copied_parameters["group_dep"] = parameters["group_dep"][disc]
            if "fill_factor" in parameters:
                copied_parameters["fill_factor"] = parameters["fill_factor"][disc]
            self.scaled_disciplines.append(
                ScalableDiscipline(
                    "ScalableDiagonalModel",
                    self.data[disc],
                    new_varsizes,
                    **copied_parameters,
                )
            )
            self.scaled_sizes.update(deepcopy(new_varsizes))

    def create_scenario(
        self,
        formulation: str = "DisciplinaryOpt",
        scenario_type: str = "MDO",
        start_at_equilibrium: bool = False,
        active_probability: float = 0.1,
        feasibility_level: float = 0.5,
        **options,
    ) -> Scenario:
        """Create a :class:`.Scenario` from the scalable disciplines.

        Args:
            formulation: The MDO formulation to use for the scenario.
            scenario_type: The type of scenario, either ``MDO`` or ``DOE``.
            start_at_equilibrium: Whether to start at equilibrium using a preliminary
                MDA.
            active_probability: The probability to set the inequality constraints as
                active at the initial step of the optimization.
            feasibility_level: The offset of satisfaction for inequality
                constraints.
            **options: The formulation options.

        Returns:
            The :class:`.Scenario` from the scalable disciplines.
        """
        equilibrium = {}
        if start_at_equilibrium:
            equilibrium = self.__get_equilibrium()

        disciplines = self.scaled_disciplines
        design_space = self._create_design_space(disciplines, formulation)
        if formulation == "BiLevel":
            self.scenario = self._create_bilevel_scenario(disciplines, **options)
        else:
            self.scenario = create_scenario(
                disciplines,
                formulation,
                self.objective_function,
                deepcopy(design_space),
                scenario_type=scenario_type,
                maximize_objective=self.maximize_objective,
                **options,
            )
        self.__add_ineq_constraints(active_probability, feasibility_level, equilibrium)
        self.__add_eq_constraints(equilibrium)
        return self.scenario

    def _create_bilevel_scenario(
        self, disciplines: Iterable[MDODiscipline], **sub_scenario_options
    ) -> Scenario:
        """Create a bi-level scenario from disciplines.

        Args:
            disciplines: The disciplines.
            **sub_scenario_options: The options of the sub-scenarios.

        Returns:
            A scenario using a bi-level formulation.
        """
        cpl_structure = MDOCouplingStructure(disciplines)
        st_cpl_disciplines = cpl_structure.strongly_coupled_disciplines
        wk_cpl_disciplines = cpl_structure.weakly_coupled_disciplines()
        obj = self.objective_function
        max_obj = self.maximize_objective

        # Construction of the subsystem scenarios
        sub_scenarios = []
        sub_inputs = []
        for discipline in st_cpl_disciplines:
            cplt_disciplines = list(set(disciplines) - {discipline})
            sub_disciplines = [discipline, *wk_cpl_disciplines]
            design_space = DesignSpace()
            inputs = get_all_inputs([discipline])
            all_inputs = get_all_inputs(cplt_disciplines)
            inputs = list(set(inputs) - set(all_inputs))
            sub_inputs += inputs
            for name in inputs:
                design_space.add_variable(
                    name, self.scaled_sizes[name], "float", 0.0, 1.0, 0.5
                )
            sub_scenarios.append(
                create_scenario(
                    sub_disciplines,
                    "DisciplinaryOpt",
                    obj,
                    design_space,
                    maximize_objective=max_obj,
                )
            )
            sub_scenarios[-1].default_inputs = sub_scenario_options

        # Construction of the system scenario
        all_inputs = get_all_inputs(disciplines)
        inputs = list(set(all_inputs) - set(sub_inputs))
        design_space = DesignSpace()
        for name in inputs:
            design_space.add_variable(
                name, self.scaled_sizes[name], "float", 0.0, 1.0, 0.5
            )
        sub_disciplines = sub_scenarios + wk_cpl_disciplines
        return create_scenario(
            sub_disciplines,
            "BiLevel",
            obj,
            design_space,
            maximize_objective=max_obj,
            mda_name="MDAJacobi",
            tolerance=1e-8,
        )

    def _create_design_space(
        self, disciplines: Sequence[MDODiscipline], formulation: str = "DisciplinaryOpt"
    ) -> DesignSpace:
        """Create a design space into the unit hypercube.

        Args:
            disciplines: The disciplines.
            formulation: The name of the formulation.

        Returns:
            The design space.
        """
        design_space = create_design_space()
        for name in self.design_variables:
            size = self.scaled_sizes[name]
            design_space.add_variable(
                name,
                size=size,
                var_type="float",
                l_b=zeros(size),
                u_b=ones(size),
                value=full(size, 0.5),
            )

        if formulation == "IDF":
            coupling_structure = MDOCouplingStructure(disciplines)
            all_couplings = set(coupling_structure.all_couplings)
            for name in all_couplings:
                size = self.scaled_sizes[name]
                design_space.add_variable(
                    name,
                    size=size,
                    var_type="float",
                    l_b=zeros(size),
                    u_b=ones(size),
                    value=full(size, 0.5),
                )

        return design_space

    def __get_equilibrium(
        self, mda_name: str = "MDAJacobi", **options: Any
    ) -> dict[str, NDArray[float]]:
        """Get the equilibrium point from an MDA method.

        Args:
            mda_name: The name of the MDA.

        Returns:
            The equilibrium point.
        """
        LOGGER.info("Build a preliminary MDA to start at equilibrium")
        factory = MDAFactory()
        mda = factory.create(mda_name, self.scaled_disciplines, **options)
        if len(mda.strong_couplings) == 0:
            mda = factory.create("MDAQuasiNewton", self.scaled_disciplines, **options)
        return mda.execute()

    def __add_ineq_constraints(
        self,
        active_probability: float,
        feasibility_level: float,
        equilibrium: Mapping[str, NDArray[float]],
    ) -> None:
        """Add the inequality constraints.

        Args:
            active_probability: The probability to set the inequality constraints
                as active at initial step of the optimization.
            feasibility_level: The offset of satisfaction
                for the inequality constraints.
            equilibrium: The starting point at equilibrium.
        """
        if not hasattr(feasibility_level, "__len__"):
            feasibility_level = dict.fromkeys(self.ineq_constraints, feasibility_level)
        for constraint, alphai in feasibility_level.items():
            if constraint in list(equilibrium.keys()):
                sample = default_rng(SEED).random(len(equilibrium[constraint]))
                val = equilibrium[constraint]
                taui = where(
                    sample < active_probability, val, alphai + (1 - alphai) * val
                )
            else:
                taui = 0.0
            self.scenario.add_constraint(constraint, constraint_type="ineq", value=taui)

    def __add_eq_constraints(self, equilibrium: Mapping[str, NDArray[float]]) -> None:
        """Add equality constraints.

        Args:
            equilibrium: The starting point at equilibrium.
        """
        for constraint in self.eq_constraints:
            self.scenario.add_constraint(
                constraint, value=equilibrium.get(constraint, array([0.0]))[0]
            )

    def exec_time(self, do_sum: bool = True) -> float | list[float]:
        """Get the total execution time.

        Args:
            do_sum: Whether to sum the disciplinary execution times.

        Returns:
            Either the total execution time
            or the total execution times per disciplines.
        """
        exec_time = [discipline.exec_time for discipline in self.scenario.disciplines]
        if do_sum:
            exec_time = sum(exec_time)
        return exec_time

    @property
    def n_calls_top_level(self) -> dict[str, int]:
        """The number of top-level disciplinary calls per discipline."""
        disciplines = self.scenario.formulation.get_top_level_disc()
        return {discipline.name: discipline.n_calls for discipline in disciplines}

    @property
    def n_calls_linearize_top_level(self) -> dict[str, int]:
        """The number of top-level disciplinary linearizations per discipline."""
        disciplines = self.scenario.formulation.get_top_level_disc()
        return {
            discipline.name: discipline.n_calls_linearize for discipline in disciplines
        }

    @property
    def n_calls(self) -> dict[str, int]:
        """The number of disciplinary calls per discipline."""
        return {
            discipline.name: discipline.n_calls
            for discipline in self.scenario.disciplines
        }

    @property
    def n_calls_linearize(self) -> dict[str, int]:
        """The number of disciplinary linearizations per discipline."""
        return {
            discipline.name: discipline.n_calls_linearize
            for discipline in self.scenario.disciplines
        }

    @property
    def status(self) -> int:
        """The status of the scenario."""
        return self.scenario.optimization_result.status

    @property
    def is_feasible(self) -> bool:
        """Whether the solution is feasible."""
        return self.scenario.optimization_result.is_feasible
