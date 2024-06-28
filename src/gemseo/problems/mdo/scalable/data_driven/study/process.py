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
"""Scalability study - Process.

The :class:`.ScalabilityStudy` class implements
the concept of scalability study:

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

import logging
import numbers
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

from numpy import inf

from gemseo.problems.mdo.scalable.data_driven.problem import ScalableProblem
from gemseo.problems.mdo.scalable.data_driven.study.result import ScalabilityResult
from gemseo.utils.logging_tools import LOGGING_SETTINGS
from gemseo.utils.logging_tools import LoggingContext
from gemseo.utils.string_tools import MultiLineString

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from gemseo.datasets.io_dataset import IODataset
    from gemseo.typing import StrKeyMapping

LOGGER = logging.getLogger(__name__)

RESULTS_DIRECTORY = Path("results")
POST_DIRECTORY = Path("visualization")
POSTOPTIM_DIRECTORY = POST_DIRECTORY / "optimization_history"
POSTSTUDY_DIRECTORY = POST_DIRECTORY / "scalability_study"
POSTSCAL_DIRECTORY = POST_DIRECTORY / "dependency_matrix"


class ScalabilityStudy:
    """Scalability Study."""

    def __init__(
        self,
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
    ) -> None:
        """Constructor.

        The constructor of the ScalabilityStudy class requires two mandatory arguments:

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
        LOGGER.info("Initialize the scalability study")
        self.prefix = prefix
        self.directory = Path(directory)
        self.__create_directories()
        self.datasets = []
        self.objective = objective
        self.design_variables = design_variables
        self.eq_constraints = eq_constraints or []
        self.ineq_constraints = ineq_constraints or []
        self.coupling_variables = coupling_variables or []
        self.maximize_objective = maximize_objective
        self.formulations = []
        self.formulations_options = []
        self.algorithms = []
        self.algorithms_options = []
        self.scalings = []
        self.var_scalings = []
        self.__check_fill_factor(fill_factor)
        self._default_fill_factor = fill_factor
        self.__check_proportion(active_probability)
        self.active_probability = active_probability
        if isinstance(feasibility_level, dict):
            for value in feasibility_level.values():
                self.__check_proportion(value)
        else:
            self.__check_proportion(feasibility_level)
        self.feasibility_level = feasibility_level
        self.start_at_equilibrium = start_at_equilibrium
        self.results = []
        self.early_stopping = early_stopping
        self._group_dep = {}
        self._fill_factor = {}
        self.top_level_diff = []
        self._all_data = None
        optimize = "maximize" if self.maximize_objective else "minimize"
        msg = MultiLineString()
        msg.indent()
        msg.add("Optimization problem")
        msg.indent()
        msg.add("Objective: {} {}", optimize, self.objective)
        msg.add("Design variables: {}", self.design_variables)
        msg.add("Equality constraints: {}", self.eq_constraints)
        msg.add("Inequality constraints: {}", self.ineq_constraints)
        msg.dedent()
        msg.add("Study properties")
        msg.indent()
        msg.add("Default fill factor: {}", self._default_fill_factor)
        msg.add("Active probability: {}", self.active_probability)
        msg.add("Feasibility level: {}", self.feasibility_level)
        msg.add("Start at equilibrium: {}", self.start_at_equilibrium)
        msg.add("Early stopping: {}", self.early_stopping)
        LOGGER.info("%s", msg)

    def __create_directories(self) -> None:
        """Create the different directories to store results, post-processings, ..."""
        self.directory.mkdir(exist_ok=True)
        post = self.directory / POST_DIRECTORY
        post.mkdir(exist_ok=True)
        postoptim = self.directory / POSTOPTIM_DIRECTORY
        postoptim.mkdir(exist_ok=True)
        poststudy = self.directory / POSTSTUDY_DIRECTORY
        poststudy.mkdir(exist_ok=True)
        postscal = self.directory / POSTSCAL_DIRECTORY
        postscal.mkdir(exist_ok=True)
        results = self.directory / RESULTS_DIRECTORY
        results.mkdir(exist_ok=True)
        msg = MultiLineString()
        msg.indent()
        msg.add("Create directories")
        msg.indent()
        msg.add("Working directory: {}", self.directory)
        msg.add("Post-processing: {}", post)
        msg.add("Optimization history view: {}", postoptim)
        msg.add("Scalability views: {}", poststudy)
        msg.add("Dependency matrices: {}", postscal)
        msg.add("Results: {}", results)
        LOGGER.info("%s", msg)

    def add_discipline(self, data: IODataset) -> None:
        """This method adds a disciplinary dataset from a dataset.

        Args:
            data: The input-output dataset.
        """
        self._group_dep[data.name] = {}
        self._all_data = data.get_view().to_numpy()
        self.datasets.append(data)
        for output_name in data.get_variable_names(data.OUTPUT_GROUP):
            self.set_fill_factor(data.name, output_name, self._default_fill_factor)
        inputs = ", ".join([
            f"{name}({data.variable_names_to_n_components[name]})"
            for name in data.get_variable_names(data.INPUT_GROUP)
        ])
        outputs = ", ".join([
            f"{name}({data.variable_names_to_n_components[name]})"
            for name in data.get_variable_names(data.OUTPUT_GROUP)
        ])
        msg = MultiLineString()
        msg.add("Add scalable discipline # {}", len(self.datasets))
        msg.indent()
        msg.add("Name: {}", data.name)
        msg.add("Inputs: {}", inputs)
        msg.add("Outputs: {}", outputs)
        msg.add("Built from {}", len(data))
        LOGGER.info("%s", msg)

    @property
    def discipline_names(self) -> list[str]:
        """The names of the disciplines."""
        return [discipline.name for discipline in self.datasets]

    def set_input_output_dependency(
        self, discipline: str, output: str, inputs: Iterable[str]
    ) -> None:
        """Set the dependency between an output and inputs for a given discipline.

        Args:
            discipline: The name of the discipline.
            output: The name of the output.
            inputs: The names of the inputs.
        """
        self.__check_discipline(discipline)
        self.__check_output(discipline, output)
        self.__check_inputs(discipline, inputs)
        self._group_dep[discipline][output] = inputs

    def set_fill_factor(self, discipline: str, output: str, fill_factor: float) -> None:
        """Set the fill factor.

        Args:
        Args:
            discipline: The name of the discipline.
            output: The name of the output.
            fill_factor: The fill factor
        """
        self.__check_discipline(discipline)
        self.__check_output(discipline, output)
        self.__check_fill_factor(fill_factor)
        if discipline not in self._fill_factor:
            self._fill_factor[discipline] = {}
        self._fill_factor[discipline][output] = fill_factor

    def __check_discipline(self, discipline: str) -> None:
        """Check if discipline is a string comprised in the list of disciplines names.

        Args:
            discipline: The name of the discipline.

        Raises:
            TypeError: When the discipline is not a string.
            ValueError: when the discipline is not available.
        """
        if not isinstance(discipline, str):
            msg = "The argument discipline should be a string"
            raise TypeError(msg)
        discipline_names = self.discipline_names
        if discipline not in discipline_names:
            msg = "The argument discipline should be a string comprised in the list %s"
            raise ValueError(
                msg,
                discipline_names,
            )

    def __check_output(self, discipline: str, varname: str) -> None:
        """Check if a variable is an output of a given discipline.

        Args:
            discipline: The name of the discipline.
            varname: The name of the variable.

        Raises:
            TypeError: When the output is not a string.
            ValueError: When the output is not available.
        """
        self.__check_discipline(discipline)
        if not isinstance(varname, str):
            msg = f"{varname} is not a string."
            raise TypeError(msg)
        output_names = next(
            dataset.get_variable_names(dataset.OUTPUT_GROUP)
            for dataset in self.datasets
            if dataset.name == discipline
        )
        if varname not in output_names:
            msg = (
                f"'{varname}' is not an output of {discipline}; "
                f"available outputs are: {output_names}"
            )
            raise ValueError(msg)

    def __check_inputs(self, discipline: str, inputs: list[str]) -> None:
        """Check if inputs is a list of inputs of discipline.

        Args:
            discipline: The name of the discipline.
            inputs: The names of the inputs.

        Raises:
            TypeError: When an input is not a string.
            ValueError: When an input is not available.
        """
        self.__check_discipline(discipline)
        if not isinstance(inputs, list):
            msg = "The argument 'inputs' must be a list of string."
            raise TypeError(msg)
        input_names = next(
            dataset.get_variable_names(dataset.INPUT_GROUP)
            for dataset in self.datasets
            if dataset.name == discipline
        )
        for inpt in inputs:
            if not isinstance(inpt, str):
                msg = f"{inpt} is not a string."
                raise TypeError(msg)
            if inpt not in input_names:
                msg = (
                    f"'{inpt}' is not a discipline input; available inputs are: "
                    f"{input_names}"
                )
                raise ValueError(msg)

    def __check_fill_factor(self, fill_factor: float) -> None:
        """Check if fill factor is a proportion or a number equal to -1.

        Args:
            fill_factor: Either a proportion or -1.

        Raises:
            TypeError: When the fill factor is neither a proportion nor -1.
        """
        try:
            self.__check_proportion(fill_factor)
        except ValueError:
            if fill_factor != -1:
                msg = (
                    "Fill factor should be a float number comprised in 0 and 1 "
                    "or a number equal to -1."
                )
                raise TypeError(msg) from None

    @staticmethod
    def __check_proportion(proportion: float) -> None:
        """Check if a proportion is a float number comprised in [0, 1].

        Args:
            proportion: A proportion comprised in [0, 1].

        Raises:
            TypeError: When the proportion is not a number.
            ValueError: When the proportion is not a number comprised in [0, 1].
        """
        if not isinstance(proportion, numbers.Number):
            msg = "A proportion should be a float number comprised in 0 and 1."
            raise TypeError(msg)
        if not 0 <= proportion <= 1:
            msg = "A proportion should be a float number comprised in 0 and 1."
            raise ValueError(msg)

    def add_optimization_strategy(
        self,
        algo: str,
        max_iter: int,
        formulation: str = "DisciplinaryOpt",
        algo_options: StrKeyMapping | None = None,
        formulation_options: str | None = None,
        top_level_diff: str = "auto",
    ) -> None:
        """Add both optimization algorithm and MDO formulation and their options.

        Args:
            algo: The name of the optimization algorithm.
            max_iter: The maximum number of iterations for the optimization algorithm.
            formulation: The name of the MDO formulation.
            algo_options: The options of the optimization algorithm.
            formulation_options: The options of the MDO formulation.
            top_level_diff: The differentiation method for the top level disciplines.
        """
        self.algorithms.append(algo)
        if algo_options is None:
            algo_options = {}
        elif not isinstance(algo_options, dict):
            msg = "algo_options must be a dictionary."
            raise TypeError(msg)
        algo_options.update({"max_iter": max_iter})
        self.algorithms_options.append(algo_options)
        self.formulations.append(formulation)
        self.formulations_options.append(formulation_options)
        self.top_level_diff.append(top_level_diff)
        if algo_options is not None:
            algo_options = ", ".join([
                f"{name}({value})" for name, value in algo_options.items()
            ])
        if formulation_options is not None:
            formulation_options = ", ".join([
                f"{name}({value})" for name, value in formulation_options.items()
            ])
        msg = MultiLineString()
        msg.add("Add optimization strategy # {}", len(self.formulations))
        msg.indent()
        msg.add("Algorithm: {}", algo)
        msg.add("Algorithm options: {}", algo_options)
        msg.add("Formulation: {}", formulation)
        msg.add("Formulation options: {}", formulation_options)
        LOGGER.info("%s", msg)

    def add_scaling_strategies(
        self,
        design_size: int | list[int] | None = None,
        coupling_size: int | list[int] | None = None,
        eq_cstr_size: int | list[int] | None = None,
        ineq_cstr_size: int | list[int] | None = None,
        variables: list[None] | None = None,
    ) -> None:
        """Add different scaling strategies.

        Args:
            design_size: The size of the design variables.
                If ``None``, use 1.
            coupling_size: The size of the coupling variables.
                If ``None``, use 1.
            eq_cstr_size: The size of the equality constraints.
                If ``None``, use 1.
            ineq_cstr_size: The size of the inequality constraints.
                If ``None``, use 1.
            variables: The size of the other variables.
        """
        n_design = self.__check_varsizes_type(design_size)
        n_coupling = self.__check_varsizes_type(coupling_size)
        n_eq = self.__check_varsizes_type(eq_cstr_size)
        n_ineq = self.__check_varsizes_type(ineq_cstr_size)
        n_var = self.__check_varsizes_type(variables)
        n_scaling = max(n_design, n_coupling, n_eq, n_ineq, n_var)
        self.__check_scaling_consistency(n_design, n_scaling)
        self.__check_scaling_consistency(n_coupling, n_scaling)
        self.__check_scaling_consistency(n_eq, n_scaling)
        self.__check_scaling_consistency(n_ineq, n_scaling)
        design_size = self.__format_scaling(design_size, n_scaling)
        coupling_size = self.__format_scaling(coupling_size, n_scaling)
        eq_cstr_size = self.__format_scaling(eq_cstr_size, n_scaling)
        ineq_cstr_size = self.__format_scaling(ineq_cstr_size, n_scaling)
        for idx in range(n_scaling):
            var_scaling = {}
            self.__update_var_scaling(
                var_scaling, design_size[idx], self.design_variables
            )
            self.__update_var_scaling(
                var_scaling, coupling_size[idx], self.coupling_variables
            )
            self.__update_var_scaling(
                var_scaling, eq_cstr_size[idx], self.eq_constraints
            )
            self.__update_var_scaling(
                var_scaling, ineq_cstr_size[idx], self.ineq_constraints
            )
            scaling = {
                "design_variables": design_size[idx],
                "coupling_variables": coupling_size[idx],
                "eq_constraint_size": eq_cstr_size[idx],
                "ineq_constraint_size": ineq_cstr_size[idx],
            }
            if variables is not None and variables[0] is not None:
                for varname, value in variables[idx].items():
                    self.__update_var_scaling(var_scaling, value, [varname])
                    scaling[varname] = value
            else:
                variables = [None] * n_scaling
            self.var_scalings.append(var_scaling)
            self.scalings.append(scaling)
        msg = MultiLineString()
        msg.add("Add scaling strategies")
        msg.indent()
        msg.add("Number of strategies: {}", n_scaling)
        for idx in range(n_scaling):
            if variables[idx] is not None:
                var_str = ", ".join([
                    f"{name}({size})" for name, size in variables[idx].items()
                ])
            else:
                var_str = None
            msg.add("Strategy # {}", idx + 1)
            msg.indent()
            msg.add("Design variables: {}", design_size[idx])
            msg.add("Coupling variables: {}", coupling_size[idx])
            msg.add("Equality constraints: {}", eq_cstr_size[idx])
            msg.add("Inequality constraints: {}", ineq_cstr_size[idx])
            msg.add("Variables: {}", var_str)
            msg.dedent()
        LOGGER.info("%s", msg)

    @staticmethod
    def __format_scaling(size: int | list[int], n_scaling: int) -> list[int]:
        """Convert a scaling size in a list of integers.

        The length of this list is equal to the number of scalings.

        Args:
            size: The size(s) of a given variable
            n_scaling: The number of scalings.

        Returns:
            A size per scaling.
        """
        formatted_sizes = size
        if isinstance(formatted_sizes, int) or formatted_sizes is None:
            formatted_sizes = [formatted_sizes]
        if len(formatted_sizes) == 1:
            formatted_sizes *= n_scaling
        return formatted_sizes

    @staticmethod
    def __update_var_scaling(
        scaling: dict[str, StrKeyMapping], size: int, varnames: Sequence[str]
    ) -> None:
        """Update a scaling dictionary for a given list of variables and a given size.

        Args:
            scaling: The variable names bound to the scaling properties,
                e.g. {'size': val}.
            size: The size of the variable.
            varnames: The names of the variables.
        """
        if size is not None:
            scaling.update(dict.fromkeys(varnames, size))

    @staticmethod
    def __check_scaling_consistency(n_var_scaling: int, n_scaling: int) -> None:
        """Check the scaling consistency.

        For the different types of variables,
        the number of scalings shall be the same or equal to 1.

        Args:
            n_var_scaling: The number of scalings.
            n_scaling: The expected number of scalings.
        """
        assert n_var_scaling in {n_scaling, 1}

    @staticmethod
    def __check_varsizes_type(varsizes: Sequence[int]) -> int:
        """Check the type of scaling sizes.

        Integer, list of integers or None is expected.
        Return the number of scalings.

        Args:
            varsizes: The sizes of the variables.

        Returns:
            The number of scalings.
        """
        length = 1
        if varsizes is not None:
            if isinstance(varsizes, list):
                for size in varsizes:
                    if isinstance(size, dict):
                        for value in size.values():
                            assert isinstance(value, int)
                    else:
                        assert isinstance(size, int)
                length = len(varsizes)
            else:
                assert isinstance(varsizes, int)
                length = 1
        return length

    def execute(self, n_replicates: int = 1) -> list[ScalabilityResult]:
        """Execute the scalability study.

        Args:
            n_replicates: The number of times the scalability study is repeated
                to study the variability.
        """
        plural = "s" if n_replicates > 1 else ""
        LOGGER.info("Execute scalability study %s time%s", n_replicates, plural)
        if not self.formulations and not self.algorithms:
            msg = (
                "A scalable study needs at least 1 optimization strategy, "
                "defined by a mandatory optimization algorithm "
                "and optional optimization algorithm and options"
            )
            raise ValueError(msg)
        counter = "Formulation: {} - Algo: {} - Scaling: {}/{} - Replicate: {}/{}"
        n_scal_strategies = len(self.var_scalings)
        n_opt_strategies = len(self.algorithms)
        for opt_index in range(n_opt_strategies):
            algo = self.algorithms[opt_index]
            formulation = self.formulations[opt_index]
            for scal_index in range(n_scal_strategies):
                scaling = self.var_scalings[scal_index]
                for replicate in range(1, n_replicates + 1):
                    msg = MultiLineString()
                    msg.indent()
                    data = (
                        formulation,
                        algo,
                        scal_index + 1,
                        n_scal_strategies,
                        replicate,
                        n_replicates,
                    )
                    msg.add(counter, *data)
                    LOGGER.info("%s", msg)
                    msg = MultiLineString()
                    msg.indent()
                    msg.indent()
                    msg.add("Create scalable problem")
                    problem = self.__create_scalable_problem(scaling, replicate)
                    path = self.__dep_mat_path(algo, formulation, scal_index, replicate)
                    directory = path.stem
                    path.mkdir(exist_ok=True)
                    msg.add("Save dependency matrices in {}", path)
                    problem.plot_dependencies(True, False, str(path))
                    msg.add("Create MDO Scenario")
                    with LoggingContext(LOGGING_SETTINGS.logger):
                        self.__create_scenario(problem, formulation, opt_index)
                        msg.add("Execute MDO Scenario")
                        formulation_options = self.formulations_options[opt_index]
                        algo_options = self.__execute_scenario(problem, algo, opt_index)

                    path = self.__optview_path(algo, formulation, scal_index, replicate)
                    msg.add("Save optim history view in {}", path)
                    fpath = str(path) + "/"
                    problem.scenario.post_process(
                        "OptHistoryView", save=True, show=False, file_path=fpath
                    )
                    result = ScalabilityResult(directory, scal_index + 1, replicate)
                    self.results.append(result)
                    statistics = self.__get_statistics(problem, scaling)
                    result.get(
                        algo=algo,
                        algo_options=algo_options,
                        formulation=formulation,
                        formulation_options=formulation_options,
                        scaling=scaling,
                        disc_names=problem.disciplines,
                        output_names=problem.outputs,
                        **statistics,
                    )
                    fpath = result.get_file_path(self.directory)
                    msg.add("Save statistics in {}", fpath)
                    result.to_pickle(str(self.directory))
                    LOGGER.debug("%s", msg)
        return self.results

    def __get_statistics(
        self, problem: ScalableProblem, scaling: Mapping[str, int]
    ) -> dict[str, int | bool | dict[str, int] | dict[str, dict[str, int]]]:
        """Get statistics from an executed scalable problem.

        Args:
            problem: The scalable problem.
            scaling: The variables scaling.
        """
        statistics = {}
        stopidx, n_iter = self.__get_stop_index(problem)
        ratio = float(stopidx) / n_iter
        n_calls = {disc: n_calls * ratio for disc, n_calls in problem.n_calls.items()}
        statistics["n_calls"] = n_calls
        tmp = problem.n_calls_linearize
        n_calls_linearize = {disc: ncl * ratio for disc, ncl in tmp.items()}
        statistics["n_calls_linearize"] = n_calls_linearize
        tmp = problem.n_calls_top_level
        n_calls_tl = {disc: n_calls * ratio for disc, n_calls in tmp.items()}
        statistics["n_calls_top_level"] = n_calls_tl
        tmp = problem.n_calls_linearize_top_level
        n_calls_linearize_tl = {disc: ncltl * ratio for disc, ncltl in tmp.items()}
        statistics["n_calls_linearize_top_level"] = n_calls_linearize_tl
        statistics["exec_time"] = problem.exec_time() * ratio
        statistics["status"] = problem.status
        statistics["is_feasible"] = problem.is_feasible

        inputs = problem.inputs
        outputs = problem.outputs
        disc_varnames = {
            disc: inputs[disc] + outputs[disc] for disc in problem.disciplines
        }
        sizes = problem.varsizes
        statistics["new_varsizes"] = {
            disc: {name: scaling.get(name, sizes[name]) for name in disc_varnames[disc]}
            for disc in problem.disciplines
        }
        statistics["old_varsizes"] = problem.varsizes
        return statistics

    def __dep_mat_path(
        self, algo: str, formulation: str, id_scaling: int, replicate: int
    ) -> Path:
        """Define the path to the directory containing the dependency matrices files.

        Args:
            algo: The name of the algorithm.
            formulation: The name of the formulation.
            id_scaling: The index of the scaling.
            replicate: The replicate number.
        """
        varnames = [algo, formulation, id_scaling + 1, replicate]
        name = "_".join([self.prefix] + [str(var) for var in varnames])
        if name[0] == "_":
            name = name[1:]
        return self.directory / POSTSCAL_DIRECTORY / name

    def __optview_path(
        self, algo: str, formulation: str, id_scaling: int, replicate: int
    ) -> Path:
        """Define the path to the directory containing the dependency matrices files.

        Args:
            algo: The name of the algorithm.
            formulation: The name of the formulation.
            id_scaling: The index of the scaling.
            replicate: The replicate number.
        """
        path = (
            self.directory
            / POSTOPTIM_DIRECTORY
            / Path(f"{self.prefix}_{algo}_{formulation}")
            / Path(f"scaling_{id_scaling + 1}")
            / Path(f"replicate_{replicate}")
        )
        path.mkdir(exist_ok=True, parents=True)
        return path

    def __create_scalable_problem(self, scaling: dict, seed: int) -> ScalableProblem:
        """Create a scalable problem.

        Args:
            scaling: The scaling.
            seed: The seed for random features.

        Returns:
            The scalable problem.
        """
        return ScalableProblem(
            self.datasets,
            self.design_variables,
            self.objective,
            self.eq_constraints,
            self.ineq_constraints,
            self.maximize_objective,
            scaling,
            fill_factor=self._fill_factor,
            seed=seed,
            group_dep=self._group_dep,
            force_input_dependency=True,
            allow_unused_inputs=False,
        )

    def __create_scenario(
        self, problem: ScalableProblem, formulation: str, opt_index: int
    ) -> None:
        """Create scenario for a given formulation.

        Args:
            problem: The scalable problem.
            formulation: The name of the formulation.
            opt_index: The optimization strategy index.
        """
        form_opt = self.formulations_options[opt_index]
        formulation_options = {} if not isinstance(form_opt, dict) else form_opt
        problem.create_scenario(
            formulation,
            "MDO",
            self.start_at_equilibrium,
            self.active_probability,
            self.feasibility_level,
            **formulation_options,
        )

    def __execute_scenario(
        self, problem: ScalableProblem, algo: str, opt_index: int
    ) -> None:
        """Execute scenario.

        Args:
            problem: The scalable problem.
            algo: The name of the optimization algorithm.
            opt_index: The optimization strategy index.
        """
        top_level_disciplines = problem.scenario.formulation.get_top_level_disc()
        for disc in top_level_disciplines:
            disc.linearization_mode = self.top_level_diff[opt_index]
        algo_options = deepcopy(self.algorithms_options[opt_index])
        max_iter = algo_options["max_iter"]
        del algo_options["max_iter"]
        problem.scenario.execute({
            "algo": algo,
            "max_iter": max_iter,
            "algo_options": algo_options,
        })
        return algo_options

    def __get_stop_index(self, problem: ScalableProblem) -> tuple[int, int]:
        """Get stop index from a database.

        Args:
            problem: The scalable problem.

        Returns:
            The stop index, the database length.
        """
        database = problem.scenario.formulation.optimization_problem.database
        n_iter = len(database)
        if self.early_stopping:
            y_prev = inf
            stopidx = 0
            for value in database.values():
                pbm = problem.scenario.formulation.optimization_problem
                if y_prev == inf:
                    diff = inf
                else:
                    diff = abs(y_prev - value[pbm.standardized_objective_name])
                    diff /= abs(y_prev)
                if diff < 1e-6:
                    break
                y_prev = value[pbm.standardized_objective_name]
                stopidx += 1  # noqa: SIM113
        else:
            stopidx = n_iter
        return stopidx, n_iter
