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
#                         documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""The base class for all formulations."""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from numpy import arange
from numpy import copy
from numpy import empty
from numpy import ndarray
from numpy import zeros

from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.core.mdo_functions.function_from_discipline import FunctionFromDiscipline
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.core.mdo_functions.taylor_polynomials import compute_linear_approximation
from gemseo.disciplines.utils import check_disciplines_consistency
from gemseo.scenarios.scenario_results.scenario_result import ScenarioResult
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from gemseo.algos.design_space import DesignSpace
    from gemseo.core.discipline import Discipline
    from gemseo.core.discipline.base_discipline import BaseDiscipline
    from gemseo.core.grammars.json_grammar import JSONGrammar
    from gemseo.scenarios.scenario import Scenario
    from gemseo.typing import StrKeyMapping


LOGGER = logging.getLogger(__name__)


class BaseFormulation(metaclass=ABCGoogleDocstringInheritanceMeta):
    """Base MDO formulation class to be extended in subclasses for use.

    This class creates the :class:`.MDOFunction` instances
    computing the constraints, objective and observables
    from the disciplines
    and add them to the attached :attr:`.optimization_problem`.

    It defines the multidisciplinary process, i.e. dataflow and workflow, implicitly.

    By default,

    - the objective is minimized,
    - the type of a constraint is equality,
    - the activation value of a constraint is 0.

    The link between the instances of :class:`.Discipline`,
    the design variables and
    the names of the discipline outputs used as constraints, objective and observables
    is made with the :class:`.DisciplineAdapterGenerator`,
    which generates instances of :class:`.MDOFunction` from the disciplines.
    """

    DEFAULT_SCENARIO_RESULT_CLASS_NAME: ClassVar[str] = ScenarioResult.__name__
    """The name of the :class:`.ScenarioResult` class to be used for post-processing."""

    optimization_problem: OptimizationProblem
    """The optimization problem generated by the formulation from the disciplines."""

    __differentiated_input_names_substitute: tuple[str, ...]
    """The names of the inputs against which to differentiate the functions.

    If empty, consider the variables of their input space.
    """

    _objective_name: str | Sequence[str]
    """The name(s) of the discipline output(s) used as objective."""

    _maximize_objective: bool
    """Whether to maximize the objective."""

    variable_sizes: dict[str, int]
    """The sizes of the design variables and differentiated inputs substitutes."""

    __disciplines: tuple[BaseDiscipline, ...]
    """The disciplines."""

    def __init__(
        self,
        disciplines: Iterable[Discipline],
        objective_name: str | Sequence[str],
        design_space: DesignSpace,
        maximize_objective: bool = False,
        differentiated_input_names_substitute: Iterable[str] = (),
        **options: Any,
    ) -> None:
        r"""
        Args:
            disciplines: The disciplines.
            objective_name: The name(s) of the discipline output(s) used as objective.
                If multiple names are passed, the objective will be a vector.
            design_space: The design space.
            maximize_objective: Whether to maximize the objective.
            differentiated_input_names_substitute: The names of the discipline inputs
                against which to differentiate the discipline outputs
                used as objective, constraints and observables.
                If empty, consider the inputs of these functions.
                More precisely,
                for each function,
                an :class:`.MDOFunction` is built from the ``disciplines``,
                which depend on input variables :math:`x_1,\ldots,x_d,x_{d+1}`,
                and over an input space
                spanned by the input variables :math:`x_1,\ldots,x_d`
                and depending on both the MDO formulation and the ``design_space``.
                Then,
                the methods :meth:`.MDOFunction.evaluate` and :meth:`.MDOFunction.jac`
                are called at a given point of the input space
                and return the output value and the Jacobian matrix,
                i.e. the matrix concatenating the partial derivatives
                with respect to the inputs :math:`x_1,\ldots,x_d`
                at this point of the input space.
                This argument can be used to compute the matrix
                concatenating the partial derivatives
                at the same point of the input space
                but with respect to custom inputs,
                e.g. :math:`x_{d-1}` and :math:`x_{d+1}`.
                Mathematically speaking,
                this matrix returned by :meth:`.MDOFunction.jac`
                is no longer a Jacobian.
            **options: The options of the formulation.
        """  # noqa: D205, D212, D415
        self.__disciplines = tuple(disciplines)
        self.__check_disciplines()
        self.__differentiated_input_names_substitute = tuple(
            differentiated_input_names_substitute
        )
        self._objective_name = objective_name
        self.optimization_problem = OptimizationProblem(design_space)
        self._maximize_objective = maximize_objective
        self.variable_sizes = design_space.variable_sizes.copy()

    @property
    def disciplines(self) -> tuple[BaseDiscipline, ...]:
        """The disciplines."""
        return self.__disciplines

    @property
    def differentiated_input_names_substitute(self) -> tuple[str, ...]:
        """The names of the inputs against which to differentiate the functions.

        If empty, consider the variables of their input space.
        """
        return self.__differentiated_input_names_substitute

    def __check_disciplines(self) -> None:
        """Check that two disciplines do not compute the same output."""
        disciplines = set(self.disciplines).difference(self.get_sub_scenarios())
        if disciplines:
            check_disciplines_consistency(disciplines, False, True)

    @property
    def design_space(self) -> DesignSpace:
        """The design space on which the formulation is applied."""
        return self.optimization_problem.design_space

    @abstractmethod
    def add_constraint(
        self,
        output_name: str | Sequence[str],
        constraint_type: MDOFunction.ConstraintType = MDOFunction.ConstraintType.EQ,
        constraint_name: str = "",
        value: float = 0,
        positive: bool = False,
    ) -> None:
        r"""Add an equality or inequality constraint to the optimization problem.

        An equality constraint is written as :math:`c(x)=a`,
        a positive inequality constraint is written as :math:`c(x)\geq a`
        and a negative inequality constraint is written as :math:`c(x)\leq a`.

        This constraint is in addition to those created by the formulation,
        e.g. consistency constraints in IDF.

        The strategy of repartition of the constraints is defined by the formulation.

        Args:
            output_name: The name(s) of the outputs computed by :math:`c(x)`.
                If several names are given,
                a single discipline must provide all outputs.
            constraint_type: The type of constraint.
            constraint_name: The name of the constraint to be stored.
                If empty,
                the name of the constraint is generated
                from ``output_name``, ``constraint_type``, ``value`` and ``positive``.
            value: The value :math:`a`.
            positive: Whether the inequality constraint is positive.
        """

    @abstractmethod
    def add_observable(
        self,
        output_names: str | Sequence[str],
        observable_name: str = "",
        discipline: Discipline | None = None,
    ) -> None:
        """Add an observable to the optimization problem.

        The repartition strategy of the observable is defined in the formulation class.

        Args:
            output_names: The name(s) of the output(s) to observe.
            observable_name: The name of the observable.
                If empty, the output name is used by default.
            discipline: The discipline computing the observed outputs.
                If ``None``, the discipline is detected from inner disciplines.
        """

    @abstractmethod
    def get_top_level_disciplines(self) -> tuple[BaseDiscipline, ...]:
        """Return the disciplines which inputs are required to run the scenario.

        A formulation seeks to
        compute the objective and constraints from the input variables.
        It structures the optimization problem into multiple levels of disciplines.
        The disciplines directly depending on these inputs
        are called top level disciplines.

        By default, this method returns all disciplines.
        This method can be overloaded by subclasses.

        Returns:
            The top level disciplines.
        """

    def _get_dv_indices(
        self,
        names: Iterable[str],
    ) -> dict[str, tuple[int, int, int]]:
        """Return the indices associated with specific variables.

        Args:
            names: The names of the variables.

        Returns:
            For each variable,
            a 3-length tuple
            whose first dimensions are its first and last indices in the design space
            and last dimension is its size.
        """
        start = end = 0
        sizes = self.variable_sizes
        names_to_indices = {}
        for name in names:
            size = sizes[name]
            end += size
            names_to_indices[name] = (start, end, size)
            start = end

        return names_to_indices

    def unmask_x_swap_order(
        self,
        masking_data_names: Iterable[str],
        x_masked: ndarray,
        all_data_names: Iterable[str] = (),
        x_full: ndarray | None = None,
    ) -> ndarray:
        """Unmask a vector or matrix from names, with respect to other names.

        This method eventually swaps the order of the values
        if the order of the data names is inconsistent between these sets.

        Args:
            masking_data_names: The names of the kept data.
            x_masked: The vector or matrix to unmask.
            all_data_names: The set of all names.
                If empty, use the design variables stored in the design space.
            x_full: The default values for the full vector or matrix.
                If ``None``, use the zero vector or matrix.

        Returns:
            The vector or matrix related to the input mask.

        Raises:
            IndexError: when the sizes of variables are inconsistent.
        """
        if not all_data_names:
            all_data_names = self.get_optim_variable_names()
        indices = self._get_dv_indices(all_data_names)
        variable_sizes = self.variable_sizes
        total_size = sum(variable_sizes[var] for var in all_data_names)

        # TODO: The support of sparse Jacobians requires modifications here.
        if x_full is None:
            x_unmask = zeros((*x_masked.shape[:-1], total_size), dtype=x_masked.dtype)
        else:
            x_unmask = copy(x_full)

        i_x = 0
        try:
            for key in all_data_names:
                if key in masking_data_names:
                    i_min, i_max, n_x = indices[key]
                    x_unmask[..., i_min:i_max] = x_masked[..., i_x : i_x + n_x]
                    i_x += n_x
        except IndexError:
            msg = (
                "Inconsistent input array size of values array "
                f"with reference data shape {x_unmask.shape}"
            )
            raise ValueError(msg) from None
        return x_unmask

    def mask_x_swap_order(
        self,
        masking_data_names: Iterable[str],
        x_vect: ndarray,
        all_data_names: Iterable[str] = (),
    ) -> ndarray:
        """Mask a vector from a subset of names, with respect to a set of names.

        This method eventually swaps the order of the values
        if the order of the data names is inconsistent between these sets.

        Args:
            masking_data_names: The names of the kept data.
            x_vect: The vector to mask.
            all_data_names: The set of all names.
                If empty, use the design variables stored in the design space.

        Returns:
            The masked version of the input vector.

        Raises:
            IndexError: when the sizes of variables are inconsistent.
        """
        x_mask = self.get_x_mask_x_swap_order(masking_data_names, all_data_names)
        return x_vect[x_mask]

    def get_x_mask_x_swap_order(
        self,
        masking_data_names: Iterable[str],
        all_data_names: Iterable[str] = (),
    ) -> ndarray:
        """Mask a vector from a subset of names, with respect to a set of names.

        This method eventually swaps the order of the values
        if the order of the data names is inconsistent between these sets.

        Args:
            masking_data_names: The names of the kept data.
            all_data_names: The set of all names.
                If empty, use the design variables stored in the design space.

        Returns:
            The masked version of the input vector.

        Raises:
            ValueError: If the sizes or the sizes of variables are inconsistent.
        """
        design_space = self.optimization_problem.design_space
        if not all_data_names:
            all_data_names = design_space

        variable_sizes = {var: design_space.get_size(var) for var in design_space}
        total_size = sum(variable_sizes[var] for var in masking_data_names)
        indices = self._get_dv_indices(all_data_names)
        x_mask = empty(total_size, dtype="int")
        i_masked_min = i_masked_max = 0
        try:
            for key in masking_data_names:
                i_min, i_max, loc_size = indices[key]
                i_masked_max += loc_size
                x_mask[i_masked_min:i_masked_max] = arange(i_min, i_max)
                i_masked_min = i_masked_max
        except KeyError as err:
            msg = (
                "Inconsistent inputs of masking. "
                f"Key {err} is in masking_data_names {masking_data_names} "
                f"but not in provided all_data_names : {all_data_names}!"
            )
            raise ValueError(msg) from None

        return x_mask

    def _remove_unused_variables(self) -> None:
        """Remove variables in the design space that are not discipline inputs."""
        design_space = self.optimization_problem.design_space
        disciplines = self.get_top_level_disciplines()
        all_inputs = {
            var for disc in disciplines for var in disc.io.input_grammar.names
        }
        for name in design_space.variable_names:
            if name not in all_inputs:
                design_space.remove_variable(name)
                LOGGER.info(
                    "Variable %s was removed from the Design Space, it is not an input"
                    " of any discipline.",
                    name,
                )

    def _remove_sub_scenario_dv_from_ds(self) -> None:
        """Remove the sub scenarios design variables from the design space."""
        for scenario in self.get_sub_scenarios():
            for var in scenario.design_space:
                if var in self.design_space:
                    self.design_space.remove_variable(var)

    def _build_objective_from_disc(
        self,
        objective_name: str | Sequence[str],
        discipline: Discipline | None = None,
        top_level_disc: bool = True,
    ) -> None:
        """Build the objective function from the discipline able to compute it.

        Args:
            objective_name: The name(s) of the discipline output(s) used as objective.
                If multiple names are passed, the objective will be a vector.
            discipline: The discipline computing the objective.
                If ``None``, the discipline is detected from the inner disciplines.
            top_level_disc: Whether to search the discipline among the top level ones.
        """
        if isinstance(objective_name, str):
            objective_name = [objective_name]
        obj_mdo_fun = FunctionFromDiscipline(
            objective_name, self, discipline=discipline, top_level_disc=top_level_disc
        )
        if obj_mdo_fun.discipline_adapter.is_linear:
            obj_mdo_fun = compute_linear_approximation(
                obj_mdo_fun, zeros(obj_mdo_fun.discipline_adapter.input_dimension)
            )

        self.optimization_problem.objective = obj_mdo_fun
        if self._maximize_objective:
            self.optimization_problem.minimize_objective = False

    def get_optim_variable_names(self) -> list[str]:
        """Get the optimization unknown names to be provided to the optimizer.

        This is different from the design variable names provided by the user,
        since it depends on the formulation,
        and can include target values for coupling for instance in IDF.

        Returns:
            The optimization variable names.
        """
        return self.optimization_problem.design_space.variable_names

    def get_x_names_of_disc(
        self,
        discipline: Discipline,
    ) -> list[str]:
        """Get the design variables names of a given discipline.

        Args:
            discipline: The discipline.

        Returns:
             The names of the design variables.
        """
        optim_variable_names = self.get_optim_variable_names()
        input_names = discipline.io.input_grammar.names
        return [name for name in optim_variable_names if name in input_names]

    def get_sub_scenarios(self) -> list[Scenario]:
        """List the disciplines that are actually scenarios.

        Returns:
            The scenarios.
        """
        from gemseo.scenarios.scenario import Scenario

        return [disc for disc in self.disciplines if isinstance(disc, Scenario)]

    def _set_default_input_values_from_design_space(self) -> None:
        """Initialize the top level disciplines from the design space."""
        if not self.optimization_problem.design_space.has_current_value:
            return

        current_x = self.optimization_problem.design_space.get_current_value(
            as_dict=True
        )

        for discipline in self.get_top_level_disciplines():
            input_names = discipline.io.input_grammar.names
            to_value = discipline.input_grammar.data_converter.convert_array_to_value
            discipline.default_input_data.update({
                name: to_value(name, value)
                for name, value in current_x.items()
                if name in input_names
            })

    @classmethod
    def get_default_sub_option_values(cls, **options: str) -> StrKeyMapping:
        """Return the default values of the sub-options of the formulation.

        When some options of the formulation depend on higher level options,
        the default values of these sub-options may be obtained here,
        mainly for use in the API.

        Args:
            **options: The options required to deduce the sub-options grammar.

        Returns:
            Either ``None`` or the sub-options default values.
        """
        return {}

    @classmethod
    def get_sub_options_grammar(cls, **options: str) -> JSONGrammar:
        """Get the sub-options grammar.

        When some options of the formulation depend on higher level options,
        the schema of the sub-options may be obtained here,
        mainly for use in the API.

        Args:
            **options: The options required to deduce the sub-options grammar.

        Returns:
            Either ``None`` or the sub-options grammar.
        """
        return {}
