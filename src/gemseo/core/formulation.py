# -*- coding: utf-8 -*-
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

from __future__ import division, unicode_literals

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from gemseo.algos.design_space import DesignSpace
from gemseo.core.discipline import MDODiscipline
from gemseo.core.execution_sequence import ExecutionSequence
from gemseo.core.json_grammar import JSONGrammar

if TYPE_CHECKING:
    from gemseo.core.scenario import Scenario

import six
from custom_inherit import DocInheritMeta
from numpy import array, copy, in1d, ndarray, where, zeros
from six import string_types

from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.core.mdofunctions.function_from_discipline import FunctionFromDiscipline
from gemseo.core.mdofunctions.function_generator import MDOFunctionGenerator
from gemseo.core.mdofunctions.mdo_function import MDOFunction

LOGGER = logging.getLogger(__name__)


@six.add_metaclass(
    DocInheritMeta(
        abstract_base_class=True,
        style="google_with_merge",
        include_special_methods=True,
    )
)
class MDOFormulation(object):
    """Abstract MDO formulation class to be extended in subclasses for use.

    This class creates the objective function and constraints from the disciplines.
    It defines the process implicitly.

    By default,

    - the objective function is minimized,
    - the type of a constraint is equality,
    - the activation value of a constraint is 0.

    The link between the instances of :class:`.MDODiscipline`,
    the name of the objective function
    and the names of the constraints
    is made with the :class:`.MDOFunctionGenerator`,
    which generates instances of :class:`.MDOFunction` from the disciplines.
    """

    NAME = "MDOFormulation"

    def __init__(
        self,
        disciplines,  # type: Sequence[MDODiscipline]
        objective_name,  # type: str
        design_space,  # type: DesignSpace
        maximize_objective=False,  # type: bool
        grammar_type=MDODiscipline.JSON_GRAMMAR_TYPE,  # type: str
        **options  # type: Any
    ):  # type: (...) -> None # pylint: disable=W0613
        """
        Args:
            disciplines: The disciplines.
            objective_name: The name of the objective function.
            design_space: The design space.
            maximize_objective: If True, the objective function is maximized.
            grammar_type: The type of the input and output grammars,
                either :attr:`.MDODiscipline.JSON_GRAMMAR_TYPE`
                or :attr:`.MDODiscipline.SIMPLE_GRAMMAR_TYPE`.
            **options: The options of the formulation.
        """
        self.check_disciplines(disciplines)
        self.disciplines = disciplines
        self._objective_name = objective_name
        self.opt_problem = OptimizationProblem(design_space)
        self._maximize_objective = maximize_objective
        self.__grammar_type = grammar_type

    @property
    def _grammar_type(self):  # type: (...) -> str
        """The type of the input and output grammars."""
        return self.__grammar_type

    @property
    def design_space(self):  # type: (...) -> DesignSpace
        """The design space on which the formulation is applied."""
        return self.opt_problem.design_space

    @staticmethod
    def check_disciplines(
        disciplines,  # type: Any
    ):  # type: (...) -> None
        """Check that the disciplines are provided as a list.

        Args:
            disciplines: The disciplines.
        """
        if not disciplines or not isinstance(disciplines, list):
            raise TypeError("Disciplines must be provided to the formulation as a list")

    @staticmethod
    def _check_add_cstr_input(
        output_name,  # type: str,
        constraint_type,  # type:str
    ):  # type: (...) -> List[str]
        """Check the output name and constraint type passed to :meth:`.add_constraint`.

        Args:
            output_name: The name of the output to be used as a constraint.
                For instance, if g_1 is given and constraint_type="eq",
                g_1=0 will be added as a constraint to the optimizer.
            constraint_type: The type of constraint,
                either "eq" for equality constraint
                or "ineq" for inequality constraint.
        """
        if constraint_type not in [MDOFunction.TYPE_EQ, MDOFunction.TYPE_INEQ]:
            raise ValueError(
                "Constraint type must be either 'eq' or 'ineq',"
                " got: %s instead" % constraint_type
            )
        if isinstance(output_name, list):
            outputs_list = output_name
        else:
            outputs_list = [output_name]
        return outputs_list

    def add_constraint(
        self,
        output_name,  # type: str
        constraint_type=MDOFunction.TYPE_EQ,  # type: str
        constraint_name=None,  # type: Optional[str]
        value=None,  # type: Optional[float]
        positive=False,  # type: bool
    ):  # type: (...) -> None
        """Add a user constraint.

        A user constraint is a design constraint
        in addition to the formulation specific constraints
        such as the targets (a.k.a. consistency constraints) in IDF.

        The strategy of repartition of constraints is defined in the formulation class.

        Args:
            output_name: The name of the output to be used as a constraint.
                For instance, if g_1 is given and constraint_type="eq",
                g_1=0 will be added as a constraint to the optimizer.
            constraint_type: The type of constraint,
                either "eq" for equality constraint or "ineq" for inequality constraint.
            constraint_name: The name of the constraint to be stored,
                If None, the name is generated from the output name.
            value: The value of activation of the constraint.
                If None, the value is equal to 0.
            positive: If True, the inequality constraint is positive.
        """
        outputs_list = self._check_add_cstr_input(output_name, constraint_type)

        mapped_cstr = FunctionFromDiscipline(outputs_list, self, top_level_disc=True)
        mapped_cstr.f_type = constraint_type

        if constraint_name is not None:
            mapped_cstr.name = constraint_name
        self.opt_problem.add_constraint(mapped_cstr, value=value, positive=positive)

    def add_observable(
        self,
        output_names,  # type: Union[str,Sequence[str]]
        observable_name=None,  # type: Optional[str]
        discipline=None,  # type: Optional[MDODiscipline]
    ):  # type: (...) -> None
        """Add an observable to the optimization problem.

        The repartition strategy of the observable is defined in the formulation class.

        Args:
            output_names: The name(s) of the output(s) to observe.
            observable_name: The name of the observable.
            discipline: The discipline computing the observed outputs.
                If None, the discipline is detected from inner disciplines.
        """
        if isinstance(output_names, string_types):
            output_names = [output_names]
        obs_fun = FunctionFromDiscipline(
            output_names, self, top_level_disc=True, discipline=discipline
        )
        if observable_name is not None:
            obs_fun.name = observable_name
        obs_fun.f_type = MDOFunction.TYPE_OBS
        self.opt_problem.add_observable(obs_fun)

    def get_top_level_disc(self):  # type: (...) -> List[MDODiscipline]
        """Return the disciplines which inputs are required to run the scenario.

        A formulation seeks to
        evaluate objective function and constraints from inputs.
        It structures the optimization problem into multiple levels of disciplines.
        The disciplines directly depending on these inputs
        are called top level disciplines.

        By default, this method returns all disciplines.
        This method can be overloaded by subclasses.

        Returns:
            The top level disciplines.
        """
        return self.disciplines

    @staticmethod
    def _get_mask_from_datanames(
        all_data_names,  # type: ndarray
        masked_data_names,  # type: ndarray
    ):  # type: (...) -> ndarray
        """Get a mask of all_data_names for masked_data_names.

        This mask is an array of the size of all_data_names
        with True values when masked_data_names are in all_data_names.

        Args:
            all_data_names: The main array for mask.
            masked_data_names: The array which masks all_data_names.

        Returns:
            A True / False valued mask array.
        """
        places = in1d(all_data_names, masked_data_names)
        return where(places)

    def _get_generator_from(
        self,
        output_names,  # type: Iterable[str]
        top_level_disc=False,  # type: bool
    ):  # type: (...) -> MDOFunctionGenerator
        """Create a generator of :class:`.MDOFunction` from the names of the outputs.

        Find a discipline which computes all the provided outputs
        and build the associated MDOFunctionGenerator.

        Args:
            output_names: The names of the outputs.
            top_level_disc: If True, search outputs among top level disciplines.

        Returns:
            A generator of :class:`.MDOFunction` instances.
        """
        if top_level_disc:
            search_among = self.get_top_level_disc()
        else:
            search_among = self.disciplines
        for discipline in search_among:
            if discipline.is_all_outputs_existing(output_names):
                return MDOFunctionGenerator(discipline)
        raise ValueError(
            "No discipline known by formulation %s"
            " has all outputs named %s" % (type(self).__name__, output_names)
        )

    def _get_generator_with_inputs(
        self,
        input_names,  # type: Iterable[str]
        top_level_disc=False,  # type: bool
    ):  # type: (...) -> MDOFunctionGenerator
        """Create a generator of :class:`.MDOFunction` from the names of the inputs.

        Find a discipline which has all the provided inputs
        and build the associated MDOFunctionGenerator.

        Args:
            input_names: The names of the inputs.
            top_level_disc: If True, search inputs among the top level disciplines.

        Returns:
            A generator of :class:`.MDOFunction` instances.
        """
        if top_level_disc:
            search_among = self.get_top_level_disc()
        else:
            search_among = self.disciplines
        for discipline in search_among:
            if discipline.is_all_inputs_existing(input_names):
                return MDOFunctionGenerator(discipline)
        raise ValueError(
            "No discipline known by formulation %s"
            " has all inputs named %s" % (type(self).__name__, input_names)
        )

    def mask_x(
        self,
        masking_data_names,  # type: Iterable[str]
        x_vect,  # type: ndarray
        all_data_names=None,  # type: Optional[Iterable[str]]
    ):  # type: (...) -> ndarray
        """Mask a vector from a subset of names, with respect to a set of names.

        Args:
            masking_data_names: The names of data to keep.
            x_vect: The vector of float to mask.
            all_data_names: The set of all names.
                If None, use the design variables stored in the design space.

        Returns:
            A boolean mask with the same shape as the input vector.
        """
        if all_data_names is None:
            all_data_names = self.get_optim_variables_names()
        i_min = 0
        x_mask = array([False] * x_vect.size)
        for key in all_data_names:
            var_length = self._get_dv_length(key)
            i_max = i_min + var_length
            if len(x_vect) < i_max:
                raise ValueError(
                    "Inconsistent input size array %s = %s"
                    " for the design variable of length %s"
                    % (key, x_vect.shape, var_length)
                )
            if key in masking_data_names:
                x_mask[i_min:i_max] = True
            i_min = i_max

        return x_vect[x_mask]

    def unmask_x(
        self,
        masking_data_names,  # type: Iterable[str]
        x_masked,  # type: ndarray
        all_data_names=None,  # type: Optional[Iterable[str]]
        x_full=None,  # type: ndarray
    ):  # type: (...) -> ndarray
        """Unmask a vector from a subset of names, with respect to a set of names.

        Args:
            masking_data_names: The names of the kept data.
            x_masked: The boolean vector to unmask.
            all_data_names: The set of all names.
                If None, use the design variables stored in the design space.
            x_full: The default values for the full vector.
                If None, use the zero vector.

        Returns:
            The vector related to the input mask.
        """
        return self.unmask_x_swap_order(
            masking_data_names, x_masked, all_data_names, x_full
        )

    def _get_dv_length(
        self,
        variable_name,  # type: str
    ):  # type: (...) -> int
        """Retrieve the length of a variable.

        This method relies on the size declared in the design space.

        Args:
            variable_name: The name of the variable.

        Returns:
            The size of the variable.
        """
        return self.opt_problem.design_space.get_size(variable_name)

    def _get_x_mask_swap(
        self,
        masking_data_names,  # type: Iterable[str]
        all_data_names=None,  # type: Optional[Iterable[str]]
    ):  # type: (...) -> Tuple[Dict[str,Tuple[int,int]],int,int]
        """Get a mask from a subset of names, with respect to a set of names.

        This method eventually swaps the order of the values
        if the order of the data names is inconsistent between these sets.

        Args:
            masking_data_names: The names of data to keep.
            all_data_names: The set of all names.
                If None, use the design variables stored in the design space.

        Returns:
            A mask as well as
            the dimension of the restricted variable space
            and the dimension of the original variable space.

            The mask is a dictionary
            indexed by the names of the variables coming from the subset.
            For a given name,
            the value is a tuple
            whose first component is its lowest dimension in the original space
            and the second one is the lowest dimension of the variable that follows it
            in the original space.
        """
        if all_data_names is None:
            all_data_names = self.get_optim_variables_names()
        i_min = 0
        x_values_dict = {}
        n_x = 0
        for key in all_data_names:
            i_max = i_min + self._get_dv_length(key)
            if key in masking_data_names:
                x_values_dict[key] = (i_min, i_max)
                n_x += i_max - i_min
            i_min = i_max
        return x_values_dict, n_x, i_max

    def unmask_x_swap_order(
        self,
        masking_data_names,  # type: Iterable[str]
        x_masked,  # type: ndarray
        all_data_names=None,  # type: Optional[Iterable[str]]
        x_full=None,  # type: ndarray
    ):  # type: (...) -> ndarray
        """Unmask a vector from a subset of names, with respect to a set of names.

        This method eventually swaps the order of the values
        if the order of the data names is inconsistent between these sets.

        Args:
            masking_data_names: The names of the kept data.
            x_masked: The boolean vector to unmask.
            all_data_names: The set of all names.
                If None, use the design variables stored in the design space.
            x_full: The default values for the full vector.
                If None, use the zero vector.

        Returns:
            The vector related to the input mask.
        """
        if all_data_names is None:
            all_data_names = self.get_optim_variables_names()
        x_values_dict, _, len_x = self._get_x_mask_swap(
            masking_data_names, all_data_names
        )
        if x_full is None:
            x_unmask = zeros(len_x, dtype=x_masked.dtype)
        else:
            x_unmask = copy(x_full)

        i_x = 0
        for key in all_data_names:
            if key in x_values_dict:
                i_min, i_max = x_values_dict[key]
                n_x = i_max - i_min
                if x_masked.size < i_x + n_x:
                    raise ValueError(
                        "Inconsistent data shapes !\n"
                        "Try to unmask data %s of length %s\n"
                        "With values of length: %s" % (key, n_x, x_masked.size)
                    )
                x_unmask[i_min:i_max] = x_masked[i_x : i_x + n_x]
                i_x += n_x
        return x_unmask

    def mask_x_swap_order(
        self,
        masking_data_names,  # type: Iterable[str]
        x_vect,  # type: ndarray
        all_data_names=None,  # type: Optional[Iterable[str]]
    ):  # type: (...) -> ndarray
        """Mask a vector from a subset of names, with respect to a set of names.

        This method eventually swaps the order of the values
        if the order of the data names is inconsistent between these sets.

        Args:
            masking_data_names: The names of the kept data.
            x_vect: The vector to mask.
            all_data_names: The set of all names.
                If None, use the design variables stored in the design space.

        Returns:
            The masked version of the input vector.
        """
        if all_data_names is None:
            all_data_names = self.get_optim_variables_names()
        x_values_dict, n_x, _ = self._get_x_mask_swap(
            masking_data_names, all_data_names
        )
        x_masked = zeros(n_x, dtype=x_vect.dtype)
        i_max = 0
        i_min = 0
        for key in masking_data_names:
            if key not in x_values_dict:
                raise ValueError(
                    "Inconsistent inputs of masking. "
                    "Key %s is in masking_data_names %s "
                    "but not in provided all_data_names : %s!"
                    % (key, masking_data_names, all_data_names)
                )
            value = x_values_dict[key]
            i_max += value[1] - value[0]
            len_x = len(x_vect)
            if len(x_masked) < i_max or len_x <= value[0] or len_x < value[1]:
                raise ValueError(
                    "Inconsistent input array size of values array %s "
                    "with reference data shape %s, "
                    "for data named: %s of size: %s"
                    % (x_vect, x_vect.shape, key, i_max)
                )
            x_masked[i_min:i_max] = x_vect[value[0] : value[1]]
            i_min = i_max

        return x_masked

    def _remove_unused_variables(self):  # type: (...) -> None
        """Remove variables in the design space that are not discipline inputs."""
        design_space = self.opt_problem.design_space
        disciplines = self.get_top_level_disc()
        all_inputs = set(
            var for disc in disciplines for var in disc.get_input_data_names()
        )
        for name in set(design_space.variables_names):
            if name not in all_inputs:
                design_space.remove_variable(name)

    def _remove_sub_scenario_dv_from_ds(self):  # type: (...) -> None
        """Remove the sub scenarios design variables from the design space."""
        for scenario in self.get_sub_scenarios():
            loc_vars = scenario.design_space.variables_names
            for var in loc_vars:
                if var in self.design_space.variables_names:
                    self.design_space.remove_variable(var)

    def _build_objective_from_disc(
        self,
        objective_name,  # type: str
        discipline=None,  # type: Optional[MDODiscipline]
        top_level_disc=True,  # type: bool
    ):  # type: (...) -> None
        """Build the objective function from the discipline able to compute it.

        Args:
            objective_name: The name of the objective function.
            discipline: The discipline computing the objective function.
                If None, the discipline is detected from the inner disciplines.
            top_level_disc: If True, search the discipline among the top level ones.
        """
        if isinstance(objective_name, string_types):
            objective_name = [objective_name]
        obj_mdo_fun = FunctionFromDiscipline(
            objective_name, self, discipline, top_level_disc
        )
        obj_mdo_fun.f_type = MDOFunction.TYPE_OBJ
        self.opt_problem.objective = obj_mdo_fun
        if self._maximize_objective:
            self.opt_problem.change_objective_sign()

    def get_optim_variables_names(self):  # type: (...) -> List[str]
        """Get the optimization unknown names to be provided to the optimizer.

        This is different from the design variable names provided by the user,
        since it depends on the formulation,
        and can include target values for coupling for instance in IDF.

        Returns:
            The optimization variable names.
        """
        return self.opt_problem.design_space.variables_names

    def get_x_names_of_disc(
        self,
        discipline,  # type: MDODiscipline
    ):  # type: (...) -> List[str]
        """Get the design variables names of a given discipline.

        Args:
            discipline: The discipline.

        Returns:
             The names of the design variables.
        """
        optim_variables_names = self.get_optim_variables_names()
        input_names = discipline.get_input_data_names()
        return [name for name in optim_variables_names if name in input_names]

    def get_sub_disciplines(self):  # type: (...) ->List[MDODiscipline]
        """Accessor to the sub-disciplines.

        This method lists the sub scenarios' disciplines.

        Returns:
            The sub-disciplines.
        """
        sub_disc = []

        def add_to_sub(
            disc_list,  # type:Iterable[MDODiscipline]
        ):  # type: (...) -> None
            """Add the disciplines of the sub-scenarios if not already added it.

            Args:
                disc_list: The disciplines.
            """
            for disc in disc_list:
                if disc not in sub_disc:
                    sub_disc.append(disc)

        for discipline in self.disciplines:
            if hasattr(discipline, "disciplines"):
                add_to_sub(discipline.disciplines)
            else:
                add_to_sub([discipline])

        return sub_disc

    def get_sub_scenarios(self):  # type: (...) -> List[Scenario]
        """List the disciplines that are actually scenarios.

        Returns:
            The scenarios.
        """
        return [disc for disc in self.disciplines if disc.is_scenario()]

    def _set_defaultinputs_from_ds(self):  # type: (...) -> None
        """Initialize the top level disciplines from the design space."""
        if not self.opt_problem.design_space.has_current_x():
            return
        x_dict = self.opt_problem.design_space.get_current_x_dict()
        for disc in self.get_top_level_disc():
            inputs = disc.get_input_data_names()
            curr_x_disc = {name: x_dict[name] for name in inputs if name in x_dict}
            disc.default_inputs.update(curr_x_disc)

    def get_expected_workflow(
        self,
    ):  # type: (...) -> List[ExecutionSequence,Tuple[ExecutionSequence]]
        """Get the expected sequence of execution of the disciplines.

        This method is used for the XDSM representation
        and can be overloaded by subclasses.

        For instance:

        * [A, B] denotes the execution of A,
          then the execution of B
        * (A, B) denotes the concurrent execution of A and B
        * [A, (B, C), D] denotes the execution of A,
          then the concurrent execution of B and C,
          then the execution of D.

        Returns:
            A sequence of elements which are either
            an :class:`.ExecutionSequence`
            or a tuple of :class:`.ExecutionSequence` for concurrent execution.
        """
        raise NotImplementedError()

    def get_expected_dataflow(
        self,
    ):  # type: (...) -> List[Tuple[MDODiscipline,MDODiscipline,List[str]]]
        """Get the expected data exchange sequence.

        This method is used for the XDSM representation
        and can be overloaded by subclasses.

        Returns:
            The expected sequence of data exchange
            where the i-th item is described by the starting discipline,
            the ending discipline and the coupling variables.
        """
        raise NotImplementedError()

    @classmethod
    def get_default_sub_options_values(
        cls, **options  # type:str
    ):  # type: (...) -> Dict  # pylint: disable=W0613
        """Get the default values of the sub-options of the formulation.

        When some options of the formulation depend on higher level options,
        the default values of these sub-options may be obtained here,
        mainly for use in the API.

        Args:
            **options: The options required to deduce the sub-options grammar.

        Returns:
            Either None or the sub-options default values.
        """

    @classmethod
    def get_sub_options_grammar(
        cls, **options  # type: str
    ):  # type: (...) -> JSONGrammar # pylint: disable=W0613
        """Get the sub-options grammar.

        When some options of the formulation depend on higher level options,
        the schema of the sub-options may be obtained here,
        mainly for use in the API.

        Args:
            **options: The options required to deduce the sub-options grammar.

        Returns:
            Either None or the sub-options grammar.
        """
