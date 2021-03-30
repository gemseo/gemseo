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
"""
Baseclass for all formulations
******************************
"""

from __future__ import absolute_import, division, print_function, unicode_literals

from future import standard_library
from numpy import array, copy, empty, in1d, where, zeros
from six import string_types

from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.core.function import MDOFunction, MDOFunctionGenerator

standard_library.install_aliases()


from gemseo import LOGGER


class MDOFormulation(object):
    """Abstract MDO formulation class
    To be extended in subclasses for use.

    The MDOFormulation creates the objective function and
    constraints from the disciplines.

    It defines the process implicitly.

    The link between MDODisciplines and objective
    functions and constraints is made with MDOFunctionGenerator,
    which generates MDOFunctions from the disciplines.
    """

    NAME = "MDOFormulation"

    def __init__(
        self,
        disciplines,
        objective_name,
        design_space,
        maximize_objective=False,
        **options
    ):  # pylint: disable=W0613
        """
        Constructor, initializes the objective functions and constraints

        :param disciplines: the disciplines list
        :param reference_input_data: the base input data dict for the
            disciplines
        :param objective_name: the objective function data name
        :param design_space: the design space
        :param maximize_objective: if True, the objective function
            is maximized, by default, a minimization is performed
        """
        self.check_disciplines(disciplines)
        self.disciplines = disciplines
        self._objective_name = objective_name
        self.opt_problem = OptimizationProblem(design_space)
        self._maximize_objective = maximize_objective

    @property
    def design_space(self):
        """
        Proxy for formulation.design_space

        :returns: the design space
        """
        return self.opt_problem.design_space

    @staticmethod
    def check_disciplines(disciplines):
        """Sets the disciplines.

        :param disciplines: the disciplines list
        """
        if not disciplines or not isinstance(disciplines, list):
            raise TypeError(
                "Disciplines must be provided" + " to the formulation as a list"
            )

    @staticmethod
    def _check_add_cstr_input(output_name, constraint_type):
        """Checks the add_constraint method inputs
        Can be reused in subclasses

        :param output_name: the output name to be used as constraint
            for instance, if g_1 is given and constraint_type="eq",
            g_1=0 will be added as constraint to the optimizer
        :param constraint_type: the type of constraint, "eq" for equality,
            "ineq" for inequality constraint
        """
        if constraint_type not in [MDOFunction.TYPE_EQ, MDOFunction.TYPE_INEQ]:
            raise ValueError(
                "Constraint type must be either 'eq' or 'ineq',"
                + " got:"
                + str(constraint_type)
                + " instead"
            )
        if isinstance(output_name, list):
            outputs_list = output_name
        else:
            outputs_list = [output_name]
        return outputs_list

    def add_constraint(
        self,
        output_name,
        constraint_type=MDOFunction.TYPE_EQ,
        constraint_name=None,
        value=None,
        positive=False,
    ):
        """Add a user constraint, i.e. a design constraint in addition to
        formulation specific constraints such as targets in IDF.
        The strategy of repartition of constraints is defined in the
        formulation class.

        :param output_name: the output name to be used as constraint
            for instance, if g_1 is given and constraint_type="eq",
            g_1=0 will be added as constraint to the optimizer
        :param constraint_type: the type of constraint, "eq" for equality,
            "ineq" for inequality constraint
            (Default value = MDOFunction.TYPE_EQ)
        :param constraint_name: name of the constraint to be stored,
            if None, generated from the output name (Default value = None)
        :param value: Default value = None)
        :param positive: Default value = False)
        :returns: the constraint as an MDOFunction
            False if the formulation does not dispatch the constraint to the
            optimizers itself
        """
        outputs_list = self._check_add_cstr_input(output_name, constraint_type)

        mapped_cstr = self._get_function_from(outputs_list, top_level_disc=True)
        mapped_cstr.f_type = constraint_type

        if constraint_name is not None:
            mapped_cstr.name = constraint_name
        self.opt_problem.add_constraint(mapped_cstr, value=value, positive=positive)

    def add_observable(self, output_names, observable_name=None, discipline=None):
        """
        Adds observable to the optimization problem. The repartition
        strategy of the observable is defined in the formulation class.

        :param output_names: names of the outputs to observe
        :type output_names: str or list(str)
        :param observable_name: name of the observable, optional.
        :type observable_name: str
        :param discipline: if None, detected from inner disciplines, otherwise
            the discipline used to build the function
            (Default value = None)
        :type discipline: MDODiscipline
        """
        if isinstance(output_names, string_types):
            output_names = [output_names]
        obs_fun = self._get_function_from(
            output_names, top_level_disc=True, discipline=discipline
        )
        if observable_name is not None:
            obs_fun.name = observable_name
        obs_fun.f_type = MDOFunction.TYPE_OBS
        self.opt_problem.add_observable(obs_fun)

    def get_top_level_disc(self):
        """Returns the disciplines which inputs are required to run the
        associated scenario
        By default, returns all disciplines
        To be overloaded by subclasses

        :returns: the list of top level disciplines
        """
        return self.disciplines

    @staticmethod
    def _get_mask_from_datanames(all_data_names, masked_data_names):
        """Gets a mask of all_data_names for masked_data_names, ie an array
        of the size of all_data_names with True values when masked_data_names
        are in all_data_names

        :param all_data_names: the main array for mask
        :param masked_data_names: the array which masks all_data_names
        :returns: a True / False valued mask array
        """
        places = in1d(all_data_names, masked_data_names)
        return where(places)

    def _get_generator_from(self, output_names, top_level_disc=False):
        """Find a discipline which has all outputs named output_names
        and builds the associated MDOFunctionGenerator

        :param top_level_disc: if True, search outputs among top
            level disciplines (Default value = False)
        :param output_names: the output names
        :returns: the discipline
        """
        if top_level_disc:
            search_among = self.get_top_level_disc()
        else:
            search_among = self.disciplines
        for discipline in search_among:
            if discipline.is_all_outputs_existing(output_names):
                return MDOFunctionGenerator(discipline)
        raise ValueError(
            "No discipline known by formulation "
            + type(self).__name__
            + " has all outputs named "
            + str(output_names)
        )

    def _get_generator_with_inputs(self, input_names, top_level_disc=False):
        """Find a discipline which has all inputs named output_names

        :param input_names: the output names
        :param top_level_disc: if True, search outputs among top
            level disciplines (Default value = False)
        :returns: the discipline
        """
        if top_level_disc:
            search_among = self.get_top_level_disc()
        else:
            search_among = self.disciplines
        for discipline in search_among:
            if discipline.is_all_inputs_existing(input_names):
                return MDOFunctionGenerator(discipline)
        raise ValueError(
            "No discipline known by formulation "
            + type(self).__name__
            + " has all inputs named "
            + str(input_names)
        )

    def mask_x(self, masking_data_names, x_vect, all_data_names=None):
        """Masks a vector x_vect, using names masking_data_names,
        and with respect to reference names all_data_names

        :param masking_data_names: the names of data to keep
        :param x_vect: the vector to mask
        :param all_data_names: reference data names, if None,
            self.get_optim_variables_names() used instead
            (Default value = None)
        :returns: masked x_vect
        """
        if all_data_names is None:
            all_data_names = self.get_optim_variables_names()
        i_min = 0
        x_mask = array([False] * x_vect.size)
        for key in all_data_names:
            var_length = self._get_dv_length(key)
            i_max = i_min + var_length
            if len(x_vect) < i_max:
                msg = "Inconsistent input size array " + str(key) + " = "
                msg += str(x_vect.shape) + " for the design variable "
                msg += "of length " + str(var_length)
                raise ValueError(msg)
            if key in masking_data_names:
                x_mask[i_min:i_max] = True
            i_min = i_max

        return x_vect[x_mask]

    def unmask_x(self, masking_data_names, x_masked, all_data_names=None, x_full=None):
        """Unmasks a vector x, using names masking_data_names,
        and with respect to
        reference names all_data_names

        :param masking_data_names: the names of data to keep
        :param x_masked: the vector to unmask
        :param all_data_names: reference data names, if None,
            self.get_optim_variables_names() used instead
            (Default value = None)
        :param x_full: the default values for the full vector, if None,
            np.zeros() is used
        :returns: unmasked x
        """
        return self.unmask_x_swap_order(
            masking_data_names, x_masked, all_data_names, x_full
        )

    def _get_dv_length(self, variable_name):
        """Retrieves the length of a variable from the size declared
        in the design space

        :param variable_name: name of the variable

        """
        return self.opt_problem.design_space.get_size(variable_name)

    def _get_x_mask_swap(self, masking_data_names, all_data_names=None):
        """gets the mask dict, using names masking_data_names,
        and with respect to reference names all_data_names
        eventually swaps the order of the values if data names order
        are inconsistent between masking_data_names and all_data_names

        :param masking_data_names: the names of data to keep
        :param all_data_names: reference data names, if None,
            self.get_optim_variables_names() used instead
            (Default value = None)
        :returns: mask dict
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
        self, masking_data_names, x_masked, all_data_names=None, x_full=None
    ):
        """Unmasks a vector x, using names masking_data_names,
        and with respect to  reference names all_data_names
        eventually swaps the order of the x values if data names order
        are inconsistent between masking_data_names and all_data_names

        :param masking_data_names: the names of data to keep
        :param x_masked: the masked vector
        :param all_data_names: reference data names, if None,
            self.get_optim_variables_names() used instead
            (Default value = None)
        :param x_full: the default values for the full vector, if None,
                        np.zeros() is used
        :returns: unmasked x
        """
        if all_data_names is None:
            all_data_names = self.get_optim_variables_names()
        x_values_dict, _, len_x = self._get_x_mask_swap(
            masking_data_names, all_data_names
        )
        if x_full is None:
            x_unmask = zeros((len_x), dtype=x_masked.dtype)
        else:
            x_unmask = copy(x_full)

        i_x = 0
        for key in all_data_names:
            if key in x_values_dict:
                i_min, i_max = x_values_dict[key]
                n_x = i_max - i_min
                if x_masked.size < i_x + n_x:
                    msg = "Inconsistent data shapes !\nTry to unmask data "
                    msg += key + " of len " + str(n_x)
                    msg += "\nWith values of len : " + str(x_masked.size)
                    raise ValueError(msg)
                x_unmask[i_min:i_max] = x_masked[i_x : i_x + n_x]
                i_x += n_x
        return x_unmask

    def mask_x_swap_order(self, masking_data_names, x_vect, all_data_names=None):
        """Masks a vector x_vect, using names masking_data_names,
        and with respect to reference names all_data_names
        possibly swaps the order of the x_vect values if data names orders
        are inconsistent between masking_data_names and all_data_names

        :param masking_data_names: the names of data to keep
        :param x_vect: the vector to mask
        :param all_data_names: reference data names, if None,
            self.get_optim_variables_names() used instead
            (Default value = None)
        :returns: masked x_vect
        """
        if all_data_names is None:
            all_data_names = self.get_optim_variables_names()
        x_values_dict, n_x, _ = self._get_x_mask_swap(
            masking_data_names, all_data_names
        )
        x_masked = zeros((n_x), dtype=x_vect.dtype)
        i_max = 0
        i_min = 0
        for key in masking_data_names:
            if key not in x_values_dict:
                raise ValueError(
                    "Inconsistent inputs of masking. Key "
                    + str(key)
                    + " is in masking_data_names"
                    + str(masking_data_names)
                    + " but not in provided all_data_names :"
                    + str(all_data_names)
                    + " !"
                )
            value = x_values_dict[key]
            i_max += value[1] - value[0]
            len_x = len(x_vect)
            if len(x_masked) < i_max or len_x <= value[0] or len_x < value[1]:
                raise ValueError(
                    "Inconsistent input array size of values "
                    + "array "
                    + str(x_vect)
                    + " with reference data shape "
                    + str(x_vect.shape)
                    + ", for data named :"
                    + str(key)
                    + " of size : "
                    + str(i_max)
                )
            x_masked[i_min:i_max] = x_vect[value[0] : value[1]]
            i_min = i_max

        return x_masked

    def _remove_unused_variables(self):
        """Removes variables in the design space that are not discipline inputs"""
        design_space = self.opt_problem.design_space
        disciplines = self.get_top_level_disc()
        all_inputs = set(
            var for disc in disciplines for var in disc.get_input_data_names()
        )
        for name in set(design_space.variables_names):
            if name not in all_inputs:
                design_space.remove_variable(name)

    def _remove_sub_scenario_dv_from_ds(self):
        """Removes the sub scenarios design variables from the design space"""
        for scenario in self.get_sub_scenarios():
            loc_vars = scenario.design_space.variables_names
            for var in loc_vars:
                if var in self.design_space.variables_names:
                    self.design_space.remove_variable(var)

    def _build_objective_from_disc(
        self, objective_name, discipline=None, top_level_disc=True
    ):
        """Given the objective name, finds the discipline which is able to
        compute it and builds the objective function from it.

        :param objective_name: the name of the objective
        :param discipline: if None, detected from inner disciplines, otherwise
            the discipline used to build the function
            (Default value = None)
        :param top_level_disc: Default value = True)
        """
        if isinstance(objective_name, string_types):
            objective_name = [objective_name]
        obj_mdo_fun = self._get_function_from(
            objective_name, discipline, top_level_disc=top_level_disc
        )
        obj_mdo_fun.f_type = MDOFunction.TYPE_OBJ
        self.opt_problem.objective = obj_mdo_fun
        if self._maximize_objective:
            self.opt_problem.change_objective_sign()

    def _get_function_from(
        self,
        output_names,
        discipline=None,
        top_level_disc=True,
        x_names=None,
        all_data_names=None,
    ):
        """Builds the objective function from a discipline.

        :param output_names: the name list of the  data
        :param discipline: if None, detected from inner disciplines, otherwise
            the discipline used to build the function
            (Default value = None)
        :param top_level_disc: if True, only high level disciplines
            are used to select discipline to build
            the function (Default value = True)
        :param x_names: names of the design variables, if None, use
            self.get_x_names_of_disc(discipline)
            (Default value = None)
        :param all_data_names: reference data names for masking x, if None,
            self.get_optim_variables_names() used instead
            (Default value = None)
        """
        if discipline is None:
            gen = self._get_generator_from(output_names, top_level_disc=top_level_disc)
            discipline = gen.discipline
        else:
            gen = MDOFunctionGenerator(discipline)

        if x_names is None:
            x_names = self.get_x_names_of_disc(discipline)

        out_x_func = gen.get_function(x_names, output_names)

        def func(x_vect):
            """Function to compute consistency constraints

            :param x: design variable vector
            :param x_vect: returns: value of consistency constraints
                    (=0 if disciplines are at equilibrium)
            :returns: value of consistency constraints
                    (=0 if disciplines are at equilibrium)
            """
            x_of_disc = self.mask_x_swap_order(x_names, x_vect, all_data_names)
            obj_allx_val = out_x_func(x_of_disc)
            return obj_allx_val

        masked_func = MDOFunction(
            func,
            out_x_func.name,
            f_type=MDOFunction.TYPE_OBJ,
            args=x_names,
            expr=out_x_func.expr,
            dim=out_x_func.dim,
            outvars=out_x_func.outvars,
        )
        if out_x_func.has_jac():

            def func_jac(x_vect):
                """Function to compute consistency constraints gradient

                :param x: design variable vector
                :param x_vect: returns: gradient of consistency constraints
                :returns: gradient of consistency constraints
                """
                x_of_disc = self.mask_x_swap_order(x_names, x_vect, all_data_names)

                loc_jac = out_x_func.jac(x_of_disc)  # pylint: disable=E1102

                if len(loc_jac.shape) == 1:
                    # This is surprising but there is a duality between the
                    # masking operation in the function inputs and the
                    # unmasking of its outputs
                    jac = self.unmask_x_swap_order(x_names, loc_jac, all_data_names)
                else:
                    n_outs = loc_jac.shape[0]
                    jac = empty((n_outs, x_vect.size), dtype=x_vect.dtype)
                    for func_ind in range(n_outs):
                        gr_u = self.unmask_x_swap_order(
                            x_names, loc_jac[func_ind, :], all_data_names
                        )
                        jac[func_ind, :] = gr_u
                return jac

            masked_func.jac = func_jac

        return masked_func

    def get_optim_variables_names(self):
        """Gets the optimization unknown names to be provided to the optimizer
        This is different from the design variable names provided by the user,
        since it depends on the formulation, and can include target values
        for coupling for instance in IDF

        :returns: optimization unknown names
        """
        return self.opt_problem.design_space.variables_names

    def get_x_names_of_disc(self, discipline):
        """Gets the design variables names of a given discipline

        :param discipline: the discipline
        :returns: design variables names
        """
        optim_variables_names = self.get_optim_variables_names()
        input_names = discipline.get_input_data_names()
        return [name for name in optim_variables_names if name in input_names]

    def get_sub_disciplines(self):
        """Accessor to the sub disciplines.
        Lists the sub scenarios' disciplines.

        :returns: list of sub disciplines
        """
        sub_disc = []

        def add_to_sub(disc_list):
            """
            Adds the discipline list to the sub_sc disc list
            if not already in it

            :param disc_list: the discipline list
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

    def get_sub_scenarios(self):
        """Lists the disciplines that are actually scenarios

        :returns: the list of scenarios
        """
        return [disc for disc in self.disciplines if disc.is_scenario()]

    def _set_defaultinputs_from_ds(self):
        """
        Sets the default inputs of the top level disciplines
        from the design space
        """
        if not self.opt_problem.design_space.has_current_x():
            return
        x_dict = self.opt_problem.design_space.get_current_x_dict()
        for disc in self.get_top_level_disc():
            inputs = disc.get_input_data_names()
            curr_x_disc = {name: x_dict[name] for name in inputs if name in x_dict}
            disc.default_inputs.update(curr_x_disc)

    def get_expected_workflow(self):
        """Returns the sequence of execution of the disciplines to be expected
        regarding the implementation.
        The returned value is an array containing disciplines,
        tuples of disciplines for concurrent execution.
        For instance :
        * [A, B] denotes the execution of A then the execution of B
        * (A, B) denotes the concurrent execution of A and B
        * [A, (B, C), D] denotes the execution of A then the concurrent
        execution of B and C then the execution of D.

        :returns: array containing disciplines or tuples of disciplines.
            Used for xdsm representation
            To be overloaded by subclasses.
        """
        raise NotImplementedError()

    def get_expected_dataflow(self):
        """Returns the expected data exchange sequence,
        uUsed for xdsm representation
        to be overloaded by subclasses

        :returns: array of tuples (disc_from, disc_to, array of variable names)
        """
        raise NotImplementedError()

    @classmethod
    def get_default_sub_options_values(cls, **options):  # pylint: disable=W0613
        """
        When some options of the formulation depend on higher level
        options, a sub option defaults may be specified here, mainly for
        use in the API

        :param options: options dict required to deduce the sub options grammar
        :returns: None, or the sub options defaults
        """

    @classmethod
    def get_sub_options_grammar(cls, **options):  # pylint: disable=W0613
        """
        When some options of the formulation depend on higher level
        options, a sub option schema may be specified here, mainly for
        use in the API

        :param options: options dict required to deduce the sub options grammar
        :returns: None, or the sub options grammar
        """
