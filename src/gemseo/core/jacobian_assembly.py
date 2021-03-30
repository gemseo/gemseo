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
#        :author: Francois Gallard, Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Coupled derivatives calculations
********************************
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from collections import defaultdict

import matplotlib.pyplot as plt
from future import standard_library
from numpy import atleast_2d, concatenate, empty, ones, zeros
from scipy.sparse import dia_matrix, dok_matrix, vstack
from scipy.sparse.csc import csc_matrix
from scipy.sparse.linalg.dsolve.linsolve import factorized
from scipy.sparse.linalg.interface import LinearOperator

from gemseo.utils.linear_solver import LinearSolver

standard_library.install_aliases()


def none_factory():
    """
    Returns None... To be used for defaultdict
    """


def default_dict_factory():
    """
    Instantiates a defaultdict(None) object
    """
    return defaultdict(none_factory)


class JacobianAssembly(object):
    """Assembly of Jacobians
    Typically, assemble disciplines's Jacobians into a system Jacobian
    """

    DIRECT_MODE = "direct"
    ADJOINT_MODE = "adjoint"
    AUTO_MODE = "auto"
    REVERSE_MODE = "reverse"
    AVAILABLE_MODES = (DIRECT_MODE, ADJOINT_MODE, AUTO_MODE, REVERSE_MODE)

    # matrix types
    SPARSE = "sparse"
    LINEAR_OPERATOR = "linear_operator"
    AVAILABLE_MAT_TYPES = [SPARSE, LINEAR_OPERATOR]

    # linear solvers
    AVAILABLE_SOLVERS = LinearSolver.AVAILABLE_SOLVERS

    def __init__(self, coupling_structure):
        """
        Constructor of the assembly

        :param coupling_structure: the disciplines that form
           the coupled system
        """
        self.coupling_structure = coupling_structure
        self.sizes = {}
        self.disciplines = {}
        self.coupled_system = CoupledSystem()
        self.n_newton_linear_resolutions = 0
        self.linear_solver = LinearSolver()

    def __check_inputs(self, functions, variables, couplings, matrix_type, use_lu_fact):
        """
        Checks the inputs before differentiation

        :param functions: the functions to differentiate
        :param variables: the differentiation variables
        :param couplings: the coupling variables
        :param matrix_type: type of matrix for linearization
        :param use_lu_fact: use LU factorization once for all second members
        """
        unknown_dvars = set(variables)
        unknown_outs = set(functions)
        for discipline in self.coupling_structure.disciplines:
            inputs = set(discipline.get_input_data_names())
            outputs = set(discipline.get_output_data_names())
            unknown_outs = unknown_outs - outputs
            unknown_dvars = unknown_dvars - inputs

        if unknown_dvars:
            raise ValueError(
                "Some of the specified variables are not "
                + "inputs of the disciplines: "
                + str(unknown_dvars)
                + " possible inputs are: "
                + str(
                    [
                        disc.get_input_data_names()
                        for disc in self.coupling_structure.disciplines
                    ]
                )
            )

        if unknown_outs:
            raise ValueError(
                "Some outputs are not computed by the disciplines:"
                + str(unknown_outs)
                + " available outputs are: "
                + str(
                    [
                        disc.get_output_data_names()
                        for disc in self.coupling_structure.disciplines
                    ]
                )
            )

        for coupling in couplings:
            if coupling in variables:
                raise ValueError(
                    "Variable "
                    + str(coupling)
                    + " is both a coupling and a design variable"
                )

        if matrix_type not in self.AVAILABLE_MAT_TYPES:
            raise ValueError(
                "Unknown matrix type "
                + str(matrix_type)
                + ", available ones are "
                + str(self.AVAILABLE_MAT_TYPES)
            )

        if use_lu_fact and matrix_type == self.LINEAR_OPERATOR:
            raise ValueError(
                "Unsupported LU factorization for "
                + "LinearOperators! Please use Sparse matrices"
                + " instead"
            )

    def compute_sizes(self, functions, variables, couplings):
        """Computes the number of scalar functions, variables
        and couplings

        :param functions: the functions to differentiate
        :param variables: the differentiation variables
        :param couplings: the coupling variables
        """
        self.sizes = {}
        self.disciplines = {}
        # search for the functions/variables/couplings in the
        # Jacobians of the disciplines

        # functions
        for function in functions:
            discipline = self.coupling_structure.find_discipline(function)
            self.disciplines[function] = discipline
            # get an arbitrary Jacobian and compute the number of rows
            size = next(iter(discipline.jac[function].values())).shape[0]
            self.sizes[function] = size
        # couplings
        for coupling in couplings:
            discipline = self.coupling_structure.find_discipline(coupling)
            self.disciplines[coupling] = discipline
            # get an arbitrary Jacobian and compute the number of rows
            size = next(iter(discipline.jac[coupling].values())).shape[0]
            self.sizes[coupling] = size
        # variables
        for variable in variables:
            for discipline in self.coupling_structure.disciplines:
                if variable not in self.sizes:
                    for variables_dict in discipline.jac.values():
                        jac = variables_dict.get(variable, None)
                        if jac is not None:
                            self.sizes[variable] = jac.shape[1]
                            self.disciplines[variable] = discipline
                            break

    @staticmethod
    def _check_mode(mode, n_variables, n_functions):
        """Set the differentiation mode (direct or adjoint)

        :param mode: user given mode
        :param n_variables: number of variables
        :param n_functions: number of functions
        """
        if mode == JacobianAssembly.AUTO_MODE:
            if n_variables <= n_functions:
                mode = JacobianAssembly.DIRECT_MODE
            else:
                mode = JacobianAssembly.ADJOINT_MODE
        return mode

    def compute_dimension(self, names):
        """Compute the total number of functions/variables/couplings
        of the whole system

        :param names: list of names of inputs or outputs
        """
        number = 0
        for name in names:
            number += self.sizes[name]
        return number

    def _dres_dvar_sparse(self, residuals, variables, n_residuals, n_variables):
        """Forms the matrix of partial derivatives of residuals
        Given disciplinary Jacobians dYi(Y0...Yn)/dvj,
        fill the sparse Jacobian:
        |           |
        |  dRi/dvj  |
        |           |

        :param residuals: the residuals (R)
        :param variables: the differentiation variables
        :param n_residuals: number of residuals
        :param n_variables: number of variables
        """
        dres_dvar = dok_matrix((n_residuals, n_variables))

        out_i = 0
        # Row blocks
        for residual in residuals:
            residual_size = self.sizes[residual]
            # Find the associated discipline
            discipline = self.disciplines[residual]
            residual_jac = discipline.jac[residual]
            # Column blocks
            out_j = 0
            for variable in variables:
                variable_size = self.sizes[variable]
                if residual == variable:
                    # residual Yi-Yi: put -I in the Jacobian
                    ones_mat = (ones(variable_size), 0)
                    shape = (variable_size, variable_size)
                    diag_mat = -dia_matrix(ones_mat, shape=shape)

                    if self.coupling_structure.is_self_coupled(discipline):
                        jac = residual_jac.get(variable, None)
                        if jac is not None:
                            diag_mat += jac
                    dres_dvar[
                        out_i : out_i + variable_size, out_j : out_j + variable_size
                    ] = diag_mat

                else:
                    # block Jacobian
                    jac = residual_jac.get(variable, None)
                    if jac is not None:
                        n_i, n_j = jac.shape
                        assert n_i == residual_size
                        assert n_j == variable_size
                        # Fill the sparse Jacobian block
                        dres_dvar[out_i : out_i + n_i, out_j : out_j + n_j] = jac
                # Shift the column by block width
                out_j += variable_size
            # Shift the row by block height
            out_i += residual_size
        return dres_dvar.real

    def _dres_dvar_linop(self, residuals, variables, n_residuals, n_variables):
        """Forms the linear operator of partial derivatives of residuals

        :param residuals: the residuals (R)
        :param variables: the differentiation variables
        :param n_residuals: number of residuals
        :param n_variables: number of variables
        """
        # define the linear function
        def dres_dvar(x_array):
            """The linear operator that represents the square matrix dR/dy

            :param x_array: vector multiplied by the matrix
            """
            assert x_array.shape[0] == n_variables
            # initialize the result
            result = zeros(n_residuals)

            out_i = 0
            # Row blocks
            for residual in residuals:
                residual_size = self.sizes[residual]
                # Find the associated discipline
                discipline = self.disciplines[residual]
                residual_jac = discipline.jac[residual]
                # Column blocks
                out_j = 0
                for variable in variables:
                    variable_size = self.sizes[variable]
                    if residual == variable:
                        # residual Yi-Yi: (-I).x = -x
                        sub_x = x_array[out_j : out_j + variable_size]
                        result[out_i : out_i + residual_size] -= sub_x
                    else:
                        # block Jacobian
                        jac = residual_jac.get(variable, None)
                        if jac is not None:
                            sub_x = x_array[out_j : out_j + variable_size]
                            sub_result = jac.dot(sub_x)
                            result[out_i : out_i + residual_size] += sub_result
                    # Shift the column by block width
                    out_j += variable_size
                # Shift the row by block height
                out_i += residual_size
            return result

        return LinearOperator((n_residuals, n_variables), matvec=dres_dvar)

    def _dres_dvar_t_linop(self, residuals, variables, n_residuals, n_variables):
        """Forms the transposed linear operator of
        partial derivatives of residuals

        :param residuals: the residuals (R)
        :param variables: the differentiation variables
        :param n_residuals: number of residuals
        :param n_variables: number of variables
        """
        # define the linear function
        def dres_t_dvar(x_array):
            """The transposed linear operator that represents the square
            matrix dR/dy

            :param x_array: vector multiplied by the matrix

            """
            assert x_array.shape[0] == n_residuals
            # initialize the result
            result = zeros(n_variables)

            out_j = 0
            # Column blocks
            for residual in residuals:
                residual_size = self.sizes[residual]
                # Find the associated discipline
                discipline = self.disciplines[residual]
                residual_jac = discipline.jac[residual]
                # Row blocks
                out_i = 0
                for variable in variables:
                    variable_size = self.sizes[variable]
                    if residual == variable:
                        # residual Yi-Yi: (-I).x = -x
                        sub_x = x_array[out_j : out_j + residual_size]
                        result[out_i : out_i + variable_size] -= sub_x
                    else:
                        # block Jacobian
                        jac = residual_jac.get(variable, None)
                        if jac is not None:
                            sub_x = x_array[out_j : out_j + residual_size]
                            sub_result = jac.T.dot(sub_x)
                            result[out_i : out_i + variable_size] += sub_result
                    # Shift the column by block width
                    out_i += variable_size
                # Shift the row by block height
                out_j += residual_size
            return result

        return LinearOperator((n_variables, n_residuals), matvec=dres_t_dvar)

    def dres_dvar(
        self,
        residuals,
        variables,
        n_residuals,
        n_variables,
        matrix_type=SPARSE,
        transpose=False,
    ):
        """Forms the matrix of partial derivatives of residuals
        Given disciplinary Jacobians dYi(Y0...Yn)/dvj,
        fill the sparse Jacobian:
        |           |
        |  dRi/dvj  |
        |           | (Default value = False)

        :param residuals: the residuals (R)
        :param variables: the differentiation variables
        :param n_residuals: number of residuals
        :param n_variables: number of variables
        :param matrix_type: type of the matrix (Default value = SPARSE)
        :param transpose: if True, transpose the matrix
        """
        if matrix_type == JacobianAssembly.SPARSE:
            sparse_dres_dvar = self._dres_dvar_sparse(
                residuals, variables, n_residuals, n_variables
            )
            if transpose:
                return sparse_dres_dvar.T
            return sparse_dres_dvar

        if matrix_type == JacobianAssembly.LINEAR_OPERATOR:
            if transpose:
                return self._dres_dvar_t_linop(
                    residuals, variables, n_residuals, n_variables
                )
            return self._dres_dvar_linop(residuals, variables, n_residuals, n_variables)

        raise TypeError("cannot handle the matrix type")

    def dfun_dvar(self, function, variables, n_variables):
        """Forms the matrix of partial derivatives of a function
        Given disciplinary Jacobians dJi(v0...vn)/dvj,
        fill the sparse Jacobian:
        |           |
        |  dJi/dvj  |
        |           |

        :param function: the function to differentiate
        :param variables: the differentiation variables
        :param n_variables: number of variables
        :returns: the full Jacobian matrix
        """
        function_size = self.sizes[function]
        dfun_dy = dok_matrix((function_size, n_variables))
        # Find the associated discipline
        discipline = self.disciplines[function]
        function_jac = discipline.jac[function]

        # Loop over differentiation variable
        out_j = 0
        for variable in variables:
            variable_size = self.sizes[variable]
            jac_var = function_jac.get(variable, None)
            if jac_var is not None:
                n_i, n_j = jac_var.shape
                assert n_j == variable_size
                assert n_i == function_size
                # Fill the sparse Jacobian block
                dfun_dy[:, out_j : out_j + n_j] = jac_var
            # Shift the column by block width
            out_j += variable_size
        return dfun_dy

    def total_derivatives(
        self,
        in_data,
        functions,
        variables,
        couplings,
        linear_solver="lgmres",
        mode=AUTO_MODE,
        matrix_type=SPARSE,
        use_lu_fact=False,
        exec_cache_tol=None,
        force_no_exec=False,
        **kwargs
    ):
        """Computes the Jacobian of total derivatives of the coupled system
        formed by the disciplines

        :param in_data: input data dict
        :param functions: the functions to differentiate
        :param variables: the differentiation variables
        :param couplings: the coupling variables
        :param linear_solver: name of the linear solver
            (Default value = 'lgmres')
        :param mode: linearization mode (auto, direct or adjoint)
            (Default value = AUTO_MODE)
        :param matrix_type: representation of the matrix dR/dy (sparse or
            linear operator) (Default value = SPARSE)
        :param use_lu_fact: if True, factorize dres_dy once
            (Default value = False), unsupported for linear operator mode
        :param force_no_exec: if True, the discipline is not
            re executed, cache is loaded anyway
        :param kwargs: dict of optional parameters
        :returns: the dictionary of dictionary of coupled (total) derivatives
        """
        if not functions:
            return defaultdict(default_dict_factory)
        self.__check_inputs(functions, variables, couplings, matrix_type, use_lu_fact)

        # linearize all the disciplines
        self._add_differentiated_inouts(functions, variables, couplings)
        for disc in self.coupling_structure.disciplines:
            if exec_cache_tol is not None:
                disc.cache_tol = exec_cache_tol
            disc.linearize(in_data, force_no_exec=force_no_exec)

        # compute the sizes from the Jacobians
        self.compute_sizes(functions, variables, couplings)
        n_variables = self.compute_dimension(variables)
        n_functions = self.compute_dimension(functions)
        n_couplings = self.compute_dimension(couplings)

        # compute the partial derivatives of the residuals
        dres_dx = self.dres_dvar(couplings, variables, n_couplings, n_variables)

        # compute the partial derivatives of the interest functions
        (dfun_dx, dfun_dy) = ({}, {})
        for fun in functions:
            dfun_dx[fun] = self.dfun_dvar(fun, variables, n_variables)
            dfun_dy[fun] = self.dfun_dvar(fun, couplings, n_couplings)

        mode = self._check_mode(mode, n_variables, n_functions)

        # compute the total derivatives
        total_derivatives = {}
        if mode == JacobianAssembly.DIRECT_MODE:
            # sparse square matrix dR/dy
            dres_dy = self.dres_dvar(
                couplings, couplings, n_couplings, n_couplings, matrix_type=matrix_type
            )
            # compute the coupled derivatives
            total_derivatives = self.coupled_system.direct_mode(
                functions,
                n_variables,
                n_couplings,
                dres_dx,
                dres_dy,
                dfun_dx,
                dfun_dy,
                linear_solver,
                use_lu_fact=use_lu_fact,
                **kwargs
            )
        elif mode == JacobianAssembly.ADJOINT_MODE:
            # transposed square matrix dR/dy^T
            dres_dy_t = self.dres_dvar(
                couplings,
                couplings,
                n_couplings,
                n_couplings,
                matrix_type=matrix_type,
                transpose=True,
            )
            # compute the coupled derivatives
            total_derivatives = self.coupled_system.adjoint_mode(
                functions,
                dres_dx,
                dres_dy_t,
                dfun_dx,
                dfun_dy,
                linear_solver,
                use_lu_fact=use_lu_fact,
                **kwargs
            )
        else:
            raise ValueError("Incorrect linearization mode " + str(mode))

        return self.split_jac(total_derivatives, variables)

    def _add_differentiated_inouts(self, functions, variables, couplings):
        """Adds functions to the list of differentiated
        outputs of all disciplines
        wrt couplings, and variables of the discipline

        :param functions: the functions to differentiate
        :param variables: the differentiation variables
        :param couplings: the coupling variables
        """
        couplings_and_functions = set(couplings) | set(functions)
        couplings_and_variables = set(couplings) | set(variables)

        for discipline in self.coupling_structure.disciplines:
            # outputs
            disc_outputs = discipline.get_output_data_names()
            outputs = list(couplings_and_functions & set(disc_outputs))

            # inputs
            disc_inputs = discipline.get_input_data_names()
            inputs = list(set(disc_inputs) & couplings_and_variables)

            if inputs and outputs:
                discipline.add_differentiated_inputs(inputs)
                discipline.add_differentiated_outputs(outputs)

    def split_jac(self, coupled_system, variables):
        """Splits a Jacobian dict into a dict of dict

        :param coupled_system: the derivatives to split
        :param variables: variables wrt wich the differentiation is performed
        :returns: the Jacobian as a dict of dict
        """
        j_split = {}
        for function, function_jac in coupled_system.items():
            i_out = 0
            sub_jac = {}
            for variable in variables:
                size = self.sizes[variable]
                sub_jac[variable] = function_jac[:, i_out : i_out + size]
                i_out += size
            j_split[function] = sub_jac
        return j_split

    # Newton step computation
    def compute_newton_step(
        self,
        in_data,
        couplings,
        relax_factor,
        linear_solver="lgmres",
        matrix_type=LINEAR_OPERATOR,
        **kwargs
    ):
        """Compute Newton step for the the coupled system of residuals
        formed by the disciplines

        :param in_data: input data dict
        :param couplings: the coupling variables
        :param relax_factor: the relaxation factor
        :param linear_solver: the name of the linear solver
            (Default value = 'lgmres')
        :param matrix_type: representation of the matrix dR/dy (sparse or
            linear operator) (Default value = LINEAR_OPERATOR)
        :param kwargs: optional parameters for the linear solver
        :returns: The Newton step -[dR/dy]^-1 . R as a dict of steps
            per coupling variable
        """
        # linearize the disciplines
        self._add_differentiated_inouts(couplings, couplings, couplings)
        for disc in self.coupling_structure.disciplines:
            disc.linearize(in_data)

        self.compute_sizes(couplings, couplings, couplings)
        n_couplings = self.compute_dimension(couplings)

        # compute the partial derivatives of the residuals
        dres_dy = self.dres_dvar(
            couplings, couplings, n_couplings, n_couplings, matrix_type=matrix_type
        )
        # form the residuals
        res = self.residuals(in_data, couplings)
        # solve the linear system
        newton_step = self.linear_solver.solve(
            dres_dy, -relax_factor * res, linear_solver=linear_solver, **kwargs
        )[:, 0]
        self.n_newton_linear_resolutions += 1

        # split the array of steps
        newton_step_dict = {}
        component = 0
        for coupling in couplings:
            size = self.sizes[coupling]
            newton_step_dict[coupling] = newton_step[component : component + size]
            component += size
        return newton_step_dict

    def residuals(self, in_data, var_names):
        """Forms the matrix of residuals wrt coupling variables
        Given disciplinary explicit calculations Yi(Y0_t,...Yn_t),
        fill the residual matrix:

        ::

            [Y0(Y0_t,...Yn_t) - Y0_t]
            [                       ]
            [Yn(Y0_t,...Yn_t) - Yn_t]

        :param in_data: dictionary of values prescribed for the calculation
            of the residuals (Y0_t,...Yn_t)
        :param var_names: names of variables associated with the residuals (R)
        """
        residual_list = []
        # Build rows blocks
        for var in var_names:
            for discipline in self.coupling_structure.disciplines:
                # Find associated discipline
                if var in discipline.get_output_data_names():
                    discipline_output = discipline.get_outputs_by_name(var)
                    residual = atleast_2d(discipline_output - in_data[var])
                    residual_list.append(residual)
        residual_array = concatenate(residual_list, axis=1)[0, :]
        return residual_array

    # plot method
    def plot_dependency_jacobian(
        self,
        functions=None,
        variables=None,
        save=True,
        show=False,
        filepath=None,
        markersize=None,
    ):
        """Plot the Jacobian matrix
        Nonzero elements of the sparse matrix are represented by blue squares

        :param functions: list of variables (Default value = None)
        :param variables: list of variables (Default value = None)
        :param show: if True, the plot is displayed (Default value = False)
        :param save: if True, the plot is saved in a PDF file (Default
            value = True)
        :param filepath: file path of the saved PDF
        """
        self.compute_sizes(functions, variables, [])
        n_variables = self.compute_dimension(variables)

        total_jac = None
        # compute the positions of the outputs
        outputs_positions = {}
        current_position = 0

        for fun in functions:
            dfun_dx = self.dfun_dvar(fun, variables, n_variables)
            outputs_positions[fun] = current_position
            current_position += self.sizes[fun]

            if total_jac is None:
                total_jac = dfun_dx
            else:
                total_jac = vstack((total_jac, dfun_dx))

        # compute the positions of the inputs
        inputs_positions = {}
        current_position = 0
        for variable in variables:
            inputs_positions[variable] = current_position
            current_position += self.sizes[variable]

        # plot the (sparse) matrix

        fig = plt.figure(figsize=(6, 10))
        ax1 = fig.add_subplot(111)
        plt.spy(total_jac, markersize=markersize)
        ax1.set_aspect("auto")

        plt.yticks(list(outputs_positions.values()), list(outputs_positions.keys()))
        plt.xticks(
            list(inputs_positions.values()), list(inputs_positions.keys()), rotation=90
        )

        filename = None
        if save:
            if filepath is None:
                filename = "coupled_jacobian.pdf"
            else:
                filename = "coupled_jacobian_" + filepath + ".pdf"
            plt.savefig(filename)

        if show:
            plt.show()
        else:
            plt.close()

        return filename


class CoupledSystem(object):
    """This class contains methods that compute coupled (total) derivatives
    of a system of residuals, using several methods:
    - direct or adjoint
    - factorized for multiple RHS
    """

    def __init__(self):
        """
        Constructor
        """
        self.n_linear_resolutions = 0
        self.n_direct_modes = 0
        self.n_adjoint_modes = 0
        self.lu_fact = 0
        self.linear_solver = LinearSolver()

    def direct_mode(
        self,
        functions,
        n_variables,
        n_couplings,
        dres_dx,
        dres_dy,
        dfun_dx,
        dfun_dy,
        linear_solver,
        use_lu_fact,
        **kwargs
    ):
        """Computation of total derivative Jacobian in direct mode

        :param functions: functions to differentiate
        :param n_variables: number of variables
        :param n_couplings: number of couplings
        :param dres_dx: Jacobian of residuals wrt design variables
        :param dres_dy: Jacobian of residuals wrt coupling variables
        :param dfun_dx: Jacobian of functions wrt design variables
        :param dfun_dy: Jacobian of functions wrt coupling variables
        :param linear_solver: name of the linear solver
        :param use_lu_fact: if True, factorize dres_dy once
        :param kwargs: optional parameters
        :type kwargs: dict
        """
        self.n_direct_modes += 1
        if use_lu_fact:
            return self._direct_mode_lu(
                functions, n_variables, n_couplings, dres_dx, dres_dy, dfun_dx, dfun_dy
            )
        return self._direct_mode(
            functions,
            n_variables,
            n_couplings,
            dres_dx,
            dres_dy,
            dfun_dx,
            dfun_dy,
            linear_solver,
            **kwargs
        )

    def adjoint_mode(
        self,
        functions,
        dres_dx,
        dres_dy_t,
        dfun_dx,
        dfun_dy,
        linear_solver,
        use_lu_fact,
        **kwargs
    ):
        """Computation of total derivative Jacobian in adjoint mode

        :param functions: functions to differentiate
        :param dres_dx: Jacobian of residuals wrt design variables
        :param dres_dy: Jacobian of residuals wrt coupling variables
        :param dfun_dx: Jacobian of functions wrt design variables
        :param dfun_dy: Jacobian of functions wrt coupling variables
        :param linear_solver: name of the linear solver
        :param use_lu_fact: if True, factorize dres_dy_t once
        :param kwargs: optional parameters
        :type kwargs: dict
        :param dres_dy_t: param kwargs
        """
        self.n_adjoint_modes += 1
        if use_lu_fact:
            return self._adjoint_mode_lu(
                functions, dres_dx, dres_dy_t, dfun_dx, dfun_dy
            )
        return self._adjoint_mode(
            functions, dres_dx, dres_dy_t, dfun_dx, dfun_dy, linear_solver, **kwargs
        )

    def _direct_mode(
        self,
        functions,
        n_variables,
        n_couplings,
        dres_dx,
        dres_dy,
        dfun_dx,
        dfun_dy,
        linear_solver,
        **kwargs
    ):
        """Computation of total derivative Jacobian in direct mode

        :param functions: functions to differentiate
        :param n_variables: number of variables
        :param n_couplings: number of couplings
        :param dres_dx: Jacobian of residuals wrt design variables
        :param dres_dy: Jacobian of residuals wrt coupling variables
        :param dfun_dx: Jacobian of functions wrt design variables
        :param dfun_dy: Jacobian of functions wrt coupling variables
        :param linear_solver: name of the linear solver
        :param kwargs: optional parameters
        :type kwargs: dict
        """
        # compute the total derivative dy/dx, independent of the
        # function to differentiate
        dy_dx = empty((n_couplings, n_variables))
        self.linear_solver.outer_v = []
        for var_index in range(n_variables):
            dy_dx[:, var_index] = self.linear_solver.solve(
                dres_dy, -dres_dx[:, var_index], linear_solver=linear_solver, **kwargs
            )[:, 0]
            self.n_linear_resolutions += 1
        # assemble the total derivatives of the functions using dy_dx
        jac = {}
        for fun in functions:
            jac[fun] = dfun_dx[fun].toarray() + dfun_dy[fun].dot(dy_dx)
        return jac

    def _adjoint_mode(
        self, functions, dres_dx, dres_dy_t, dfun_dx, dfun_dy, linear_solver, **kwargs
    ):
        """Computation of total derivative Jacobian in adjoint mode

        :param functions: functions to differentiate
        :param dres_dx: Jacobian of residuals wrt design variables
        :param dres_dy: Jacobian of residuals wrt coupling variables
        :param dfun_dx: Jacobian of functions wrt design variables
        :param dfun_dy: Jacobian of functions wrt coupling variables
        :param linear_solver: name of the linear solver
        :param kwargs: optional parameters
        :type kwargs: dict
        :param dres_dy_t: derivatives of the residuals wrt coupling vars
        """
        jac = {}
        # adjoint vector for each interest function
        self.linear_solver.outer_v = []
        for fun in functions:
            dfunction_dx = dfun_dx[fun]
            dfunction_dy = dfun_dy[fun]
            jac[fun] = empty(dfunction_dx.shape)
            # compute adjoint vector for each component of the function
            for fun_component in range(dfunction_dy.shape[0]):
                adjoint = self.linear_solver.solve(
                    dres_dy_t,
                    -dfunction_dy[fun_component, :].T,
                    linear_solver=linear_solver,
                    **kwargs
                )
                self.n_linear_resolutions += 1
                jac[fun][fun_component, :] = (
                    dfunction_dx[fun_component, :] + (dres_dx.T.dot(adjoint)).T
                )
        return jac

    def _direct_mode_lu(
        self, functions, n_variables, n_couplings, dres_dx, dres_dy, dfun_dx, dfun_dy
    ):
        """Computation of total derivative Jacobian in direct mode

        :param functions: functions to differentiate
        :param n_variables: number of variables
        :param n_couplings: number of couplings
        :param dres_dx: Jacobian of residuals wrt design variables
        :param dres_dy: Jacobian of residuals wrt coupling variables
        :param dfun_dx: Jacobian of functions wrt design variables
        :param dfun_dy: Jacobian of functions wrt coupling variables
        :param linear_solver: name of the linear solver
        :param kwargs: optional parameters
        :type kwargs: dict
        """
        # compute the total derivative dy/dx, independent of the
        # function to differentiate
        dy_dx = empty((n_couplings, n_variables))
        # compute LU decomposition
        solve = factorized(csc_matrix(dres_dy))
        self.lu_fact += 1

        for var_index in range(n_variables):
            rhs = -dres_dx[:, var_index].todense()
            dy_dx[:, var_index] = solve(rhs).squeeze()
            self.n_linear_resolutions += 1
        # assemble the total derivatives of the functions using dy_dx
        jac = {}
        for fun in functions:
            jac[fun] = dfun_dx[fun].toarray() + dfun_dy[fun].dot(dy_dx)
        return jac

    def _adjoint_mode_lu(self, functions, dres_dx, dres_dy_t, dfun_dx, dfun_dy):
        """Computation of total derivative Jacobian in adjoint mode

        :param functions: functions to differentiate
        :param dres_dx: Jacobian of residuals wrt design variables
        :param dres_dy: Jacobian of residuals wrt coupling variables
        :param dfun_dx: Jacobian of functions wrt design variables
        :param dfun_dy: Jacobian of functions wrt coupling variables
        :param dres_dy_t:
        """
        jac = {}
        # compute LU factorization
        solve = factorized(dres_dy_t)
        self.lu_fact += 1
        # adjoint vector for each interest function
        for fun in functions:
            dfunction_dx = dfun_dx[fun]
            dfunction_dy = dfun_dy[fun]
            jac[fun] = empty(dfunction_dx.shape)
            # compute adjoint vector for each component of the function
            for fun_component in range(dfunction_dy.shape[0]):
                adjoint = solve(-dfunction_dy[fun_component, :].todense().T)
                self.n_linear_resolutions += 1
                jac[fun][fun_component, :] = (
                    dfunction_dx[fun_component, :] + (dres_dx.T.dot(adjoint)).T
                )
        return jac
