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
"""Coupled derivatives calculations."""
from __future__ import annotations

import itertools
import logging
from collections import defaultdict
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from numpy import atleast_2d
from numpy import concatenate
from numpy import empty
from numpy import ones
from numpy import zeros
from numpy.linalg import norm
from scipy.sparse import dia_matrix
from scipy.sparse import dok_matrix
from scipy.sparse import vstack
from scipy.sparse.csc import csc_matrix
from scipy.sparse.linalg import factorized
from scipy.sparse.linalg import LinearOperator

from gemseo.algos.linear_solvers.linear_problem import LinearProblem
from gemseo.algos.linear_solvers.linear_solvers_factory import LinearSolversFactory
from gemseo.core.derivatives import derivation_modes
from gemseo.core.derivatives.mda_derivatives import traverse_add_diff_io_mda
from gemseo.utils.matplotlib_figure import save_show_figure

if TYPE_CHECKING:
    from gemseo.core.coupling_structure import MDOCouplingStructure
    from typing import Sequence

LOGGER = logging.getLogger(__name__)


def none_factory():
    """Returns None...

    To be used for defaultdict
    """


def default_dict_factory():
    """Instantiates a defaultdict(None) object."""
    return defaultdict(none_factory)


class JacobianAssembly:
    """Assembly of Jacobians.

    Typically, assemble discipline's Jacobians into a system Jacobian.
    """

    DIRECT_MODE = derivation_modes.DIRECT_MODE
    ADJOINT_MODE = derivation_modes.ADJOINT_MODE
    AUTO_MODE = derivation_modes.AUTO_MODE
    REVERSE_MODE = derivation_modes.REVERSE_MODE
    AVAILABLE_MODES = (DIRECT_MODE, ADJOINT_MODE, AUTO_MODE, REVERSE_MODE)

    # matrix types
    SPARSE = "sparse"
    LINEAR_OPERATOR = "linear_operator"
    AVAILABLE_MAT_TYPES = [SPARSE, LINEAR_OPERATOR]

    def __init__(self, coupling_structure):
        """
        Args:
            coupling_structure: The CouplingStructure associated disciplines that form
                the coupled system.
        """  # noqa: D205, D212, D415
        self.coupling_structure = coupling_structure
        self.sizes = {}
        self.disciplines = {}
        self.__last_diff_inouts = tuple()
        self.__minimal_couplings = []
        self.coupled_system = CoupledSystem()
        self.n_newton_linear_resolutions = 0
        self.__linear_solver_factory = LinearSolversFactory()

    def __check_inputs(self, functions, variables, couplings, matrix_type, use_lu_fact):
        """Check the inputs before differentiation.

        Args:
            functions: The functions to differentiate.
            variables: The differentiation variables.
            couplings: The coupling variables.
            matrix_type: The type of matrix for linearization.
            use_lu_fact: Whether to use the LU factorization once for all second members.

        Raises:
            ValueError: When the inputs are inconsistent.
        """
        unknown_dvars = set(variables)
        unknown_outs = set(functions)

        for discipline in self.coupling_structure.disciplines:
            inputs = set(discipline.get_input_data_names())
            outputs = set(discipline.get_output_data_names())
            unknown_outs -= outputs
            unknown_dvars -= inputs

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

        for coupling in set(couplings) & set(variables):
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

    def compute_sizes(self, functions, variables, couplings, residual_variables=None):
        """Compute the number of scalar functions, variables and couplings.

        Args:
            functions: The functions to differentiate.
            variables: The differentiation variables.
            couplings: The coupling variables.
            residual_variables: The mapping of residuals of disciplines to their
                respective state variables.

        Raises:
            ValueError: When the size of some variables could not be determined.
        """
        self.sizes = {}
        self.disciplines = {}
        # search for the functions/variables/couplings in the
        # Jacobians of the disciplines

        if residual_variables:
            outputs = itertools.chain(
                functions,
                couplings,
                residual_variables.keys(),
                residual_variables.values(),
            )
        else:
            outputs = itertools.chain(functions, couplings)

        # functions and coupling and states
        for output in outputs:
            discipline = self.coupling_structure.find_discipline(output)
            self.disciplines[output] = discipline
            # get an arbitrary Jacobian and compute the number of rows
            self.sizes[output] = discipline.local_data[output].shape[0]

        # variables
        for variable in variables:
            for discipline in self.coupling_structure.disciplines:
                if variable not in self.sizes:
                    for jacobian in discipline.jac.values():
                        jacobian_wrt_variable = jacobian.get(variable, None)
                        if jacobian_wrt_variable is not None:
                            self.sizes[variable] = jacobian_wrt_variable.shape[1]
                            self.disciplines[variable] = discipline
                            break

            if variable not in self.sizes:
                raise ValueError(
                    f"Failed to determine the size of input variable {variable}"
                )

    @staticmethod
    def _check_mode(mode, n_variables, n_functions):
        """Check the differentiation mode (direct or adjoint).

        Args:
            mode: The differentiation mode.
            n_variables: The number of variables.
            n_functions: The number of functions.

        Returns:
            The linearization mode.
        """
        if mode == JacobianAssembly.AUTO_MODE:
            if n_variables <= n_functions:
                mode = JacobianAssembly.DIRECT_MODE
            else:
                mode = JacobianAssembly.ADJOINT_MODE
        return mode

    def compute_dimension(self, names):
        """Compute the total number of functions/variables/couplings of the full system.

        Args:
            names: The names of the inputs or the outputs.

        Returns:
            The dimension if the system.
        """
        number = 0
        for name in names:
            number += self.sizes[name]
        return number

    def _dres_dvar_sparse(self, residuals, variables, n_residuals, n_variables):
        """Form the matrix of partial derivatives of residuals.

        Given disciplinary Jacobians dYi(Y0...Yn)/dvj,
        fill the sparse Jacobian:
        |           |
        |  dRi/dvj  |
        |           |

        Args:
            residuals: The residuals.
            variables: The differentiation variables.
            n_residuals: The number of residuals.
            n_variables: The number of variables.

        Returns:
            The derivatives of the residuals wrt the variables.
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
            out_i += residual_size

        return dres_dvar.real

    def _dres_dvar_linop(self, residuals, variables, n_residuals, n_variables):
        """Form the linear operator of partial derivatives of residuals.

        Args:
            residuals: The residuals.
            variables: The differentiation variables.
            n_residuals:  The number of residuals.
            n_variables:  The number of variables.

        Returns:
            The operator dres_dvar.
        """
        # define the linear function
        def dres_dvar(x_array):
            """The linear operator that represents the square matrix dR/dy.

            Args:
                x_array: vector multiplied by the matrix
            """
            assert x_array.shape[0] == n_variables
            # initialize the result
            result = zeros(n_residuals, dtype=x_array.dtype)

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
                        if self.coupling_structure.is_self_coupled(discipline):
                            jac = residual_jac.get(variable, None)
                            if jac is not None:
                                result[out_i : out_i + residual_size] += jac.dot(sub_x)
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
        """Form the transposed linear operator of partial derivatives of residuals.

        Args:
            residuals: The residuals.
            variables: The differentiation variables.
            n_residuals: The number of residuals.
            n_variables: The number of variables.

        Returns:
            The transpose of the operator dres_dvar.
        """
        # define the linear function
        def dres_t_dvar(x_array):
            """The transposed linear operator that represents the square matrix dR/dy.

            Args:
                x_array: The vector multiplied by the matrix.
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
                        if self.coupling_structure.is_self_coupled(discipline):
                            jac = residual_jac.get(variable, None)
                            if jac is not None:
                                result[out_i : out_i + residual_size] += jac.T.dot(
                                    sub_x
                                )
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
        """Form the matrix of partial derivatives of residuals.

        Given disciplinary Jacobians dYi(Y0...Yn)/dvj,
        fill the sparse Jacobian:
        |           |
        |  dRi/dvj  |
        |           | (Default value = False)

        Args:
            residuals: The residuals.
            variables: The differentiation variables.
            n_residuals: The number of residuals.
            n_variables: The number of variables.
            matrix_type: The type of the matrix.
            transpose: Whether to transpose the matrix.

        Returns:
            The jacobian of dres_dvar.

        Raises:
            TypeError: When the matrix type is unknown.
        """
        if matrix_type == JacobianAssembly.SPARSE:
            sparse_dres_dvar = self._dres_dvar_sparse(
                residuals,
                variables,
                n_residuals,
                n_variables,
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
        """Forms the matrix of partial derivatives of a function.

        Given disciplinary Jacobians dJi(v0...vn)/dvj,
        fill the sparse Jacobian:
        |           |
        |  dJi/dvj  |
        |           |

        Args:
            function: The function to differentiate.
            variables: The differentiation variables.
            n_variables: The number of variables.

        Returns:
            The full Jacobian matrix.
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

    def _compute_diff_ios_and_couplings(
        self,
        variables: Sequence[str],
        functions: Sequence[str],
        states: Sequence[str],
        coupling_structure: MDOCouplingStructure,
    ):
        """Compute the minimal differentiated inputs, outputs and couplings.

        This is done form the
        disciplines that are required to differentiate the functions with respect to the
        variables.

        This uses a graph algorithm that computes the dependency
        process graph and address the "jacobian accumulation" problem with a heuristic
        but conservative approach.

        Args:
            variables: The differentiation variables.
            functions: The functions to differentiate.
            states: The state variables.
            coupling_structure: The coupling structure containing all the disciplines.

        Returns: The minimal coupling variables set requires to differentiate the
            functions with respect to the variables.
        """
        diff_ios = (set(variables), set(functions))
        if self.__last_diff_inouts != diff_ios:
            diff_ios_merged = traverse_add_diff_io_mda(
                coupling_structure, variables, functions
            )
            self.__last_diff_inouts = diff_ios

            couplings = [
                coupl
                for coupls in diff_ios_merged.values()
                for coupl in list(coupls[0]) + list(coupls[1])
            ]

            minimal_couplings = set(couplings).intersection(
                coupling_structure.all_couplings
            )
            # The state variables are not coupling variables although they are inputs
            # and outputs of the disciplines with residuals.
            minimal_couplings = sorted(minimal_couplings.difference(states))

            self.__minimal_couplings = minimal_couplings
        return self.__minimal_couplings

    def total_derivatives(
        self,
        in_data,
        functions,
        variables,
        couplings,
        linear_solver="DEFAULT",
        mode=AUTO_MODE,
        matrix_type=SPARSE,
        use_lu_fact=False,
        exec_cache_tol=None,
        force_no_exec=False,
        residual_variables=None,
        **linear_solver_options,
    ):
        """Compute the Jacobian of total derivatives of the coupled system.

        Args:
            in_data: The input data dict.
            functions: The functions to differentiate.
            variables: The differentiation variables.
            couplings: The coupling variables.
            linear_solver: The name of the linear solver.
            mode: The linearization mode (auto, direct or adjoint).
            matrix_type: The representation of the matrix dR/dy (sparse or
                linear operator).
            use_lu_fact: Whether to factorize dres_dy once,
                unsupported for linear operator mode.
            exec_cache_tol: The discipline cache tolerance to
                when calling the linearize method.
                If None, no tolerance is set (equivalent to tol=0.0).
            force_no_exec: Whether the discipline is not re-executed,
                the cache is loaded anyway.
            linear_solver_options: The options passed to the linear solver factory.
            residual_variables: a mapping of residuals of disciplines to
                their respective state variables.
            **linear_solver_options: The options passed to the linear solver factory.

        Returns:
            The total coupled derivatives.

        Raises:
            ValueError: When the linearization_mode is incorrect.
        """
        if not functions:
            return defaultdict(default_dict_factory)

        self.__check_inputs(functions, variables, couplings, matrix_type, use_lu_fact)

        if residual_variables:
            states = list(residual_variables.values())
        else:
            states = []
        couplings_minimal = self._compute_diff_ios_and_couplings(
            variables,
            functions,
            states,
            self.coupling_structure,
        )

        couplings_and_res = couplings_minimal.copy()
        couplings_and_states = couplings_minimal.copy()
        # linearize all the disciplines
        if residual_variables is not None and residual_variables:
            couplings_and_res += residual_variables.keys()
            couplings_and_states += states

        for disc in self.coupling_structure.disciplines:
            if disc.cache is not None and exec_cache_tol is not None:
                disc.cache_tol = exec_cache_tol
            disc.linearize(in_data, force_no_exec=force_no_exec)

        # compute the sizes from the Jacobians
        self.compute_sizes(functions, variables, couplings_minimal, residual_variables)
        n_variables = self.compute_dimension(variables)
        n_functions = self.compute_dimension(functions)
        n_residuals = self.compute_dimension(couplings_minimal)
        if residual_variables:
            n_residuals += self.compute_dimension(residual_variables.keys())
            n_variables += self.compute_dimension(residual_variables.values())
        # compute the partial derivatives of the residuals
        dres_dx = self.dres_dvar(
            couplings_and_res,
            variables,
            n_residuals,
            n_variables,
        )

        # compute the partial derivatives of the interest functions
        (dfun_dx, dfun_dy) = ({}, {})
        for fun in functions:
            dfun_dx[fun] = self.dfun_dvar(fun, variables, n_variables)
            dfun_dy[fun] = self.dfun_dvar(fun, couplings_minimal, n_residuals)

        mode = self._check_mode(mode, n_variables, n_functions)

        # compute the total derivatives
        if mode == JacobianAssembly.DIRECT_MODE:
            # sparse square matrix dR/dy

            dres_dy = self.dres_dvar(
                couplings_and_res,
                couplings_and_states,
                n_residuals,
                n_residuals,
                matrix_type=matrix_type,
            )
            # compute the coupled derivatives
            total_derivatives = self.coupled_system.direct_mode(
                functions,
                n_variables,
                n_residuals,
                dres_dx,
                dres_dy,
                dfun_dx,
                dfun_dy,
                linear_solver,
                use_lu_fact=use_lu_fact,
                **linear_solver_options,
            )
        elif mode == JacobianAssembly.ADJOINT_MODE:
            # transposed square matrix dR/dy^T
            dres_dy_t = self.dres_dvar(
                couplings_and_res,
                couplings_and_states,
                n_residuals,
                n_residuals,
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
                **linear_solver_options,
            )
        else:
            raise ValueError("Incorrect linearization mode " + str(mode))

        return self.split_jac(total_derivatives, variables)

    def split_jac(self, coupled_system, variables):
        """Split a Jacobian dict into a dict of dict.

        Args:
            coupled_system: The derivatives to split.
            variables: The variables wrt which the differentiation is performed.

        Returns:
            The Jacobian.
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

    def compute_newton_step(
        self,
        in_data,
        couplings,
        relax_factor,
        linear_solver="DEFAULT",
        matrix_type=SPARSE,
        **linear_solver_options,
    ):
        """Compute the Newton step for the coupled system of disciplines residuals.

        Args:
            in_data: The input data.
            couplings: The coupling variables.
            relax_factor: The relaxation factor.
            linear_solver: The name of the linear solver.
            matrix_type: The representation of the matrix dR/dy (sparse or
                linear operator).
            **linear_solver_options: The options passed to the linear solver factory.

        Returns:
            The Newton step -[dR/dy]^-1 . R as a dict of steps
            per coupling variable.
        """
        # linearize the disciplines
        for disc in self.coupling_structure.disciplines:
            inputs_to_linearize = set(disc.get_input_data_names()).intersection(
                couplings
            )
            outputs_to_linearize = set(disc.get_output_data_names()).intersection(
                couplings
            )

            # If outputs and inputs to linearize not empty, then linearize
            if inputs_to_linearize and outputs_to_linearize:
                disc.add_differentiated_inputs(inputs_to_linearize)
                disc.add_differentiated_outputs(outputs_to_linearize)
                disc.linearize(in_data)
            # Otherwise,
            # Execute and populate with empty dicts the Jacobian
            # with the outputs to linearize
            # This will be needed when creating the dRes/dCoupling matrix.
            else:
                disc.execute(in_data)
                disc.jac = {}
                for out in outputs_to_linearize:
                    disc.jac[out] = {}

        self.compute_sizes(couplings, couplings, couplings)
        n_couplings = self.compute_dimension(couplings)

        # compute the partial derivatives of the residuals
        dres_dy = self.dres_dvar(
            couplings, couplings, n_couplings, n_couplings, matrix_type=matrix_type
        )
        # form the residuals
        res = self.residuals(in_data, couplings)
        # solve the linear system
        linear_problem = LinearProblem(dres_dy, -relax_factor * res)
        self.__linear_solver_factory.execute(
            linear_problem, linear_solver, **linear_solver_options
        )
        newton_step = linear_problem.solution
        self.n_newton_linear_resolutions += 1

        # split the array of steps
        couplings_to_steps = {}
        component = 0
        for coupling in couplings:
            size = self.sizes[coupling]
            couplings_to_steps[coupling] = newton_step[component : component + size]
            component += size

        return couplings_to_steps

    def residuals(self, in_data, var_names):
        """Form the matrix of residuals wrt coupling variables.

        Given disciplinary explicit calculations Yi(Y0_t,...Yn_t),
        fill the residual matrix::

            [Y0(Y0_t,...Yn_t) - Y0_t]
            [                       ]
            [Yn(Y0_t,...Yn_t) - Yn_t]

        Args:
            in_data: The values prescribed for the calculation
                of the residuals (Y0_t,...Yn_t).
            var_names: The names of variables associated with the residuals (R).

        Returns:
            The residuals array.
        """
        residuals = []
        # Build rows blocks
        for name in var_names:
            for discipline in self.coupling_structure.disciplines:
                # Find associated discipline
                if name in discipline.get_output_data_names():
                    residuals.append(
                        atleast_2d(discipline.get_outputs_by_name(name) - in_data[name])
                    )

        return concatenate(residuals, axis=1)[0, :]

    def plot_dependency_jacobian(
        self,
        functions,
        variables,
        save=True,
        show=False,
        filepath=None,
        markersize=None,
    ):
        """Plot the Jacobian matrix.

        Nonzero elements of the sparse matrix are represented by blue squares.

        Args:
            functions: The functions to plot.
            variables: The variables.
            show: WHether the plot is displayed.
            save: WHether the plot is saved in a PDF file.
            filepath: The file name to save to.
                If None, ``coupled_jacobian.pdf`` is used, otherwise
                ``coupled_jacobian_ + filepath + .pdf``.

        Returns:
            The file name.
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

        fig = plt.figure(figsize=(6.0, 10.0))
        ax1 = fig.add_subplot(111)
        plt.spy(total_jac, markersize=markersize)
        ax1.set_aspect("auto")

        plt.yticks(list(outputs_positions.values()), list(outputs_positions.keys()))
        plt.xticks(
            list(inputs_positions.values()), list(inputs_positions.keys()), rotation=90
        )

        if save:
            if filepath is None:
                filename = "coupled_jacobian.pdf"
            else:
                filename = f"coupled_jacobian_{filepath}.pdf"
        else:
            filename = None

        save_show_figure(fig, show, filename)
        return filename


class CoupledSystem:
    """Compute coupled (total) derivatives of a system of residuals.

    Use several methods:

        - direct or adjoint
        - factorized for multiple RHS
    """

    def __init__(self):  # noqa:D107
        self.n_linear_resolutions = 0
        self.n_direct_modes = 0
        self.n_adjoint_modes = 0
        self.lu_fact = 0
        self.__linear_solver_factory = LinearSolversFactory()
        self.linear_problem = None

    def direct_mode(
        self,
        functions,
        n_variables,
        n_couplings,
        dres_dx,
        dres_dy,
        dfun_dx,
        dfun_dy,
        linear_solver="DEFAULT",
        use_lu_fact=False,
        **linear_solver_options,
    ):
        """Compute the total derivative Jacobian in direct mode.

        Args:
            functions: The functions to differentiate.
            n_variables: The number of variables.
            n_couplings: The number of couplings.
            dres_dx: The Jacobian of the residuals wrt the design variables.
            dres_dy: The Jacobian of the residuals wrt the coupling variables.
            dfun_dx: The Jacobian of the functions wrt the design variables.
            dfun_dy: The Jacobian of the functions wrt the coupling variables.
            linear_solver: The name of the linear solver.
            use_lu_fact: Whether to factorize dres_dy once.
            **linear_solver_options: The optional parameters.

        Returns:
            The Jacobian of the total coupled derivatives.
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
            **linear_solver_options,
        )

    def adjoint_mode(
        self,
        functions,
        dres_dx,
        dres_dy_t,
        dfun_dx,
        dfun_dy,
        linear_solver="DEFAULT",
        use_lu_fact=False,
        **linear_solver_options,
    ):
        """Compute the total derivative Jacobian in adjoint mode.

        Args:
            functions: The functions to differentiate.
            dres_dx: The Jacobian of the residuals wrt the design variables.
            dres_dy_t: The Jacobian of the residuals wrt the coupling variables.
            dfun_dx: The Jacobian of the functions wrt the design variables.
            dfun_dy: The Jacobian of the functions wrt the coupling variables.
            linear_solver: The name of the linear solver.
            use_lu_fact: Whether to factorize dres_dy_t once.
            **linear_solver_options: The optional parameters.

        Returns:
            The Jacobian of total coupled derivatives.
        """
        self.n_adjoint_modes += 1
        if use_lu_fact:
            return self._adjoint_mode_lu(
                functions, dres_dx, dres_dy_t, dfun_dx, dfun_dy
            )
        return self._adjoint_mode(
            functions,
            dres_dx,
            dres_dy_t,
            dfun_dx,
            dfun_dy,
            linear_solver,
            **linear_solver_options,
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
        linear_solver="DEFAULT",
        **linear_solver_options,
    ):
        """Compute the total derivative Jacobian in direct mode.

        Args:
            functions: The functions to differentiate.
            n_variables: The number of variables.
            n_couplings: The number of couplings.
            dres_dx: The Jacobian of the residuals wrt the design variables.
            dres_dy: The Jacobian of the residuals wrt the coupling variables.
            dfun_dx: The Jacobian of the functions wrt the design variables.
            dfun_dy: The Jacobian of the functions wrt the coupling variables.
            linear_solver: The name of the linear solver.
            **linear_solver_options: The optional parameters.

        Returns:
            The Jacobian of total coupled derivatives.
        """
        # compute the total derivative dy/dx, independent of the
        # function to differentiate
        dy_dx = empty((n_couplings, n_variables))
        self.linear_problem = LinearProblem(dres_dy)
        if linear_solver in ["DEFAULT", "LGMRES"]:
            # Reinit outerV, and store it for all RHS
            linear_solver_options["outer_v"] = []
        for var_index in range(n_variables):
            self.linear_problem.rhs = -dres_dx[:, var_index]
            self.__linear_solver_factory.execute(
                self.linear_problem, linear_solver, **linear_solver_options
            )
            dy_dx[:, var_index] = self.linear_problem.solution
            self.n_linear_resolutions += 1
        # assemble the total derivatives of the functions using dy_dx
        jac = {}
        for fun in functions:
            jac[fun] = dfun_dx[fun].toarray() + dfun_dy[fun].dot(dy_dx)
        return jac

    def _adjoint_mode(
        self,
        functions,
        dres_dx,
        dres_dy_t,
        dfun_dx,
        dfun_dy,
        linear_solver="DEFAULT",
        **linear_solver_options,
    ):
        """Compute the total derivative Jacobian in adjoint mode.

        Args:
            functions: The functions to differentiate.
            dres_dx: The Jacobian of the residuals wrt the design variables.
            dres_dy: The Jacobian of the residuals wrt the coupling variables.
            dfun_dx: The Jacobian of the functions wrt the design variables.
            dfun_dy: The Jacobian of the functions wrt the coupling variables.
            linear_solver: The name of the linear solver.
            dres_dy_t: The derivatives of the residuals wrt coupling vars.
            **linear_solver_options: The optional parameters.

        Returns:
            The Jacobian of total coupled derivatives.
        """
        jac = {}

        # adjoint vector for each interest function
        if linear_solver in ["DEFAULT", "LGMRES"]:
            # Reinit outerV, and store it for all RHS
            linear_solver_options["outer_v"] = []

        self.linear_problem = LinearProblem(dres_dy_t)

        for fun in functions:
            dfunction_dx = dfun_dx[fun]
            dfunction_dy = dfun_dy[fun]
            jac[fun] = empty(dfunction_dx.shape)
            # compute adjoint vector for each component of the function
            for fun_component in range(dfunction_dy.shape[0]):
                self.linear_problem.rhs = -dfunction_dy[fun_component, :].T
                self.__linear_solver_factory.execute(
                    self.linear_problem, linear_solver, **linear_solver_options
                )
                adjoint = self.linear_problem.solution
                self.n_linear_resolutions += 1
                jac[fun][fun_component, :] = (
                    dfunction_dx[fun_component, :] + (dres_dx.T.dot(adjoint)).T
                )
        return jac

    def _direct_mode_lu(
        self,
        functions,
        n_variables,
        n_couplings,
        dres_dx,
        dres_dy,
        dfun_dx,
        dfun_dy,
        tol=1e-10,
    ):
        """Compute the total derivative Jacobian in direct mode.

        Args:
            functions: The functions to differentiate.
            n_variables: The number of variables.
            n_couplings: The number of couplings.
            dres_dx: The Jacobian of the residuals wrt the design variables.
            dres_dy: The Jacobian of the residuals wrt the coupling variables.
            dfun_dx: The Jacobian of the functions wrt the design variables.
            dfun_dy: The Jacobian of the functions wrt the coupling variables.
            tol: Tolerance on the relative residuals norm to consider that the linear
                system is solved. If not, raises a warning.

        Returns:
            The Jacobian of total coupled derivatives.
        """
        # compute the total derivative dy/dx, independent of the
        # function to differentiate
        dy_dx = empty((n_couplings, n_variables))
        # compute LU decomposition
        lhs = csc_matrix(dres_dy)
        solve = factorized(lhs)
        self.lu_fact += 1

        for var_index in range(n_variables):
            rhs = -dres_dx[:, var_index].todense()
            sol = solve(rhs)
            dy_dx[:, var_index] = sol.squeeze()
            self.n_linear_resolutions += 1
            res = norm(lhs.dot(sol) - rhs) / norm(rhs)
            if res > tol:
                LOGGER.warning(
                    "The linear system in _direct_mode_lu used to compute the coupled "
                    "derivatives is not well resolved, "
                    "residuals > tolerance : %s > %s ",
                    res,
                    tol,
                )
        # assemble the total derivatives of the functions using dy_dx
        jac = {}
        for fun in functions:
            jac[fun] = dfun_dx[fun].toarray() + dfun_dy[fun].dot(dy_dx)
        return jac

    def _adjoint_mode_lu(
        self, functions, dres_dx, dres_dy_t, dfun_dx, dfun_dy, tol=1e-10
    ):
        """Compute the total derivative Jacobian in adjoint mode.

        Args:
            functions: The functions to differentiate.
            dres_dx: The Jacobian of the residuals wrt the design variables.
            dfun_dx: The Jacobian of the functions wrt the design variables.
            dfun_dy: The Jacobian of the functions wrt the coupling variables.
            dres_dy_t: The Jacobian of the residuals wrt the coupling variables.
            tol: Tolerance on the relative residuals norm to consider that the linear
                system is solved. If not, raises a warning.

        Returns:
            The Jacobian of total coupled derivatives.
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
                rhs = -dfunction_dy[fun_component, :].todense().T
                adjoint = solve(rhs)
                res = norm(dres_dy_t.dot(adjoint) - rhs) / norm(rhs)
                if res > tol:
                    LOGGER.warning(
                        "The linear system in _adjoint_mode_lu used to compute the "
                        "coupled "
                        "derivatives is not well resolved, "
                        "residuals > tolerance : %s > %s ",
                        res,
                        tol,
                    )

                self.n_linear_resolutions += 1
                jac[fun][fun_component, :] = (
                    dfunction_dx[fun_component, :] + (dres_dx.T.dot(adjoint)).T
                )
        return jac
