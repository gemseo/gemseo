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
from multiprocessing import cpu_count
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Final
from typing import NamedTuple

import matplotlib.pyplot as plt
from numpy import concatenate
from numpy import empty
from numpy import fill_diagonal
from numpy import ndarray
from numpy import zeros
from numpy.linalg import norm
from scipy.sparse import bmat
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import dok_matrix
from scipy.sparse import eye
from scipy.sparse import vstack
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import factorized
from strenum import StrEnum

from gemseo.algos.linear_solvers.linear_problem import LinearProblem
from gemseo.algos.linear_solvers.linear_solvers_factory import LinearSolversFactory
from gemseo.core.derivatives import derivation_modes
from gemseo.core.derivatives.jacobian_operator import JacobianOperator
from gemseo.core.derivatives.mda_derivatives import traverse_add_diff_io_mda
from gemseo.utils.compatibility.scipy import sparse_classes
from gemseo.utils.matplotlib_figure import save_show_figure

if TYPE_CHECKING:
    from collections.abc import Collection
    from collections.abc import Iterable
    from collections.abc import Iterator
    from collections.abc import Mapping

    from numpy.typing import NDArray

    from gemseo.core.coupling_structure import MDOCouplingStructure
    from gemseo.core.discipline import MDODiscipline

LOGGER = logging.getLogger(__name__)


def none_factory() -> None:
    """Returns None...

    To be used for defaultdict
    """


def default_dict_factory() -> dict[Any, None]:
    """Instantiates a defaultdict(None) object."""
    return defaultdict(none_factory)


class AssembledJacobianOperator(LinearOperator):
    """Representation of the assembled Jacobian as a SciPy ``LinearOperator``."""

    def __init__(
        self,
        functions: Iterable[str],
        variables: Iterable[str],
        n_functions: int,
        n_variables: int,
        get_jacobian_generator: Callable[
            [Iterable[str], Iterable[str], bool], Iterator
        ],
        is_residual: bool = False,
    ) -> None:
        """
        Args:
            functions: The functions to differentiate.
            variables: The differentiation variables.
            n_functions: The number of functions components.
            n_variables: The number of variables components.
            get_jacobian_generator: The method to iterate over the Jacobians associated
                with the provided functions and variables.
            is_residual: Whether the functions are residuals.
        """  # noqa: D205, D212, D415
        super().__init__(shape=(n_functions, n_variables), dtype=float)

        self.__functions = functions
        self.__variables = variables
        self.__is_residual = is_residual
        self.__get_jacobian_generator = get_jacobian_generator

    def _matvec(self, x: ndarray) -> ndarray:
        """The matrix-vector product involving the Jacobian ∂f/∂v.

        Args:
            x: The vector to apply ∂f/∂v to.

        Returns:
            The resulting vector ∂f/∂v x.
        """
        # Initialize the result with appropriate dimension
        result = zeros(self.shape[0], dtype=x.dtype)

        jacobian_generator = self.__get_jacobian_generator(
            self.__functions, self.__variables, self.__is_residual
        )

        for jacobian, position in jacobian_generator:
            result[position.row_slice] += jacobian.dot(x[position.column_slice])

        return result

    def _rmatvec(self, x: ndarray) -> ndarray:
        """The matrix-vector product involving the transposed Jacobian ∂f/∂v.

        Args:
            x: The vector to apply the transpose of ∂f/∂v to.

        Returns:
            The resulting vector (∂f/∂v)^T x.
        """
        # Initialize the result with appropriate dimension
        result = zeros(self.shape[1], dtype=x.dtype)

        jacobian_generator = self.__get_jacobian_generator(
            self.__functions, self.__variables, self.__is_residual
        )

        for jacobian, position in jacobian_generator:
            result[position.column_slice] += jacobian.T.dot(x[position.row_slice])

        return result


class JacobianAssembly:
    """Assembly of Jacobians.

    Typically, assemble discipline's Jacobians into a system Jacobian.
    """

    coupling_structure: MDOCouplingStructure
    """The considered coupling structure."""

    sizes: dict[str, int]
    """The number of elements of a given str."""

    disciplines: dict[str, MDODiscipline]
    """The MDODisciplines, stored using their name."""

    __last_diff_inouts: tuple[set[str], set[str]]
    """The last diff in-outs stored."""

    __minimal_couplings: list[str]
    """The minimal couplings."""

    coupled_system: CoupledSystem
    """The coupled derivative system of residuals."""

    __linear_solver_factory: LinearSolversFactory
    """The linear solver factory."""

    DerivationMode = derivation_modes.DerivationMode

    N_CPUS: Final[int] = cpu_count()
    """The number of available CPUs."""

    class JacobianType(StrEnum):
        """The available types for the Jacobian matrix."""

        LINEAR_OPERATOR = "linear_operator"
        """Jacobian as a SciPy ``LinearOperator`` implementing the appropriate method to
        perform matrix-vector products."""

        MATRIX = "matrix"
        """Jacobian matrix in Compressed Sparse Row (CSR) format."""

    class JacobianPosition(NamedTuple):
        """The position of the discipline's Jacobians within the assembled Jacobian."""

        row_slice: slice
        """The row slice indicating where to position the disciplinary Jacobian within
        the assembled Jacobian when defined as an array."""

        column_slice: slice
        """The column slice indicating where to position the disciplinary Jacobian
        within the assembled Jacobian when defined as an array."""

        row_index: int
        """The row index of the disciplinary Jacobian within the assembled Jacobian when
        defined blockwise."""

        column_index: int
        """The column index of the disciplinary Jacobian within the assembled Jacobian
        when defined blockwise."""

    def __init__(self, coupling_structure: MDOCouplingStructure) -> None:
        """
        Args:
            coupling_structure: The MDOCouplingStructure associated disciplines that
                form the coupled system.
        """  # noqa: D205, D212, D415
        self.coupling_structure = coupling_structure
        self.sizes = {}
        self.disciplines = {}
        self.__last_diff_inouts = ()
        self.__minimal_couplings = []
        self.coupled_system = CoupledSystem()
        self.__linear_solver_factory = LinearSolversFactory(use_cache=True)

    def __check_inputs(
        self,
        functions: Iterable[str],
        variables: Iterable[str],
        couplings: Iterable[str],
        matrix_type: JacobianType,
        use_lu_fact: bool,
    ) -> None:
        """Check the inputs before differentiation.

        Args:
            functions: The functions to differentiate.
            variables: The differentiation variables.
            couplings: The coupling variables.
            matrix_type: The type of matrix for linearization.
            use_lu_fact: Whether to use the LU factorization once for all second
                members.

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
            inputs = [
                disc.get_input_data_names()
                for disc in self.coupling_structure.disciplines
            ]
            raise ValueError(
                "Some of the specified variables are not "
                "inputs of the disciplines: "
                f"{unknown_dvars}"
                " possible inputs are: "
                f"{inputs}"
            )

        if unknown_outs:
            raise ValueError(
                "Some outputs are not computed by the disciplines:"
                + str(unknown_outs)
                + " available outputs are: "
                + str([
                    disc.get_output_data_names()
                    for disc in self.coupling_structure.disciplines
                ])
            )

        for coupling in set(couplings) & set(variables):
            raise ValueError(
                "Variable "
                + str(coupling)
                + " is both a coupling and a design variable"
            )

        matrix_type = self.JacobianType(matrix_type)

        if use_lu_fact and matrix_type == self.JacobianType.LINEAR_OPERATOR:
            raise ValueError(
                "Unsupported LU factorization for "
                "LinearOperators! Please use Sparse matrices"
                " instead"
            )

    def compute_sizes(
        self,
        functions: Iterable[str],
        variables: Iterable[str],
        couplings: Iterable[str],
        residual_variables: Mapping[str, str] | None = None,
    ) -> None:
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
        # search for functions/variables/couplings in the Jacobians of the disciplines
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
            self.sizes[output] = (
                discipline.output_grammar.data_converter.get_value_size(
                    output, discipline.local_data[output]
                )
            )

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

    # TODO: API: give a better name like get_derivation_mode for instance.
    @classmethod
    def _check_mode(
        cls,
        mode: DerivationMode,
        n_variables: int,
        n_functions: int,
    ) -> DerivationMode:
        """Check the differentiation mode.

        Args:
            mode: The differentiation mode.
            n_variables: The number of variables.
            n_functions: The number of functions.

        Returns:
            The differentiation mode.
        """
        if mode != cls.DerivationMode.AUTO:
            return mode
        if n_variables <= n_functions:
            return cls.DerivationMode.DIRECT
        return cls.DerivationMode.ADJOINT

    def compute_dimension(self, names: Iterable[str]) -> int:
        """Compute the total number of functions/variables/couplings of the full system.

        Args:
            names: The names of the inputs or the outputs.

        Returns:
            The dimension if the system.
        """
        return sum(self.sizes[name] for name in names)

    def _get_jacobian_generator(
        self,
        functions: Iterable[str],
        variables: Iterable[str],
        is_residual: bool = False,
    ) -> Iterator[tuple[ndarray | csr_matrix | JacobianOperator, JacobianPosition]]:
        """Iterate over Jacobian matrices.

        Provide a generator to iterate over the Jacobians associated with each provided
        pair (function, variable). The generator yields the Jacobian along with its
        relative position in the to be assembled Jacobian.

        Args:
            functions: The functions to differentiate.
            variables: The differentiation variables.
            is_residual: Whether the functions are residuals.

        Yields:
            A tuple of the form (Jacobian, Position).
        """
        row = 0
        # Iterate over outputs
        for row_index, function in enumerate(functions):
            column = 0
            function_jacobian = self.disciplines[function].jac[function]
            # Iterate over inputs
            for column_index, variable in enumerate(variables):
                jacobian = function_jacobian.get(variable, None)
                variable_size = self.sizes[variable]

                # If residual of the form Yi-Yi, then add -I to the Jacobian
                if is_residual and function == variable:
                    if jacobian is not None:
                        # Make a copy to avoid in-place modifications
                        jacobian_copy = jacobian.copy()

                        if isinstance(jacobian_copy, ndarray):
                            fill_diagonal(jacobian_copy, jacobian.diagonal() - 1)

                        elif isinstance(jacobian_copy, sparse_classes):
                            jacobian_copy.setdiag(jacobian.diagonal() - 1)

                        elif isinstance(jacobian_copy, JacobianOperator):
                            jacobian_copy = jacobian_copy.shift_identity()

                        jacobian = jacobian_copy

                    else:
                        jacobian = -eye(variable_size, dtype=int)

                # Yield only if Jacobian exists
                if jacobian is not None:
                    yield (
                        jacobian.real,
                        self.JacobianPosition(
                            row_slice=slice(row, row + jacobian.shape[0]),
                            column_slice=slice(column, column + jacobian.shape[1]),
                            row_index=row_index,
                            column_index=column_index,
                        ),
                    )

                column += variable_size
            row += self.sizes[function]

    def _assemble_jacobian_as_matrix(
        self,
        functions: Collection[str],
        variables: Collection[str],
        is_residual: bool = False,
    ) -> csr_matrix:
        """Form the Jacobian as a sparse matrix in Compressed Sparse Row (CSR) format.

        The CSR format is well-adapted to perform matrix-vector and matrix-matrix
        multiplications efficiently.

        Args:
            functions: The functions to differentiate.
            variables: The differentiation variables.
            is_residual: Whether the functions are residuals.

        Returns:
            The Jacobian as a sparse matrix in CSR format.
        """
        # Fill in with zero blocks of appropriate dimension if necessary
        variable_sizes = [self.sizes[variable_name] for variable_name in variables]
        function_sizes = [self.sizes[function_name] for function_name in functions]

        # Initialize the block Jacobian with the appropriate structure
        total_jacobian: list[list[None | csr_matrix]] = [
            [None for _ in variables] for _ in functions
        ]

        variable_sizes_0 = variable_sizes[0]
        for i, function_size in enumerate(function_sizes):
            total_jacobian[i][0] = csr_matrix((function_size, variable_sizes_0))

        function_sizes_0 = function_sizes[0]
        total_jacobian_0 = total_jacobian[0]
        for j, variable_size in enumerate(variable_sizes):
            total_jacobian_0[j] = csr_matrix((function_sizes_0, variable_size))

        # Perform the assembly
        jacobian_generator = self._get_jacobian_generator(
            functions, variables, is_residual=is_residual
        )

        for jacobian, position in jacobian_generator:
            if isinstance(jacobian, JacobianOperator):
                jacobian = jacobian.get_matrix_representation()

            total_jacobian[position.row_index][position.column_index] = csr_matrix(
                jacobian.real
            )

        return bmat(total_jacobian, format="csr")

    def assemble_jacobian(
        self,
        functions: Collection[str],
        variables: Collection[str],
        is_residual: bool = False,
        jacobian_type: JacobianType = JacobianType.MATRIX,
    ) -> csr_matrix | AssembledJacobianOperator:
        """Form the Jacobian as a SciPy ``LinearOperator``.

        Args:
            functions: The functions to differentiate.
            variables: The differentiation variables.
            is_residual: Whether the functions are residuals.
            jacobian_type: The type of representation for the Jacobian ∂f/∂v.

        Returns:
            The Jacobian ∂f/∂v in the specified type.
        """
        if jacobian_type == self.JacobianType.MATRIX:
            return self._assemble_jacobian_as_matrix(
                functions,
                variables,
                is_residual,
            )

        if jacobian_type == self.JacobianType.LINEAR_OPERATOR:
            return AssembledJacobianOperator(
                functions,
                variables,
                self.compute_dimension(functions),
                self.compute_dimension(variables),
                self._get_jacobian_generator,
                is_residual,
            )

        raise ValueError(f"Bad jacobian_type: {jacobian_type}")

    def _compute_diff_ios_and_couplings(
        self,
        variables: Iterable[str],
        functions: Iterable[str],
        states: Iterable[str],
        coupling_structure: MDOCouplingStructure,
    ) -> list[str]:
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
            # The state variables are not coupling variables, although they are inputs
            # and outputs of the disciplines with residuals.
            minimal_couplings = sorted(minimal_couplings.difference(states))

            self.__minimal_couplings = minimal_couplings
        return self.__minimal_couplings

    def total_derivatives(
        self,
        in_data,
        functions: Collection[str],
        variables: Collection[str],
        couplings: Iterable[str],
        linear_solver: str = "DEFAULT",
        mode: DerivationMode = DerivationMode.AUTO,
        matrix_type: JacobianType = JacobianType.MATRIX,
        use_lu_fact: bool = False,
        exec_cache_tol: float | None = None,
        execute: bool = True,
        residual_variables: Mapping[str, str] | None = None,
        **linear_solver_options: Any,
    ) -> dict[str, dict[str, ndarray]] | dict[Any, dict[Any, None]]:
        """Compute the Jacobian of total derivatives of the coupled system.

        Args:
            in_data: The input data dict.
            functions: The functions to differentiate.
            variables: The differentiation variables.
            couplings: The coupling variables.
            linear_solver: The name of the linear solver.
            mode: The linearization mode (auto, direct or adjoint).
            matrix_type: The representation of the matrix ∂R/∂y (sparse or
                linear operator).
            use_lu_fact: Whether to factorize dres_dy once,
                unsupported for linear operator mode.
            exec_cache_tol: The discipline cache tolerance to
                when calling the linearize method.
                If ``None``, no tolerance is set (equivalent to tol=0.0).
            execute: Whether to start by executing the discipline
                with the input data for which to compute the Jacobian;
                this allows to ensure that the discipline was executed
                with the right input data;
                it can be almost free if the corresponding output data
                have been stored in the :attr:`.MDODiscipline.cache`.
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

        # Retrieve states variables and local residuals if provided
        states = list(residual_variables.values()) if residual_variables else []

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

            disc.linearize(in_data, execute=execute)

        # compute the sizes from the Jacobians
        self.compute_sizes(functions, variables, couplings_minimal, residual_variables)
        n_variables = self.compute_dimension(variables)
        n_functions = self.compute_dimension(functions)
        n_residuals = self.compute_dimension(couplings_minimal)
        if residual_variables:
            n_residuals += self.compute_dimension(residual_variables.keys())
        # compute the partial derivatives of the residuals
        dres_dx = self.assemble_jacobian(couplings_and_res, variables, is_residual=True)

        # compute the partial derivatives of the interest functions
        (dfun_dx, dfun_dy) = ({}, {})
        for fun in functions:
            dfun_dx[fun] = self.assemble_jacobian([fun], variables)
            dfun_dy[fun] = self.assemble_jacobian([fun], couplings_and_res)

        mode = self._check_mode(mode, n_variables, n_functions)

        # compute the total derivatives
        if mode == self.DerivationMode.DIRECT:
            # sparse square matrix ∂R/∂y

            dres_dy = self.assemble_jacobian(
                couplings_and_res,
                couplings_and_states,
                is_residual=True,
                jacobian_type=matrix_type,
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
        elif mode == self.DerivationMode.ADJOINT:
            # transposed square matrix ∂R/∂y^T
            dres_dy_t = self.assemble_jacobian(
                couplings_and_res,
                couplings_and_states,
                is_residual=True,
                jacobian_type=matrix_type,
            )

            # compute the coupled derivatives
            total_derivatives = self.coupled_system.adjoint_mode(
                functions,
                dres_dx,
                dres_dy_t.T,
                dfun_dx,
                dfun_dy,
                linear_solver,
                use_lu_fact=use_lu_fact,
                **linear_solver_options,
            )
        else:
            raise ValueError("Incorrect linearization mode " + str(mode))

        return self.split_jac(total_derivatives, variables)

    def split_jac(
        self,
        coupled_system: Mapping[str, ndarray | dok_matrix],
        variables: Iterable[str],
    ) -> dict[str, ndarray | dok_matrix]:
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

    def set_newton_differentiated_ios(
        self,
        couplings: Collection[str],
    ) -> None:
        """Set the differentiated inputs and outputs for the Newton algorithm.

        Also ensures that :attr:`.JacobianAssembly.sizes` contains the sizes of all
        the coupling sizes needed for Newton.

        Args:
            couplings: The coupling variables.
        """
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

        self.compute_sizes(couplings, couplings, couplings)

    # Newton step computation
    def compute_newton_step(
        self,
        in_data: Mapping[str, NDArray[float]],
        couplings: Collection[str],
        linear_solver: str = "DEFAULT",
        matrix_type: JacobianType = JacobianType.MATRIX,
        residuals: ndarray | None = None,
        resolved_residual_names: Collection[str] = (),
        **linear_solver_options: Any,
    ) -> tuple[ndarray, bool]:
        """Compute the Newton step for the coupled system of disciplines residuals.

        Args:
            in_data: The input data.
            couplings: The coupling variables.
            linear_solver: The name of the linear solver.
            matrix_type: The representation of the matrix ∂R/∂y (sparse or
                linear operator).
            residuals: The residuals vector, if ``None`` use :attr:`.residuals`.
            resolved_residual_names: The names of residual variables.
            **linear_solver_options: The options passed to the linear solver factory.

        Returns:
            The Newton step - relax_factor . [∂R/∂y]^-1 . R as an array of steps
            for which the order is given by the `couplings` argument.
            Whether the linear solver converged.
        """
        residual_names = (
            resolved_residual_names if resolved_residual_names else couplings
        )

        self.compute_sizes(residual_names, couplings, couplings)

        # compute the partial derivatives of the residuals
        dres_dy = self.assemble_jacobian(
            residual_names,
            couplings,
            is_residual=True,
            jacobian_type=matrix_type,
        )

        # form the residuals
        if residuals is None:
            residuals = self.residuals(in_data, couplings)
        # solve the linear system
        linear_problem = LinearProblem(dres_dy, -residuals)
        self.__linear_solver_factory.execute(
            linear_problem, linear_solver, **linear_solver_options
        )
        return linear_problem.solution, linear_problem.is_converged

    def residuals(
        self, in_data: Mapping[str, Any], var_names: Iterable[str]
    ) -> ndarray:
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
                if name in discipline.output_grammar:
                    to_array = (
                        discipline.output_grammar.data_converter.convert_value_to_array
                    )
                    local_data_array = to_array(name, discipline.local_data[name])
                    in_data_array = to_array(name, in_data[name])
                    residuals.append(local_data_array - in_data_array)

        return concatenate(residuals)

    # plot method
    def plot_dependency_jacobian(
        self,
        functions: Collection[str],
        variables: Collection[str],
        save: bool = True,
        show: bool = False,
        filepath: str | None = None,
        markersize: float | None = None,
    ) -> str:
        """Plot the Jacobian matrix.

        Nonzero elements of the sparse matrix are represented by blue squares.

        Args:
            functions: The functions to plot.
            variables: The variables.
            show: Whether the plot is displayed.
            save: Whether the plot is saved in a PDF file.
            filepath: The file name to save to.
                If ``None``, ``coupled_jacobian.pdf`` is used, otherwise
                ``coupled_jacobian_ + filepath + .pdf``.
            markersize: size of the markers

        Returns:
            The file name.
        """
        self.compute_sizes(functions, variables, [])

        total_jac = None
        # compute the positions of the outputs
        outputs_positions = {}
        current_position = 0

        for fun in functions:
            dfun_dx = self.assemble_jacobian([fun], variables)
            outputs_positions[fun] = current_position
            current_position += self.sizes[fun]

            total_jac = dfun_dx if total_jac is None else vstack((total_jac, dfun_dx))

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

    n_linear_resolutions: int
    """The number of linear resolutions."""

    n_direct_modes: int
    """The number of direct mode calls."""

    n_adjoint_modes: int
    """The number of adjoint mode calls."""

    lu_fact: int
    """The number of LU mode calls (adjoint or direct)."""

    __linear_solver_factory: LinearSolversFactory
    """The linear solver factory."""

    linear_problem: LinearProblem | None
    """The considered linear problem."""

    DEFAULT_LINEAR_SOLVER: ClassVar[str] = "DEFAULT"
    """The default linear solver."""

    def __init__(self) -> None:  # noqa:D107
        self.n_linear_resolutions = 0
        self.n_direct_modes = 0
        self.n_adjoint_modes = 0
        self.lu_fact = 0
        self.__linear_solver_factory = LinearSolversFactory(use_cache=True)
        self.linear_problem = None

    def direct_mode(
        self,
        functions: Iterable[str],
        n_variables: int,
        n_couplings: int,
        dres_dx: dok_matrix | LinearOperator,
        dres_dy: dok_matrix | LinearOperator,
        dfun_dx: Mapping[str, dok_matrix],
        dfun_dy: Mapping[str, dok_matrix],
        linear_solver: str = DEFAULT_LINEAR_SOLVER,
        use_lu_fact: bool = False,
        **linear_solver_options: Any,
    ) -> dict[str, dok_matrix]:
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
        functions: Iterable[str],
        dres_dx: dok_matrix | LinearOperator,
        dres_dy_t: dok_matrix | LinearOperator,
        dfun_dx: Mapping[str, dok_matrix],
        dfun_dy: Mapping[str, dok_matrix],
        linear_solver: str = DEFAULT_LINEAR_SOLVER,
        use_lu_fact: bool = False,
        **linear_solver_options: Any,
    ) -> dict[str, ndarray]:
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
        functions: Iterable[str],
        n_variables: int,
        n_couplings: int,
        dres_dx: dok_matrix | LinearOperator,
        dres_dy: dok_matrix | LinearOperator,
        dfun_dx: Mapping[str, dok_matrix],
        dfun_dy: Mapping[str, dok_matrix],
        linear_solver: str = DEFAULT_LINEAR_SOLVER,
        **linear_solver_options: Any,
    ) -> dict[str, dok_matrix]:
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
        if linear_solver in {"DEFAULT", "LGMRES"}:
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
        functions: Iterable[str],
        dres_dx: dok_matrix | LinearOperator,
        dres_dy_t: dok_matrix | LinearOperator,
        dfun_dx: Mapping[str, dok_matrix],
        dfun_dy: Mapping[str, dok_matrix],
        linear_solver: str = DEFAULT_LINEAR_SOLVER,
        **linear_solver_options: Any,
    ) -> dict[str, ndarray]:
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
        if linear_solver in {"DEFAULT", "LGMRES"}:
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
        functions: Iterable[str],
        n_variables: int,
        n_couplings: int,
        dres_dx: dok_matrix | LinearOperator,
        dres_dy: dok_matrix | LinearOperator,
        dfun_dx: Mapping[str, dok_matrix],
        dfun_dy: Mapping[str, dok_matrix],
        tol: float = 1e-10,
    ) -> dict[str, dok_matrix]:
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
        self,
        functions: Iterable[str],
        dres_dx: dok_matrix | LinearOperator,
        dres_dy_t: dok_matrix | LinearOperator,
        dfun_dx: Mapping[str, dok_matrix],
        dfun_dy: Mapping[str, dok_matrix],
        tol: float = 1e-10,
    ) -> dict[str, ndarray]:
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
