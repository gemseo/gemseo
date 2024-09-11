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
"""A function to be attached to a problem."""

from __future__ import annotations

from multiprocessing import Value
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import ClassVar

from numpy import isnan
from numpy import ndarray

from gemseo.algos.database import Database
from gemseo.algos.stop_criteria import DesvarIsNan
from gemseo.algos.stop_criteria import FunctionIsNan
from gemseo.algos.stop_criteria import MaxIterReachedException
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.core.mdo_functions.mdo_function import OutputType
from gemseo.core.mdo_functions.mdo_function import WrappedFunctionType
from gemseo.core.serializable import Serializable
from gemseo.utils.derivatives.factory import GradientApproximatorFactory

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.evaluation_counter import EvaluationCounter
    from gemseo.typing import NumberArray
    from gemseo.typing import RealOrComplexArrayT
    from gemseo.utils.derivatives.approximation_modes import ApproximationMode


class ProblemFunction(MDOFunction, Serializable):
    """A function to be attached to a problem."""

    enable_statistics: ClassVar[bool] = True
    """Whether to count the number of function evaluations."""

    stop_if_nan: bool
    """Whether to stop the evaluation when a value is NaN."""

    _database: Database
    """The database containing all the evaluations."""

    _evaluation_counter: EvaluationCounter
    """The counter of evaluations."""

    _output_evaluation_sequence: Iterable[Callable[[NumberArray], NumberArray]]
    """The execution sequence to compute an output value from an input value."""

    _gradient_name: str
    """The name of the gradient variable."""

    _jacobian_evaluation_sequence: Iterable[Callable[[NumberArray], NumberArray]]
    """The execution sequence to compute a Jacobian from an input value."""

    _n_calls: Value
    """The number of calls to :meth:`evaluate`."""

    _normalize_grad: Callable[[RealOrComplexArrayT], RealOrComplexArrayT]
    """The function to normalize an unnormalized gradient."""

    _unnormalize_grad: Callable[[RealOrComplexArrayT], RealOrComplexArrayT]
    """The function to unnormalize a normalized gradient."""

    _unnormalize_vect: Callable[
        [RealOrComplexArrayT, bool, bool, ndarray | None], RealOrComplexArrayT
    ]
    """The function to unnormalize a normalized vector of the design space."""

    __store_jacobian: bool
    """Whether to store the Jacobian matrices in the database."""

    def __init__(
        self,
        function: MDOFunction,
        output_evaluation_sequence: Iterable[Callable[[NumberArray], NumberArray]],
        jacobian_evaluation_sequence: Iterable[Callable[[NumberArray], NumberArray]],
        with_normalized_inputs: bool,
        database: Database | None,
        counter: EvaluationCounter,
        stop_if_nan: bool,
        design_space: DesignSpace,
        store_jacobian: bool = True,
        differentiation_method: ApproximationMode | None = None,
        **differentiation_method_options: Any,
    ):
        """
        Args:
            function: The original function.
            output_evaluation_sequence: The execution sequence
                to compute an output value from an input value.
            jacobian_evaluation_sequence: The execution sequence
                to compute a Jacobian from an input value.
            with_normalized_inputs: Whether the function expects normalized inputs.
            use_database: Whether to use the database to store and retrieve values.
            database: The database to store and retrieve the evaluations;
                if ``None``, do not use database.
            counter: The counter of evaluations.
            stop_if_nan: Whether the evaluation stops when a function returns ``NaN``.
            design_space: The design space on which to evaluate the function.
            store_jacobian: Whether to store the Jacobian matrices in the database.
            differentiation_method: The differentiation method to compute the Jacobian.
                If ``None``, use the original derivatives.
            **differentiation_method_options: The options of the differentiation method.

        """  # noqa: D205, D212, D415
        self._init_shared_memory_attrs_before()
        self._output_evaluation_sequence = output_evaluation_sequence
        self._jacobian_evaluation_sequence = jacobian_evaluation_sequence
        self.__store_jacobian = store_jacobian

        use_database = database is not None
        if use_database and with_normalized_inputs:
            compute_output = self._compute_output_db_norm
            compute_jacobian = self._compute_jacobian_db_norm
        elif use_database:
            compute_output = self._compute_output_db
            compute_jacobian = self._compute_jacobian_db
        else:
            compute_output = self._compute_output
            compute_jacobian = self._compute_jacobian

        self._gradient_name = Database.get_gradient_name(function.name)
        self._evaluation_counter = counter
        self.stop_if_nan = stop_if_nan
        self._database = database
        self._unnormalize_vect = design_space.unnormalize_vect
        self._normalize_grad = design_space.normalize_grad
        self._unnormalize_grad = design_space.unnormalize_grad
        if differentiation_method is not None:
            gradient_approximator = GradientApproximatorFactory().create(
                differentiation_method,
                self._compute_output,
                design_space=design_space,
                **differentiation_method_options,
            )
            self._jacobian_evaluation_sequence = (gradient_approximator.f_gradient,)

        super().__init__(
            compute_output,
            function.name,
            jac=compute_jacobian,
            f_type=function.f_type,
            expr=function.expr,
            input_names=function.input_names,
            dim=function.dim,
            output_names=function.output_names,
            force_real=function.force_real,
            special_repr=function.special_repr,
            original_name=function.original_name,
            with_normalized_inputs=with_normalized_inputs,
        )

    @MDOFunction.func.setter
    def func(self, f_pointer: WrappedFunctionType) -> None:  # noqa: D102
        if self.enable_statistics:
            self._n_calls.value = 0

        super(__class__, self.__class__).func.fset(self, f_pointer)

    def evaluate(self, x_vect: NumberArray) -> OutputType:  # noqa: D102
        if self.enable_statistics:
            # This evaluation is both multiprocess- and multithread-safe,
            # thanks to a locking process.
            with self._n_calls.get_lock():
                self._n_calls.value += 1

        return super().evaluate(x_vect)

    def _compute_output(self, input_value: NumberArray) -> NumberArray:
        """Compute the output value from an input value.

        Args:
            input_value: The input value.

        Returns:
            The output value.
        """
        for func in self._output_evaluation_sequence:
            input_value = func(input_value)

        return input_value

    def _compute_jacobian(self, input_value: NumberArray) -> NumberArray:
        """Compute the Jacobian from an input value.

        Args:
            input_value: The input value.

        Returns:
            The Jacobian.
        """
        for func in self._jacobian_evaluation_sequence:
            input_value = func(input_value)

        return input_value

    def _compute_output_db(self, input_value: NumberArray) -> NumberArray:
        """Compute the output value from a database and an input value.

        The database is used to store and retrieve output and Jacobian values.

        Args:
            input_value: The input value.

        Returns:
            The output value.
        """
        name = self.name
        self.check_function_output_includes_nan(input_value)
        database = self._database
        hashed_xu = database.get_hashable_ndarray(input_value)
        output_value = database.get_function_value(name, hashed_xu)
        if output_value is None:
            if (
                not database.get(hashed_xu)
                and self._evaluation_counter.maximum_is_reached
            ):
                raise MaxIterReachedException

            output_value = self._compute_output(input_value)
            self.check_function_output_includes_nan(
                output_value, self.stop_if_nan, name, input_value
            )
            database.store(hashed_xu, {name: output_value})

        return output_value

    def _compute_jacobian_db(self, input_value: NumberArray) -> NumberArray:
        """Compute the Jacobian from a database and an input value.

        The database is used to store and retrieve output and Jacobian values.

        Args:
            input_value: The input value.

        Returns:
            The Jacobian.
        """
        name = self._gradient_name
        self.check_function_output_includes_nan(input_value)
        database = self._database
        hashed_xu = database.get_hashable_ndarray(input_value)
        jacobian = database.get_function_value(name, hashed_xu)
        if jacobian is None:
            if (
                not database.get(hashed_xu)
                and self._evaluation_counter.maximum_is_reached
            ):
                raise MaxIterReachedException

            jacobian = self._compute_jacobian(input_value).real
            self.check_function_output_includes_nan(
                jacobian, self.stop_if_nan, name, input_value
            )
            if self.__store_jacobian:
                database.store(hashed_xu, {name: jacobian})

        return jacobian

    def _compute_output_db_norm(self, input_value: NumberArray) -> NumberArray:
        """Compute the output value from a database and a normalized input value.

        The database is used to store and retrieve output and Jacobian values.

        Args:
            input_value: The normalized input value.

        Returns:
            The output value.
        """
        self.check_function_output_includes_nan(input_value)
        xn_vect = input_value
        xu_vect = self._unnormalize_vect(xn_vect)
        database = self._database
        hashed_xu = database.get_hashable_ndarray(xu_vect)
        output_value = database.get_function_value(self.name, hashed_xu)
        if output_value is None:
            if (
                not database.get(hashed_xu)
                and self._evaluation_counter.maximum_is_reached
            ):
                raise MaxIterReachedException

            output_value = self._compute_output(xn_vect)
            self.check_function_output_includes_nan(
                output_value, self.stop_if_nan, self.name, xu_vect
            )
            database.store(hashed_xu, {self.name: output_value})

        return output_value

    def _compute_jacobian_db_norm(self, input_value: NumberArray) -> NumberArray:
        """Compute the Jacobian assisted by a database from a normalized input value.

        The database is used to store and retrieve output and Jacobian values.

        Args:
            input_value: The normalized input value.

        Returns:
            The Jacobian.
        """
        self.check_function_output_includes_nan(input_value)
        xn_vect = input_value
        xu_vect = self._unnormalize_vect(xn_vect)
        database = self._database
        hashed_xu = database.get_hashable_ndarray(xu_vect)
        jac_u = database.get_function_value(self._gradient_name, hashed_xu)
        if jac_u is None:
            if (
                not database.get(hashed_xu)
                and self._evaluation_counter.maximum_is_reached
            ):
                raise MaxIterReachedException

            jac_n = self._compute_jacobian(xn_vect)
            jac_u = self._unnormalize_grad(jac_n)
            self.check_function_output_includes_nan(
                jac_u.data,
                self.stop_if_nan,
                self._gradient_name,
                xu_vect,
            )
            if self.__store_jacobian:
                database.store(hashed_xu, {self._gradient_name: jac_u})
        else:
            jac_n = self._normalize_grad(jac_u)

        return jac_n.real

    @staticmethod
    def check_function_output_includes_nan(
        value: ndarray,
        stop_if_nan: bool = True,
        function_name: str = "",
        xu_vect: ndarray | None = None,
    ) -> None:
        """Check if an array contains a NaN value.

        Args:
            value: The array to be checked.
            stop_if_nan: Whether to stop if `value` contains a NaN.
            function_name: The name of the function.
                If empty,
                the arguments ``function_name`` and ``xu_vect`` are ignored.
            xu_vect: The point at which the function is evaluated.
                ``None`` if and only if ``function_name`` is empty.

        Raises:
            DesvarIsNan: If the value is a function input containing a NaN.
            FunctionIsNan: If the value is a function output containing a NaN.
        """
        if stop_if_nan and isnan(value).any():
            if function_name:
                msg = (
                    f"The function {function_name} contains a NaN value "
                    f"for x={xu_vect}."
                )
                raise FunctionIsNan(msg)

            msg = f"The input vector contains a NaN value: {value}."
            raise DesvarIsNan(msg)

    @property
    def n_calls(self) -> int:
        """The number of times the function has been evaluated.

        This count is both multiprocess- and multithread-safe, thanks to the locking
        process used by :meth:`.MDOFunction.evaluate`.
        """
        if self.enable_statistics:
            return self._n_calls.value
        return 0

    @n_calls.setter
    def n_calls(
        self,
        value: int,
    ) -> None:
        if not self.enable_statistics:
            msg = "The function counters are disabled."
            raise RuntimeError(msg)

        with self._n_calls.get_lock():
            self._n_calls.value = value

    def _init_shared_memory_attrs_before(self) -> None:
        """Initialize the shared attributes in multiprocessing."""
        self._n_calls = Value("i", 0)
