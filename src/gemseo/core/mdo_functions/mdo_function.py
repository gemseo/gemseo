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
# Copyright 2023 Capgemini Engineering
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                        documentation
#        :author: Francois Gallard, Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Base class to describe a function."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from collections.abc import Sequence
from collections.abc import Sized
from numbers import Complex
from numbers import Number
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Final
from typing import Union

from numpy import abs as np_abs
from numpy import ndarray
from numpy import ufunc
from numpy import where
from numpy.linalg import norm
from strenum import StrEnum

from gemseo.core.mdo_functions._operations import _AdditionFunctionMaker
from gemseo.core.mdo_functions._operations import _MultiplicationFunctionMaker
from gemseo.core.mdo_functions._operations import _OperationFunctionMaker
from gemseo.core.mdo_functions.not_implementable_callable import NotImplementedCallable
from gemseo.core.mdo_functions.set_pt_from_database import SetPtFromDatabase
from gemseo.typing import NumberArray
from gemseo.utils.compatibility.scipy import sparse_classes
from gemseo.utils.derivatives.approximation_modes import ApproximationMode
from gemseo.utils.derivatives.factory import GradientApproximatorFactory
from gemseo.utils.enumeration import merge_enums
from gemseo.utils.string_tools import pretty_str
from gemseo.utils.string_tools import repr_variable

if TYPE_CHECKING:
    from gemseo.algos.database import Database
    from gemseo.algos.design_space import DesignSpace

LOGGER = logging.getLogger(__name__)

OutputType = Union[NumberArray, Complex]
OperatorType = Union[Callable[[OutputType, OutputType], OutputType], ufunc]
WrappedFunctionType = Callable[[NumberArray], OutputType]
WrappedJacobianType = Callable[[NumberArray], NumberArray]


class MDOFunction:
    """The standard definition of an array-based function with algebraic operations.

    :class:`.MDOFunction` is the key class
    to define the objective function, the constraints and the observables
    of an :class:`.OptimizationProblem`.

    an :class:`.MDOFunction` is initialized from an optional callable and a name,
    e.g. ``func = MDOFunction(lambda x: 2*x, "my_function")``.

    .. note::

       The callable can be set to ``None``
       when the user does not want to use a callable
       but a database to browse for the output vector corresponding to an input vector
       (see :meth:`.MDOFunction.set_pt_from_database`).

    The following information can also be provided at initialization:

    - the type of the function,
      e.g. ``f_type="obj"`` if the function will be used as an objective
      (see :attr:`.MDOFunction.FunctionType`),
    - the function computing the Jacobian matrix,
      e.g. ``jac=lambda x: array([2.])``,
    - the literal expression to be used for the string representation of the object,
      e.g. ``expr="2*x"``,
    - the names of the inputs and outputs of the function,
      e.g. ``input_names=["x"]`` and ``output_names=["y"]``.

    .. warning::

       For the literal expression,
       do not use `"f(x) = 2*x"` nor `"f = 2*x"` but `"2*x"`.
       The other elements will be added automatically
       in the string representation of the function
       based on the name of the function and the names of its inputs.

    After the initialization,
    all of these arguments can be overloaded with setters,
    e.g. :attr:`.MDOFunction.input_names`.

    The original function and Jacobian function
    can be accessed with the properties :attr:`.MDOFunction.func`
    and :attr:`.MDOFunction.jac`.

    an :class:`.MDOFunction` is callable:
    ``output = func(array([3.])) # expected: array([6.])``.

    Elementary operations can be performed with :class:`.MDOFunction` instances:
    addition (``func = func1 + func2`` or ``func = func1 + offset``),
    subtraction (``func = func1 - func2`` or ``func = func1 - offset``),
    multiplication (``func = func1 * func2`` or ``func = func1 * factor``)
    and opposite  (``func = -func1``).
    It is also possible to build an :class:`.MDOFunction`
    as a concatenation of :class:`.MDOFunction` objects:
    ``func = MDOFunction.concatenate([func1, func2, func3], "my_func_123"``).

    Moreover, an :class:`.MDOFunction` can be approximated
    with either a first-order or second-order Taylor polynomial at a given input vector,
    using respectively :meth:`.MDOFunction.linear_approximation`
    and :meth:`quadratic_approx`;
    such an approximation is also an :class:`.MDOFunction`.

    Lastly, the user can check the Jacobian function by means of approximation methods
    (see :meth:`.MDOFunction.check_grad`).
    """

    class ConstraintType(StrEnum):
        """The type of constraint."""

        EQ = "eq"
        """The type of function for equality constraint."""

        INEQ = "ineq"
        """The type of function for inequality constraint."""

    class _FunctionType(StrEnum):
        """The type of function complementary to the constraints."""

        OBJ = "obj"
        """The type of function for objective."""

        OBS = "obs"
        """The type of function for observable."""

        NONE = ""
        """The type of function is not set."""

    FunctionType = merge_enums("FunctionType", StrEnum, _FunctionType, ConstraintType)

    ApproximationMode = ApproximationMode

    DICT_REPR_ATTR: ClassVar[list[str]] = [
        "name",
        "f_type",
        "expr",
        "input_names",
        "dim",
        "special_repr",
        "output_names",
    ]
    """The names of the attributes to be serialized."""

    DEFAULT_BASE_INPUT_NAME: str = "x"
    """The default base name for the inputs."""

    COEFF_FORMAT_1D: str = "{:.2e}"
    """The format to be applied to a number when represented in a vector."""
    # ensure that coefficients strings have same length

    COEFF_FORMAT_ND: str = "{: .2e}"
    """The format to be applied to a number when represented in a matrix."""
    # ensure that coefficients strings have same length

    # N.B. the space character ensures same length whatever the sign of the coefficient

    expr: str
    """The expression of the function, e.g. `"2*x"`."""

    f_type: FunctionType
    """The type of the function."""

    force_real: bool
    """Whether to cast the results to real value."""

    has_default_name: bool
    """Whether the name has been set with a default value."""

    last_eval: OutputType | None
    """The value of the function output at the last evaluation.

    ``None`` if it has not yet been evaluated.
    """

    name: str
    """The name of the function."""

    original: MDOFunction
    """The function before preprocessing by the :class:`.OptimizationProblem."""

    special_repr: str
    """The string representation of the function overloading its default string ones."""

    _dim: int
    """The dimension of the output space of the function."""

    _func: WrappedFunctionType
    """The function to be evaluated from a given input vector."""

    _jac: WrappedJacobianType
    """The Jacobian function to be evaluated from a given input vector."""

    _input_names: list[str]
    """The names of the inputs of the function."""

    _output_names: list[str]
    """The names of the outputs of the function."""

    __original_name: str
    """The original name of the function.

    By default, it is the same as :attr:`.name`.
    When the value of :attr:`.name` changes,
    :attr:`.original_name` stores its former value.
    """

    __INPUT_NAME_PATTERN: Final[str] = "x"
    """The pattern to define a variable name, as ``"x[1]"``."""

    def __init__(
        self,
        func: WrappedFunctionType | None,
        name: str,
        f_type: FunctionType = FunctionType.NONE,
        jac: WrappedJacobianType | None = None,
        expr: str = "",
        input_names: Iterable[str] = (),
        dim: int = 0,
        output_names: Iterable[str] = (),
        force_real: bool = False,
        special_repr: str = "",
        original_name: str = "",
        with_normalized_inputs: bool = False,
    ) -> None:
        """
        Args:
            func: The original function to be actually called.
                If ``None``, the function will not have an original function.
            name: The name of the function.
            f_type: The type of the function.
            jac: The original Jacobian function to be actually called.
                If ``None``, the function will not have an original Jacobian function.
            expr: The expression of the function, e.g. `"2*x"`, if any.
            input_names: The names of the inputs of the function.
                If empty, the inputs of the function will have no names.
            dim: The dimension of the output space of the function.
                If 0, the dimension of the output space of the function
                will be deduced from the evaluation of the function.
            output_names: The names of the outputs of the function.
                If empty, the outputs of the function will have no names.
            force_real: Whether to cast the output values to real.
            special_repr: The string representation of the function.
                If empty, use :meth:`.default_repr`.
            original_name: The original name of the function.
                If empty, use the same name than the ``name`` input.
            with_normalized_inputs: Whether the function expects normalized inputs.
        """  # noqa: D205, D212, D415
        super().__init__()
        self.__original_name = original_name or name
        self.name = name
        self.func = func
        self.jac = jac
        self.f_type = f_type
        self.expr = expr
        self.input_names = input_names
        self.dim = dim
        self.output_names = output_names
        self.last_eval = None
        self.force_real = force_real
        self.special_repr = special_repr or ""
        self.has_default_name = bool(self.name)
        self.__expects_normalized_inputs = with_normalized_inputs
        self.original = self

    @property
    def original_name(self) -> str:
        """The original name of the function."""
        return self.__original_name

    @property
    def func(self) -> WrappedFunctionType:
        """The wrapped function."""
        return self._func

    @func.setter
    def func(self, f_pointer: WrappedFunctionType | None) -> None:
        if f_pointer is None:
            self._func = NotImplementedCallable(self.name, "value")
        else:
            self._func = f_pointer

    def evaluate(self, x_vect: NumberArray) -> OutputType:
        """Evaluate the function and store the output value in :attr:`.last_eval`.

        When the output dimension :attr:`.dim` is not defined,
        it is inferred on the first evaluation.

        Args:
            x_vect: The input value of the function.

        Returns:
            Either the raw output value
            or its real part when :attr:`.force_real` is `True`.
        """
        output_value = self.last_eval = self._func(x_vect)

        if self.force_real:
            output_value = output_value.real

        if not self.dim:
            self.dim = output_value.size if isinstance(output_value, ndarray) else 1

        return output_value

    @property
    def jac(self) -> WrappedJacobianType:
        """The Jacobian function to be evaluated from a given input vector."""
        return self._jac

    @jac.setter
    def jac(self, jac: WrappedJacobianType | None) -> None:
        self._jac = jac or NotImplementedCallable(self.name, "Jacobian")

    @property
    def input_names(self) -> list[str]:
        """The names of the inputs of the function.

        Use a copy of the original names.
        """
        return self._input_names

    @input_names.setter
    def input_names(self, input_names: Iterable[str]) -> None:
        self._input_names = list(input_names)

    @property
    def output_names(self) -> list[str]:
        """The names of the outputs of the function.

        Use a copy of the original names.
        """
        return self._output_names

    @output_names.setter
    def output_names(self, output_names: Iterable[str]) -> None:
        self._output_names = list(output_names)

    def is_constraint(self) -> bool:
        """Check if the function is a constraint.

        The constraint type is either ``"eq"`` or "``ineq"``.

        Returns:
            Whether the function is a constraint.
        """
        return self.f_type in set(self.ConstraintType)

    def __repr__(self) -> str:
        return self.special_repr or self.default_repr

    @property
    def default_repr(self) -> str:
        """The default string representation of the function."""
        if self.is_constraint():
            if self.expr:
                left = self.expr
            else:
                name = "#".join(self.output_names) or self.name
                if self.input_names:
                    left = f"{name}({pretty_str(self.input_names, sort=False)})"
                else:
                    left = f"{name}"

            sign = "==" if self.f_type == self.ConstraintType.EQ else "<="
            return f"{left} {sign} 0.0"

        if self.input_names:
            strings = [f"{self.name}({pretty_str(self.input_names, sort=False)})"]
        else:
            strings = [self.name]

        if not self.expr or strings[-1] == self.expr:
            return "".join(strings)

        strings.append(" = ")
        prefix = ""
        for index, line in enumerate(self.expr.split("\n")):
            strings.append(f"{prefix}{line}\n")
            if index == 0:
                prefix = " " * (sum(len(string) for string in strings) + 3)

        strings[-1] = strings[-1][:-1]
        return "".join(strings)

    @property
    def has_jac(self) -> bool:
        """Whether the function has an implemented Jacobian function."""
        return self.jac is not None and not isinstance(
            self._jac, NotImplementedCallable
        )

    def __add__(self, other: MDOFunction | Number) -> MDOFunction:
        """Operator defining the sum of the function and another one.

        This operator supports automatic differentiation
        if both functions have an implemented Jacobian function.

        Args:
            other: The other function.

        Returns:
            The sum of the function and the other one.
        """
        return _AdditionFunctionMaker(MDOFunction, self, other).function

    def __sub__(self, other: MDOFunction | Number) -> MDOFunction:
        """Operator defining the difference of the function and another one.

        This operator supports automatic differentiation
        if both functions have an implemented Jacobian function.

        Args:
            other: The other function.

        Returns:
            The difference of the function and the other one.
        """
        return _AdditionFunctionMaker(MDOFunction, self, other, inverse=True).function

    def _min_pt(self, x_vect: NumberArray) -> NumberArray:
        """Evaluate the function and return its opposite value.

        Args:
            x_vect: The value of the inputs of the function.

        Returns:
            The opposite of the value of the outputs of the function.
        """
        return -self.evaluate(x_vect)

    def _min_jac(self, x_vect: NumberArray) -> NumberArray:
        """Evaluate the Jacobian function and return its opposite value.

        Args:
            x_vect: The value of the inputs of the Jacobian function.

        Returns:
            The opposite of the value of the Jacobian function.
        """
        return -self.jac(x_vect)

    def __neg__(self) -> MDOFunction:
        """Operator defining the opposite of the function.

        This operator supports automatic differentiation
        if the function has an implemented Jacobian function.

        Returns:
            The opposite of the function.
        """
        jac = self._min_jac if self.has_jac else None

        name = f"-{self.name}"
        if self.expr:
            expr = f"-({self.expr})"
        elif self.input_names:
            expr = f"{name}({pretty_str(self.input_names, sort=False)})"
        else:
            expr = name

        return MDOFunction(
            self._min_pt,
            name,
            jac=jac,
            input_names=self.input_names,
            f_type=self.f_type,
            dim=self.dim,
            output_names=self.output_names,
            expr=expr,
            original_name=self.original_name,
            special_repr=f"-({self.special_repr})" if self.special_repr else "",
        )

    def __truediv__(self, other: MDOFunction | OutputType) -> MDOFunction:
        """Define the division operation for MDOFunction.

        This operation supports automatic differentiation
        if the different functions have an implemented Jacobian function.

        Args:
            other: The function or number to divide by.

        Returns:
            A function dividing a function by another function or a number.
        """
        return _MultiplicationFunctionMaker(
            MDOFunction, self, other, inverse=True
        ).function

    def __mul__(self, other: MDOFunction | OutputType) -> MDOFunction:
        """Define the multiplication operation for MDOFunction.

        This operation supports automatic differentiation
        if the different functions have an implemented Jacobian function.

        Args:
            other: The function or number to multiply by.

        Returns:
            A function multiplying a function by another function or a number.
        """
        return _MultiplicationFunctionMaker(MDOFunction, self, other).function

    def offset(self, value: OutputType) -> MDOFunction:
        """Add an offset value to the function.

        Args:
            value: The offset value.

        Returns:
            The offset function.
        """
        operator = "+"
        if isinstance(value, Sized):
            second_operand = "offset"
        elif value >= 0:
            second_operand = value
        else:
            operator = "-"
            second_operand = -value

        function = self + value
        name = f"{self.name}({pretty_str(self.input_names, sort=False)})"
        function.name = _OperationFunctionMaker.get_string_representation(
            self.name, operator, second_operand, True
        )
        function.expr = _OperationFunctionMaker.get_string_representation(
            self.expr or name, operator, second_operand
        )
        function.special_repr = _OperationFunctionMaker.get_string_representation(
            self.special_repr or name, operator, second_operand
        )
        return function

    def check_grad(
        self,
        x_vect: NumberArray,
        approximation_mode: ApproximationMode = ApproximationMode.FINITE_DIFFERENCES,
        step: float = 1e-6,
        error_max: float = 1e-8,
    ) -> None:
        """Check the gradients of the function.

        Args:
            x_vect: The vector at which the function is checked.
            approximation_mode: The approximation mode.
            step: The step for the approximation of the gradients.
            error_max: The maximum value of the error.

        Raises:
            ValueError: Either if the approximation method is unknown,
                if the shapes of
                the analytical and approximated Jacobian matrices
                are inconsistent
                or if the analytical gradients are wrong.
        """
        gradient_approximator = GradientApproximatorFactory().create(
            approximation_mode, self.evaluate, step=step
        )

        approximation = gradient_approximator.f_gradient(x_vect).real
        reference = self._jac(x_vect).real

        if isinstance(reference, sparse_classes):
            reference = reference.todense()

        if approximation.shape != reference.shape:
            approximation_is_1d = approximation.ndim == 1 or approximation.shape[0] == 1
            reference_is_1d = reference.ndim == 1 or reference.shape[0] == 1
            shapes_are_1d = approximation_is_1d and reference_is_1d
            flatten_diff = reference.flatten().shape != approximation.flatten().shape
            if not shapes_are_1d or (shapes_are_1d and flatten_diff):
                msg = (
                    f"The Jacobian matrix computed by {self} has a wrong shape; "
                    f"got: {reference.shape} while expected: {approximation.shape}."
                )
                raise ValueError(msg)

        if self.rel_err(reference, approximation, error_max) > error_max:
            LOGGER.error("The Jacobian matrix computed by %s is wrong.", self)
            LOGGER.error("Error =\n%s", self.filt_0(reference - approximation))
            LOGGER.error("Analytic jacobian=\n%s", self.filt_0(reference))
            LOGGER.error("Approximate step gradient=\n%s", self.filt_0(approximation))
            msg = f"The Jacobian matrix computed by {self} is wrong."
            raise ValueError(msg)

    @staticmethod
    def rel_err(a_vect: NumberArray, b_vect: NumberArray, error_max: float) -> float:
        """Compute the 2-norm of the difference between two vectors.

        Normalize it with the 2-norm of the reference vector
        if the latter is greater than the maximal error.

        Args:
            a_vect: A first vector.
            b_vect: A second vector, used as a reference.
            error_max: The maximum value of the error.

        Returns:
            The difference between two vectors,
            normalized if required.
        """
        if norm(b_vect) > error_max:
            return norm(a_vect - b_vect) / norm(b_vect)
        return norm(a_vect - b_vect)

    @staticmethod
    def filt_0(arr: NumberArray, floor_value: float = 1e-6) -> NumberArray:
        """Set the non-significant components of a vector to zero.

        The component of a vector is non-significant
        if its absolute value is lower than a threshold.

        Args:
            arr: The original vector.
            floor_value: The threshold.

        Returns:
            The original vector
            whose non-significant components have been set at zero.
        """
        return where(np_abs(arr) < floor_value, 0.0, arr)

    def to_dict(self) -> dict[str, str | int | list[str]]:
        """Create a dictionary representation of the function.

        This is used for serialization.
        The pointers to the functions are removed.

        Returns:
            Some attributes of the function indexed by their names.
            See :attr:`.MDOFunction.DICT_REPR_ATTR`.
        """
        repr_dict = {}
        for attr_name in self.DICT_REPR_ATTR:
            attr = getattr(self, attr_name)
            if attr is not None:
                repr_dict[attr_name] = attr
        return repr_dict

    @staticmethod
    def init_from_dict_repr(**attributes: Any) -> MDOFunction:
        """Initialize a new function.

        This is typically used for deserialization.

        Args:
            **attributes: The values of the serializable attributes
                listed in :attr:`.MDOFunction.DICT_REPR_ATTR`.

        Returns:
            A function initialized from the provided data.

        Raises:
            ValueError: If the name of an argument is not in
                :attr:`.MDOFunction.DICT_REPR_ATTR`.
        """
        serializable_attributes = MDOFunction.DICT_REPR_ATTR
        args = attributes.pop("args", None)
        if args is not None:
            attributes["input_names"] = args
        for attribute in attributes:
            if attribute not in serializable_attributes:
                msg = (
                    f"Cannot initialize MDOFunction attribute: {attribute}, "
                    f"allowed ones are: {pretty_str(serializable_attributes)}."
                )
                raise ValueError(msg)
        return MDOFunction(func=None, **attributes)

    def set_pt_from_database(
        self,
        database: Database,
        design_space: DesignSpace,
        normalize: bool = False,
        jac: bool = True,
        x_tolerance: float = 1e-10,
    ) -> None:
        """Set the original function and Jacobian function from a database.

        For a given input vector,
        the method :meth:`.MDOFunction.func` will return
        either the output vector stored in the database
        if the input vector is present
        or ``None``.
        The same for the method :meth:`.MDOFunction.jac`.

        Args:
            database: The database to read.
            design_space: The design space used for normalization.
            normalize: If ``True``,
                the values of the inputs are unnormalized before call.
            jac: If ``True``, a Jacobian pointer is also generated.
            x_tolerance: The tolerance on the distance between inputs.
        """
        SetPtFromDatabase(database, design_space, self, normalize, jac, x_tolerance)

    @classmethod
    def generate_input_names(
        cls, input_dim: int, input_names: Sequence[str] = ()
    ) -> Sequence[str]:
        """Generate the names of the inputs of the function.

        Args:
            input_dim: The dimension of the input space of the function.
            input_names: The initial names of the inputs of the function.
                If the number of names matches the dimension of the input space,
                use these names as is.
                Otherwise,
                if there is only one name,
                e.g. ``["var"]``,
                and if the dimension of the input space is equal to 3,
                use this name as a base name
                and generate the names of the inputs,
                e.g. ``["var[0]", "var[1]", "var[2]"]``.
                If empty,
                use ``"x"`` as a base name and generate the names of the inputs,
                i.e. ``["x[0]", "x[1]", "x[2]"]``.

        Returns:
            The names of the inputs of the function.
        """
        n_input_names = len(input_names)
        if n_input_names == input_dim:
            return input_names

        name = input_names[0] if n_input_names == 1 else cls.__INPUT_NAME_PATTERN
        return [repr_variable(name, i, input_dim) for i in range(input_dim)]

    @property
    def expects_normalized_inputs(self) -> bool:
        """Whether the function expects normalized inputs."""
        return self.__expects_normalized_inputs

    @expects_normalized_inputs.setter
    def expects_normalized_inputs(self, value: bool) -> None:
        self.__expects_normalized_inputs = value

    def get_indexed_name(self, index: int) -> str:
        """Return the name of function component.

        Args:
            index: The index of the function component.

        Returns:
            The name of the function component.
        """
        return repr_variable(self.name, index, self.dim)
