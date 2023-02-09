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
#                        documentation
#        :author: Francois Gallard, Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Base class to describe a function."""
from __future__ import annotations

import logging
import pickle
from multiprocessing import Value
from numbers import Number
from pathlib import Path
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Iterable
from typing import Sequence
from typing import Sized
from typing import TYPE_CHECKING
from typing import Union

from numpy import abs as np_abs
from numpy import ndarray
from numpy import ufunc
from numpy import where
from numpy.linalg import norm
from numpy.typing import NDArray

from gemseo.algos.database import Database
from gemseo.algos.design_space import DesignSpace
from gemseo.core.mdofunctions._operations import _AdditionFunctionMaker
from gemseo.core.mdofunctions._operations import _MultiplicationFunctionMaker
from gemseo.core.mdofunctions.not_implementable_callable import NotImplementedCallable
from gemseo.core.mdofunctions.set_pt_from_database import SetPtFromDatabase
from gemseo.utils.derivatives.complex_step import ComplexStep
from gemseo.utils.derivatives.finite_differences import FirstOrderFD
from gemseo.utils.python_compatibility import Final
from gemseo.utils.string_tools import pretty_str

if TYPE_CHECKING:
    from gemseo.core.mdofunctions.mdo_linear_function import MDOLinearFunction
    from gemseo.core.mdofunctions.mdo_quadratic_function import MDOQuadraticFunction
    from gemseo.core.mdofunctions.convex_linear_approx import ConvexLinearApprox
    from gemseo.core.mdofunctions.concatenate import Concatenate
    from gemseo.core.mdofunctions.function_restriction import FunctionRestriction

LOGGER = logging.getLogger(__name__)

ArrayType = NDArray[Number]
OperandType = Union[ArrayType, Number]
OperatorType = Union[Callable[[OperandType, OperandType], OperandType], ufunc]
OutputType = Union[ArrayType, Number]
WrappedFunctionType = Callable[[ArrayType], OutputType]
WrappedJacobianType = Callable[[ArrayType], ArrayType]


class MDOFunction:
    """The standard definition of an array-based function with algebraic operations.

    :class:`.MDOFunction` is the key class
    to define the objective function, the constraints and the observables
    of an :class:`.OptimizationProblem`.

    A :class:`.MDOFunction` is initialized from an optional callable and a name,
    e.g. :code:`func = MDOFunction(lambda x: 2*x, "my_function")`.

    .. note::

       The callable can be set to :code:`None`
       when the user does not want to use a callable
       but a database to browse for the output vector corresponding to an input vector
       (see :meth:`.MDOFunction.set_pt_from_database`).

    The following information can also be provided at initialization:

    - the type of the function,
      e.g. :code:`f_type="obj"` if the function will be used as an objective
      (see :attr:`.MDOFunction.AVAILABLE_TYPES` for the available types),
    - the function computing the Jacobian matrix,
      e.g. :code:`jac=lambda x: array([2.])`,
    - the literal expression to be used for the string representation of the object,
      e.g. :code:`expr="2*x"`,
    - the names of the inputs and outputs of the function,
      e.g. :code:`args=["x"]` and :code:`outvars=["y"]`.

    .. warning::

       For the literal expression,
       do not use `"f(x) = 2*x"` nor `"f = 2*x"` but `"2*x"`.
       The other elements will be added automatically
       in the string representation of the function
       based on the name of the function and the names of its inputs.

    After the initialization,
    all of these arguments can be overloaded with setters,
    e.g. :attr:`.MDOFunction.args`.

    The original function and Jacobian function
    can be accessed with the properties :attr:`.MDOFunction.func`
    and :attr:`.MDOFunction.jac`.

    A :class:`.MDOFunction` is callable:
    :code:`output = func(array([3.])) # expected: array([6.])`.

    Elementary operations can be performed with :class:`.MDOFunction` instances:
    addition (:code:`func = func1 + func2` or :code:`func = func1 + offset`),
    subtraction (:code:`func = func1 - func2` or :code:`func = func1 - offset`),
    multiplication (:code:`func = func1 * func2` or :code:`func = func1 * factor`)
    and opposite  (:code:`func = -func1`).
    It is also possible to build a :class:`.MDOFunction`
    as a concatenation of :class:`.MDOFunction` objects:
    :code:`func = MDOFunction.concatenate([func1, func2, func3], "my_func_123"`).

    Moreover, a :class:`.MDOFunction` can be approximated
    with either a first-order or second-order Taylor polynomial at a given input vector,
    using respectively :meth:`.MDOFunction.linear_approximation`
    and :meth:`quadratic_approx`;
    such an approximation is also a :class:`.MDOFunction`.

    Lastly, the user can check the Jacobian function by means of approximation methods
    (see :meth:`.MDOFunction.check_grad`).
    """

    TYPE_OBJ: str = "obj"
    """The type of function for objective."""

    TYPE_EQ: str = "eq"
    """The type of function for equality constraint."""

    TYPE_INEQ: str = "ineq"
    """The type of function for inequality constraint."""

    TYPE_OBS: str = "obs"
    """The type of function for observable."""

    __CONSTRAINT_TYPES: Final[tuple[str]] = (TYPE_INEQ, TYPE_EQ)
    """The different types of constraint."""

    AVAILABLE_TYPES: list[str] = [TYPE_OBJ, TYPE_EQ, TYPE_INEQ, TYPE_OBS]
    """The available types of function."""

    DICT_REPR_ATTR: list[str] = [
        "name",
        "f_type",
        "expr",
        "args",
        "dim",
        "special_repr",
    ]
    """The names of the attributes to be serialized."""

    DEFAULT_ARGS_BASE: str = "x"
    """The default name base for the inputs."""

    INDEX_PREFIX: str = "!"
    """The character used to separate a name base and a prefix, e.g. ``"x!1``."""

    COEFF_FORMAT_1D: str = "{:.2e}"
    """The format to be applied to a number when represented in a vector."""
    # ensure that coefficients strings have same length

    COEFF_FORMAT_ND: str = "{: .2e}"
    """The format to be applied to a number when represented in a matrix."""
    # ensure that coefficients strings have same length

    # N.B. the space character ensures same length whatever the sign of the coefficient

    activate_counters: ClassVar[bool] = True
    """Whether to count the number of function evaluations."""

    has_default_name: bool
    """Whether the name has been set with a default value."""

    last_eval: OutputType | None
    """The value of the function output at the last evaluation.

    ``None`` if it has not yet been evaluated.
    """

    force_real: bool
    """Whether to cast the results to real value."""

    special_repr: str
    """The string representation of the function overloading its default string ones."""

    _n_calls: Value
    """The number of times that the function has been evaluated."""

    _f_type: str
    """The type of the function, among :attr:`.MDOFunction.AVAILABLE_TYPES`."""

    _func: WrappedFunctionType
    """The function to be evaluated from a given input vector."""

    _jac: WrappedJacobianType
    """The Jacobian function to be evaluated from a given input vector."""

    _name: str
    """The name of the function."""

    _args: list[str]
    """The names of the inputs of the function."""

    _expr: str
    """The expression of the function, e.g. `"2*x"`."""

    _dim: int
    """The dimension of the output space of the function."""

    _outvars: list[str]
    """The names of the outputs of the function."""

    _ATTR_NOT_TO_SERIALIZE: tuple[str] = ("_n_calls",)
    """The attributes that shall be skipped at serialization.

    Private attributes shall be written following name mangling conventions:
    ``_ClassName__attribute_name``. Subclasses must expand this class attribute if
    needed.
    """

    __INPUT_NAME_PATTERN: Final[str] = "x"
    """The pattern to define a variable name, as ``"x!1"``."""

    def __init__(
        self,
        func: WrappedFunctionType | None,
        name: str,
        f_type: str = "",
        jac: WrappedJacobianType | None = None,
        expr: str = "",
        args: Iterable[str] | None = None,
        dim: int = 0,
        outvars: Iterable[str] | None = None,
        force_real: bool = False,
        special_repr: str = "",
    ) -> None:
        """
        Args:
            func: The original function to be actually called.
                If ``None``, the function will not have an original function.
            name: The name of the function.
            f_type: The type of the function among :attr:`.MDOFunction.AVAILABLE_TYPES`
                if any.
            jac: The original Jacobian function to be actually called.
                If ``None``, the function will not have an original Jacobian function.
            expr: The expression of the function, e.g. `"2*x"`, if any.
            args: The names of the inputs of the function.
                If ``None``, the inputs of the function will have no names.
            dim: The dimension of the output space of the function.
                If 0, the dimension of the output space of the function
                will be deduced from the evaluation of the function.
            outvars: The names of the outputs of the function.
                If ``None``, the outputs of the function will have no names.
            force_real: Whether to cast the output values to real.
            special_repr: The string representation of the function.
                If empty, use :meth:`.default_repr`.
        """  # noqa: D205, D212, D415
        super().__init__()

        # Initialize attributes
        self._f_type = ""
        self._func = NotImplementedCallable()
        self._jac = NotImplementedCallable()
        self._name = ""
        self._args = []
        self._expr = ""
        self._dim = 0
        # TODO: API: rename to _output_variables
        self._outvars = []
        self._init_shared_attrs()
        # Use setters to check values
        self.func = func
        self.jac = jac
        self.name = name
        self.f_type = f_type
        self.expr = expr
        self.args = args
        self.dim = dim
        # TODO: API: rename to output_variables
        self.outvars = outvars
        self.last_eval = None
        self.force_real = force_real
        self.special_repr = special_repr or ""
        self.has_default_name = False

    @property
    def n_calls(self) -> int:
        """The number of times the function has been evaluated.

        This count is both multiprocess- and multithread-safe, thanks to the locking
        process used by :meth:`.MDOFunction.evaluate`.
        """
        if self.activate_counters:
            return self._n_calls.value

    @n_calls.setter
    def n_calls(
        self,
        value: int,
    ) -> None:
        if not self.activate_counters:
            raise RuntimeError("The function counters are disabled.")

        with self._n_calls.get_lock():
            self._n_calls.value = value

    @property
    def func(self) -> WrappedFunctionType:
        """The function to be evaluated from a given input vector."""
        return self.__counted_f

    def __counted_f(self, x_vect: ArrayType) -> OutputType:
        """Evaluate the function and store the result in :attr:`.MDOFunction.last_eval`.

        This evaluation is both multiprocess- and multithread-safe,
        thanks to a locking process.

        Args:
            x_vect: The value of the inputs of the function.

        Returns:
            The value of the outputs of the function.
        """
        if self.activate_counters:
            with self._n_calls.get_lock():
                self._n_calls.value += 1

        self.last_eval = self._func(x_vect)
        return self.last_eval

    @func.setter
    def func(self, f_pointer: WrappedFunctionType) -> None:
        if self.activate_counters:
            self._n_calls.value = 0

        self._func = f_pointer or NotImplementedCallable()

    def serialize(self, file_path: str | Path) -> None:
        """Serialize the function and store it in a file.

        Args:
            file_path: The path to the file to store the function.
        """
        with Path(file_path).open("wb") as outfobj:
            pickler = pickle.Pickler(outfobj, protocol=2)
            pickler.dump(self)

    @staticmethod
    def deserialize(file_path: str | Path) -> MDOFunction:
        """Deserialize a function from a file.

        Args:
            file_path: The path to the file containing the function.

        Returns:
            The function instance.
        """
        with Path(file_path).open("rb") as file_:
            obj = pickle.Unpickler(file_).load()

        return obj

    def __getstate__(self) -> dict[str, Any]:
        """Used by pickle to define what to serialize.

        Returns:
            The attributes to be serialized.
        """
        state = {}
        for attribute_name in list(self.__dict__.keys() - self._ATTR_NOT_TO_SERIALIZE):
            attribute_value = self.__dict__[attribute_name]

            # At this point, there are no Synchronized attributes in MDOFunction or its
            # child classes other than _n_calls, which is not serialized.
            # If a Synchronized attribute is added in the future, the following check
            # (and its counterpart in __setstate__) shall be uncommented.

            # if isinstance(attribute_value, Synchronized):
            #     # Don´t serialize shared memory object,
            #     # this is meaningless, save the value instead
            #     attribute_value = attribute_value.value

            state[attribute_name] = attribute_value

        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._init_shared_attrs()
        for attribute_name, attribute_value in state.items():
            # At this point, there are no Synchronized attributes in MDOFunction or its
            # child classes other than _n_calls, which is not serialized.
            # If a Synchronized attribute is added in the future, the following check
            # (and its counterpart in __getstate__) shall be uncommented.

            # if isinstance(attribute_value, Synchronized):
            #     # Don´t serialize shared memory object,
            #     # this is meaningless, save the value instead
            #     attribute_value = attribute_value.value

            self.__dict__[attribute_name] = attribute_value

    def _init_shared_attrs(self) -> None:
        """Initialize the shared attributes in multiprocessing."""
        self._n_calls = Value("i", 0)

    def __call__(self, x_vect: ArrayType) -> OutputType:
        """Evaluate the function.

        This method can cast the result to real value
        according to the value of the attribute :attr:`.MDOFunction.force_real`.

        Args:
            x_vect: The value of the inputs of the function.

        Returns:
            The value of the outputs of the function.
        """
        return self.evaluate(x_vect)

    def evaluate(self, x_vect: ArrayType) -> OutputType:
        """Evaluate the function and store the dimension of the output space.

        Args:
            x_vect: The value of the inputs of the function.

        Returns:
            The value of the output of the function.
        """
        if self.activate_counters:
            val = self.__counted_f(x_vect)
        else:
            # we duplicate the logic here of __counted_f on purpose for performance
            val = self._func(x_vect)
            self.last_eval = val

        if self.force_real:
            val = val.real

        if not self.dim:
            self.dim = val.size if isinstance(val, ndarray) else 1
        return val

    @property
    def name(self) -> str:
        """The name of the function."""
        return self._name

    @name.setter
    def name(self, name: str | None) -> None:
        self._name = name or ""

    @property
    def f_type(self) -> str:
        """The type of the function, among :attr:`.MDOFunction.AVAILABLE_TYPES`."""
        return self._f_type

    @f_type.setter
    def f_type(
        self,
        f_type: str,
    ) -> None:
        if f_type not in [None, ""] and f_type not in self.AVAILABLE_TYPES:
            raise ValueError(
                "MDOFunction type must be among {}; got {} instead.".format(
                    self.AVAILABLE_TYPES, f_type
                )
            )
        self._f_type = f_type or ""

    @property
    def jac(self) -> WrappedJacobianType:
        """The Jacobian function to be evaluated from a given input vector."""
        return self._jac

    @jac.setter
    def jac(self, jac: WrappedJacobianType | None) -> None:
        self._jac = jac or NotImplementedCallable()

    @property
    def args(self) -> list[str]:
        """The names of the inputs of the function.

        Use a copy of the original names.
        """
        return self._args

    @args.setter
    def args(self, args: Iterable[str] | None) -> None:
        if args is None:
            self._args = []
        else:
            self._args = list(args)

    @property
    def expr(self) -> str:
        """The expression of the function, e.g. `"2*x"`."""
        return self._expr

    @expr.setter
    def expr(self, expr: str) -> None:
        self._expr = expr or ""

    @property
    def dim(self) -> int:
        """The dimension of the output space of the function."""
        return self._dim

    @dim.setter
    def dim(self, dim: int | None) -> None:
        self._dim = dim or 0

    @property
    def outvars(self) -> list[str]:
        """The names of the outputs of the function.

        Use a copy of the original names.
        """
        return self._outvars

    @outvars.setter
    def outvars(self, outvars: Iterable[str]) -> None:
        if outvars is None:
            self._outvars = []
        else:
            self._outvars = list(outvars)

    def is_constraint(self) -> bool:
        """Check if the function is a constraint.

        The type of a constraint function is either 'eq' or 'ineq'.

        Returns:
            Whether the function is a constraint.
        """
        return self.f_type in self.__CONSTRAINT_TYPES

    def __repr__(self) -> str:
        return self.special_repr or self.default_repr

    @property
    def default_repr(self) -> str:
        """The default string representation of the function."""
        if self.is_constraint():
            if self.expr:
                left = self.expr
            else:
                name = "#".join(self.outvars) or self.name
                left = f"{name}({pretty_str(self.args, sort=False)})"

            sign = "==" if self.f_type == self.TYPE_EQ else "<="
            return f"{left} {sign} 0.0"

        strings = [self.name]
        if self.has_args():
            strings.append(f"({pretty_str(self.args, sort=False)})")

        if not self.has_expr():
            return "".join(strings)

        strings.append(" = ")
        prefix = ""
        for index, line in enumerate(self.expr.split("\n")):
            strings.append(f"{prefix}{line}\n")
            if index == 0:
                prefix = " " * (sum(len(string) for string in strings) + 3)

        strings[-1] = strings[-1][:-1]
        return "".join(strings)

    def has_jac(self) -> bool:
        """Check if the function has an implemented Jacobian function.

        Returns:
            Whether the function has an implemented Jacobian function.
        """
        return self.jac is not None and not isinstance(
            self._jac, NotImplementedCallable
        )

    def has_dim(self) -> bool:
        """Check if the dimension of the output space of the function is defined.

        Returns:
            Whether the dimension of the output space of the function is defined.
        """
        return bool(self.dim)

    def has_outvars(self) -> bool:
        """Check if the outputs of the function have names.

        Returns:
            Whether the outputs of the function have names.
        """
        return bool(self.outvars)

    def has_expr(self) -> bool:
        """Check if the function has an expression.

        Returns:
            Whether the function has an expression.
        """
        return bool(self.expr)

    def has_args(self) -> bool:
        """Check if the inputs of the function have names.

        Returns:
            Whether the inputs of the function have names.
        """
        return bool(self.args)

    def has_f_type(self) -> bool:
        """Check if the function has a type.

        Returns:
            Whether the function has a type.
        """
        return bool(self.f_type)

    def __add__(self, other_f: MDOFunction) -> MDOFunction:
        """Operator defining the sum of the function and another one.

        This operator supports automatic differentiation
        if both functions have an implemented Jacobian function.

        Args:
            other_f: The other function.

        Returns:
            The sum of the function and the other one.
        """
        return _AdditionFunctionMaker(MDOFunction, self, other_f).function

    def __sub__(self, other_f: MDOFunction) -> MDOFunction:
        """Operator defining the difference of the function and another one.

        This operator supports automatic differentiation
        if both functions have an implemented Jacobian function.

        Args:
            other_f: The other function.

        Returns:
            The difference of the function and the other one.
        """
        return _AdditionFunctionMaker(MDOFunction, self, other_f, inverse=True).function

    def _min_pt(self, x_vect: ArrayType) -> ArrayType:
        """Evaluate the function and return its opposite value.

        Args:
            x_vect: The value of the inputs of the function.

        Returns:
            The opposite of the value of the outputs of the function.
        """
        return -self(x_vect)

    def _min_jac(self, x_vect: ArrayType) -> ArrayType:
        """Evaluate the Jacobian function and return its opposite value.

        Args:
            x_vect: The value of the inputs of the Jacobian function.

        Returns:
            The opposite of the value of the Jacobian function.
        """
        return -self.jac(x_vect)  # pylint: disable=E1102

    def __neg__(self) -> MDOFunction:
        """Operator defining the opposite of the function.

        This operator supports automatic differentiation
        if the function has an implemented Jacobian function.

        Returns:
            The opposite of the function.
        """
        jac = self._min_jac if self.has_jac() else None

        if self.has_expr():
            expr = "-" + self.expr.translate({ord("-"): "+", ord("+"): "-"})
            name = "-" + self.name.translate({ord("-"): "+", ord("+"): "-"})
        else:
            expr = f"-{self.name}({pretty_str(self.args, sort=False)})"
            name = f"-{self.name}"

        return MDOFunction(
            self._min_pt,
            name,
            jac=jac,
            args=self.args,
            f_type=self.f_type,
            dim=self.dim,
            outvars=self.outvars,
            expr=expr,
        )

    def __truediv__(self, other: MDOFunction | Number) -> MDOFunction:
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

    def __mul__(self, other: MDOFunction | Number) -> MDOFunction:
        """Define the multiplication operation for MDOFunction.

        This operation supports automatic differentiation
        if the different functions have an implemented Jacobian function.

        Args:
            other: The function or number to multiply by.

        Returns:
            A function multiplying a function by another function or a number.
        """
        return _MultiplicationFunctionMaker(MDOFunction, self, other).function

    @staticmethod
    def _compute_operation_expression(
        operand_1: str, operator: str, operand_2: str | float | int
    ) -> str:
        """Return the string expression of an operation between two operands.

        Args:
            operand_1: The first operand.
            operator: The operator applying to both operands.
            operand_2: The second operand.

        Returns:
            The string expression of the sum of the operands.
        """
        return f"{operand_1} {operator} {operand_2}"

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
        name = f"{self.name}({pretty_str(self.args, sort=False)})"
        function.name = self._compute_operation_expression(
            self.name, operator, second_operand
        )
        function.expr = self._compute_operation_expression(
            self.expr or name, operator, second_operand
        )
        return function

    # TODO: Remove this deprecated method.
    def restrict(
        self,
        frozen_indexes: ndarray[int],
        frozen_values: ArrayType,
        input_dim: int,
        name: str | None = None,
        f_type: str | None = None,
        expr: str | None = None,
        args: Sequence[str] | None = None,
    ) -> FunctionRestriction:
        r"""Return a restriction of the function.

        :math:`\newcommand{\frozeninds}{I}\newcommand{\xfrozen}{\hat{x}}\newcommand{
        \frestr}{\hat{f}}`
        For a subset :math:`\approxinds` of the variables indexes of a function
        :math:`f` to remain frozen at values :math:`\xfrozen_{i \in \frozeninds}` the
        restriction of :math:`f` is given by

        .. math::
            \frestr:
            x_{i \not\in \approxinds}
            \longmapsto
            f(\xref_{i \in \approxinds}, x_{i \not\in \approxinds}).

        Args:
            frozen_indexes: The indexes of the inputs that will be frozen
            frozen_values: The values of the inputs that will be frozen.
            input_dim: The dimension of input space of the function before restriction.
            name: The name of the function after restriction.
                If ``None``,
                create a default name
                based on the name of the current function
                and on the argument `args`.
            f_type: The type of the function after restriction.
                If ``None``, the function will have no type.
            expr: The expression of the function after restriction.
                If ``None``, the function will have no expression.
            args: The names of the inputs of the function after restriction.
                If ``None``, the inputs of the function will have no names.

        Returns:
            The restriction of the function.
        """
        from gemseo.core.mdofunctions.function_restriction import FunctionRestriction

        return FunctionRestriction(
            frozen_indexes, frozen_values, input_dim, self, name, f_type, expr, args
        )

    # TODO: Remove this deprecated method.
    def linear_approximation(
        self,
        x_vect: ArrayType,
        name: str | None = None,
        f_type: str | None = None,
        args: Sequence[str] | None = None,
    ) -> MDOLinearFunction:
        r"""Compute a first-order Taylor polynomial of the function.

        :math:`\newcommand{\xref}{\hat{x}}\newcommand{\dim}{d}`
        The first-order Taylor polynomial of a (possibly vector-valued) function
        :math:`f` at a point :math:`\xref` is defined as

        .. math::
            \newcommand{\partialder}{\frac{\partial f}{\partial x_i}(\xref)}
            f(x)
            \approx
            f(\xref) + \sum_{i = 1}^{\dim} \partialder \, (x_i - \xref_i).

        Args:
            x_vect: The input vector at which to build the Taylor polynomial.
            name: The name of the linear approximation function.
                If ``None``, create a name from the name of the function.
            f_type: The type of the linear approximation function.
                If ``None``, the function will have no type.
            args: The names of the inputs of the linear approximation function,
                or a name base.
                If ``None``, use the names of the inputs of the function.

        Returns:
            The first-order Taylor polynomial of the function at the input vector.
        """
        from gemseo.core.mdofunctions.taylor_polynomials import (
            compute_linear_approximation,
        )

        return compute_linear_approximation(
            self, x_vect, name=name, f_type=f_type, args=args
        )

    # TODO: Remove this deprecated method.
    def convex_linear_approx(
        self,
        x_vect: ArrayType,
        approx_indexes: ndarray[bool] | None = None,
        sign_threshold: float = 1e-9,
    ) -> ConvexLinearApprox:
        r"""Compute a convex linearization of the function.

        :math:`\newcommand{\xref}{\hat{x}}\newcommand{\dim}{d}`
        The convex linearization of a function :math:`f` at a point :math:`\xref`
        is defined as

        .. math::
            \newcommand{\partialder}{\frac{\partial f}{\partial x_i}(\xref)}
            f(x)
            \approx
            f(\xref)
            +
            \sum_{\substack{i = 1 \\ \partialder > 0}}^{\dim}
            \partialder \, (x_i - \xref_i)
            -
            \sum_{\substack{i = 1 \\ \partialder < 0}}^{\dim}
            \partialder \, \xref_i^2 \, \left(\frac{1}{x_i} - \frac{1}{\xref_i}\right).

        :math:`\newcommand{\approxinds}{I}`
        Optionally, one may require the convex linearization of :math:`f` with
        respect to a subset of its variables
        :math:`x_{i \in \approxinds}`, :math:`I \subset \{1, \dots, \dim\}`,
        rather than all of them:

        .. math::
            f(x)
            =
            f(x_{i \in \approxinds}, x_{i \not\in \approxinds})
            \approx
            f(\xref_{i \in \approxinds}, x_{i \not\in \approxinds})
            +
            \sum_{\substack{i \in \approxinds \\ \partialder > 0}}
            \partialder \, (x_i - \xref_i)
            -
            \sum_{\substack{i \in \approxinds \\ \partialder < 0}}
            \partialder
            \, \xref_i^2 \, \left(\frac{1}{x_i} - \frac{1}{\xref_i}\right).

        Args:
            x_vect: The input vector at which to build the convex linearization.
            approx_indexes: A boolean mask
                specifying w.r.t. which inputs the function should be approximated.
                If ``None``, consider all the inputs.
            sign_threshold: The threshold for the sign of the derivatives.

        Returns:
            The convex linearization of the function at the given input vector.
        """
        from gemseo.core.mdofunctions.convex_linear_approx import ConvexLinearApprox

        return ConvexLinearApprox(
            x_vect, self, approx_indexes=approx_indexes, sign_threshold=sign_threshold
        )

    # TODO: Remove this deprecated method.
    def quadratic_approx(
        self,
        x_vect: ArrayType,
        hessian_approx: ArrayType,
        args: Sequence[str] | None = None,
    ) -> MDOQuadraticFunction:
        r"""Build a quadratic approximation of the function at a given point.

        The function must be scalar-valued.

        :math:`\newcommand{\xref}{\hat{x}}\newcommand{\dim}{d}\newcommand{
        \hessapprox}{\hat{H}}`
        For a given approximation :math:`\hessapprox` of the Hessian matrix of a
        function :math:`f` at a point :math:`\xref`, the quadratic approximation of
        :math:`f` is defined as

        .. math::
            \newcommand{\partialder}{\frac{\partial f}{\partial x_i}(\xref)}
            f(x)
            \approx
            f(\xref)
            + \sum_{i = 1}^{\dim} \partialder \, (x_i - \xref_i)
            + \frac{1}{2} \sum_{i = 1}^{\dim} \sum_{j = 1}^{\dim}
            \hessapprox_{ij} (x_i - \xref_i) (x_j - \xref_j).

        Args:
            x_vect: The input vector at which to build the quadratic approximation.
            hessian_approx: The approximation of the Hessian matrix
                at this input vector.
            args: The names of the inputs of the quadratic approximation function,
                or a name base.
                If ``None``, use the ones of the current function.

        Returns:
            The second-order Taylor polynomial of the function at the given point.

        Raises:
            ValueError: Either if the approximated Hessian matrix is not square,
                or if it is not consistent with the dimension of the given point.
            AttributeError: If the function does not have
                an implemented Jacobian function.
        """
        from gemseo.core.mdofunctions.taylor_polynomials import (
            compute_quadratic_approximation,
        )

        return compute_quadratic_approximation(self, x_vect, hessian_approx, args=args)

    def check_grad(
        self,
        x_vect: ArrayType,
        method: str = "FirstOrderFD",
        step: float = 1e-6,
        error_max: float = 1e-8,
    ) -> None:
        """Check the gradients of the function.

        Args:
            x_vect: The vector at which the function is checked.
            method: The method used to approximate the gradients,
                either "FirstOrderFD" or "ComplexStep".
            step: The step for the approximation of the gradients.
            error_max: The maximum value of the error.

        Raises:
            ValueError: Either if the approximation method is unknown,
                if the shapes of
                the analytical and approximated Jacobian matrices
                are inconsistent
                or if the analytical gradients are wrong.
        """
        if method == "FirstOrderFD":
            gradient_approximator = FirstOrderFD(self, step)
        elif method == "ComplexStep":
            gradient_approximator = ComplexStep(self, step)
        else:
            raise ValueError(
                f"Unknown approximation method {method},"
                "use 'FirstOrderFD' or 'ComplexStep'."
            )

        approximation = gradient_approximator.f_gradient(x_vect).real
        reference = self._jac(x_vect).real
        if approximation.shape != reference.shape:
            approximation_is_1d = approximation.ndim == 1 or approximation.shape[0] == 1
            reference_is_1d = reference.ndim == 1 or reference.shape[0] == 1
            shapes_are_1d = approximation_is_1d and reference_is_1d
            flatten_diff = reference.flatten().shape != approximation.flatten().shape
            if not shapes_are_1d or (shapes_are_1d and flatten_diff):
                raise ValueError(
                    "Inconsistent function jacobian shape; "
                    f"got: {reference.shape} while expected: {approximation.shape}."
                )

        if self.rel_err(reference, approximation, error_max) > error_max:
            LOGGER.error("Function jacobian is wrong %s", self)
            LOGGER.error("Error =\n%s", self.filt_0(reference - approximation))
            LOGGER.error("Analytic jacobian=\n%s", self.filt_0(reference))
            LOGGER.error("Approximate step gradient=\n%s", self.filt_0(approximation))
            raise ValueError(f"Function jacobian is wrong {self}.")

    @staticmethod
    def rel_err(a_vect: ArrayType, b_vect: ArrayType, error_max: float) -> float:
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
    def filt_0(arr: ArrayType, floor_value: float = 1e-6) -> ArrayType:
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
    def init_from_dict_repr(**attributes) -> MDOFunction:
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
        for attribute in attributes:
            if attribute not in serializable_attributes:
                raise ValueError(
                    f"Cannot initialize MDOFunction attribute: {attribute}, "
                    f"allowed ones are: {pretty_str(serializable_attributes)}."
                )
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
            normalize: If True, the values of the inputs are unnormalized before call.
            jac: If True, a Jacobian pointer is also generated.
            x_tolerance: The tolerance on the distance between inputs.
        """
        SetPtFromDatabase(database, design_space, self, normalize, jac, x_tolerance)

    @classmethod
    def generate_args(
        cls, input_dim: int, args: Sequence[str] | None = None
    ) -> Sequence[str]:
        """Generate the names of the inputs of the function.

        Args:
            input_dim: The dimension of the input space of the function.
            args: The initial names of the inputs of the function.
                If there is only one name,
                e.g. ``["var"]``,
                use this name as a base name
                and generate the names of the inputs,
                e.g. ``["var!0", "var!1", "var!2"]``
                if the dimension of the input space is equal to 3.
                If ``None``,
                use ``"x"`` as a base name and generate the names of the inputs,
                i.e. ``["x!0", "x!1", "x!2"]``.

        Returns:
            The names of the inputs of the function.
        """
        args = args or []
        n_args = len(args)
        if n_args == input_dim:
            return args

        return cls._generate_args(
            args[0] if n_args == 1 else cls.__INPUT_NAME_PATTERN, input_dim
        )

    @classmethod
    def _generate_args(cls, args_base: str, input_dim: int) -> list[str]:
        """Generate the names of the inputs from a base name and their indices.

        Args:
            args_base: The base name.
            input_dim: The number of scalar inputs.

        Returns:
            The names of the inputs.
        """
        return [f"{args_base}{cls.INDEX_PREFIX}{i}" for i in range(input_dim)]

    # TODO: Remove this deprecated method.
    @staticmethod
    def concatenate(
        functions: Iterable[MDOFunction], name: str, f_type: str | None = None
    ) -> Concatenate:
        """Concatenate functions.

        Args:
            functions: The functions to be concatenated.
            name: The name of the concatenation function.
            f_type: The type of the concatenation function.
                If ``None``, the function will have no type.

        Returns:
            The concatenation of the functions.
        """
        from gemseo.core.mdofunctions.concatenate import Concatenate

        return Concatenate(functions, name, f_type)

    @property
    def expects_normalized_inputs(self) -> bool:
        """Whether the functions expect normalized inputs or not."""
        return False

    def get_indexed_name(self, index: int) -> str:
        """Return the name of function component.

        Args:
            index: The index of the function component.

        Returns:
            The name of the function component.
        """
        return f"{self.name}{DesignSpace.SEP}{index}"
