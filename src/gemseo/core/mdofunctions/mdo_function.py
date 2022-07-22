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
from operator import mul
from operator import truediv
from pathlib import Path
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Iterable
from typing import NoReturn
from typing import Sequence
from typing import Sized
from typing import Union

from numpy import abs as np_abs
from numpy import absolute
from numpy import add
from numpy import array
from numpy import atleast_1d
from numpy import atleast_2d
from numpy import concatenate
from numpy import empty
from numpy import matmul
from numpy import multiply
from numpy import ndarray
from numpy import ones_like
from numpy import subtract
from numpy import vstack
from numpy import where
from numpy import zeros
from numpy import zeros_like
from numpy.linalg import multi_dot
from numpy.linalg import norm

from gemseo.algos.database import Database
from gemseo.algos.design_space import DesignSpace
from gemseo.utils.derivatives.complex_step import ComplexStep
from gemseo.utils.derivatives.finite_differences import FirstOrderFD

LOGGER = logging.getLogger(__name__)

OperandType = Union[ndarray, Number]
OperatorType = Callable[[OperandType, OperandType], OperandType]


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
    can be accessed with the properties :attr:`.MDOFunction.func` and :attr:`.MDOFunction.jac`.

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
    using respectively :meth:`.MDOFunction.linear_approximation` and :meth:`quadratic_approx`;
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

    last_eval: ndarray | None
    """The value of the function output at the last evaluation.

    ``None`` if it has not yet been evaluated.
    """

    force_real: bool
    """Whether to cast the results to real value."""

    special_repr: str | None
    """The string representation of the function overloading its default string ones.

    If ``None``, the default string representation is used.
    """

    _n_calls: Value
    """The number of times that the function has been evaluated."""

    _f_type: str
    """The type of the function, among :attr:`.MDOFunction.AVAILABLE_TYPES`."""

    _func: Callable[[ndarray], ndarray]
    """The function to be evaluated from a given input vector."""

    _jac: Callable[[ndarray], ndarray]
    """The Jacobian function to be evaluated from a given input vector."""

    _name: str
    """The name of the function."""

    _args: Sequence[str]
    """The names of the inputs of the function."""

    _expr: str
    """The expression of the function, e.g. `"2*x"`."""

    _dim: int
    """The dimension of the output space of the function."""

    _outvars: Sequence[str]
    """The names of the outputs of the function."""

    _ATTR_NOT_TO_SERIALIZE: tuple[str] = ("_n_calls",)
    """The attributes that shall be skipped at serialization. Private attributes shall
    be written following name mangling conventions: ``_ClassName__attribute_name``.
    Subclasses must expand this class attribute if needed. """

    def __init__(
        self,
        func: Callable[[ndarray], ndarray] | None,
        name: str,
        f_type: str | None = None,
        jac: Callable[[ndarray], ndarray] | None = None,
        expr: str | None = None,
        args: Sequence[str] | None = None,
        dim: int | None = None,
        outvars: Sequence[str] | None = None,
        force_real: bool = False,
        special_repr: str | None = None,
    ) -> None:
        """
        Args:
            func: The original function to be actually called.
                If None, the function will not have an original function.
            name: The name of the function.
            f_type: The type of the function among :attr:`.MDOFunction.AVAILABLE_TYPES`.
                If None, the function will have no type.
            jac: The original Jacobian function to be actually called.
                If None, the function will not have an original Jacobian function.
            expr: The expression of the function, e.g. `"2*x"`.
                If None, the function will have no expression.
            args: The names of the inputs of the function.
                If None, the inputs of the function will have no names.
            dim: The dimension of the output space of the function.
                If None, the dimension of the output space of the function
                will be deduced from the evaluation of the function.
            outvars: The names of the outputs of the function.
                If None, the outputs of the function will have no names.
            force_real: If True, cast the results to real value.
            special_repr: Overload the default string representation of the function.
                If None, use the default string representation.
        """
        super().__init__()

        # Initialize attributes
        self._f_type = None
        self._func = NotImplementedCallable()
        self._jac = NotImplementedCallable()
        self._name = None
        self._args = None
        self._expr = None
        self._dim = None
        self._outvars = None
        self._init_shared_attrs()
        # Use setters to check values
        self.func = func
        self.jac = jac
        self.name = name
        self.f_type = f_type
        self.expr = expr
        self.args = args
        self.dim = dim
        self.outvars = outvars
        self.last_eval = None
        self.force_real = force_real
        self.special_repr = special_repr
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
    def func(self) -> Callable[[ndarray], ndarray]:
        """The function to be evaluated from a given input vector."""
        return self.__counted_f

    def __counted_f(
        self,
        x_vect: ndarray,
    ) -> ndarray:
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
    def func(
        self,
        f_pointer: Callable[[ndarray], ndarray],
    ) -> None:
        if self.activate_counters:
            self._n_calls.value = 0

        self._func = f_pointer

    def serialize(
        self,
        file_path: str | Path,
    ) -> None:
        """Serialize the function and store it in a file.

        Args:
            file_path: The path to the file to store the function.
        """
        with Path(file_path).open("wb") as outfobj:
            pickler = pickle.Pickler(outfobj, protocol=2)
            pickler.dump(self)

    @staticmethod
    def deserialize(
        file_path: str | Path,
    ) -> MDOFunction:
        """Deserialize a function from a file.

        Args:
            file_path: The path to the file containing the function.

        Returns:
            The function instance.
        """
        with Path(file_path).open("rb") as file_:
            pickler = pickle.Unpickler(file_)
            obj = pickler.load()
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

    def __setstate__(
        self,
        state: dict[str, Any],
    ) -> None:
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

    def __call__(
        self,
        x_vect: ndarray,
    ) -> ndarray:
        """Evaluate the function.

        This method can cast the result to real value
        according to the value of the attribute :attr:`.MDOFunction.force_real`.

        Args:
            x_vect: The value of the inputs of the function.

        Returns:
            The value of the outputs of the function.
        """
        return self.evaluate(x_vect)

    def evaluate(
        self,
        x_vect: ndarray,
    ) -> ndarray:
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

        if self.dim is None:
            self.dim = val.size if isinstance(val, ndarray) else 1
        return val

    @property
    def name(self) -> str:
        """The name of the function.

        Raises:
            TypeError: If the name of the function is not a string.
        """
        return self._name

    @name.setter
    def name(
        self,
        name: str,
    ) -> None:
        if name is not None and not isinstance(name, str):
            raise TypeError(
                f"MDOFunction name must be a string; got {type(name)} instead."
            )
        self._name = name

    @property
    def f_type(self) -> str:
        """The type of the function, among :attr:`.MDOFunction.AVAILABLE_TYPES`.

        Raises:
            ValueError: If the type of function is not available.
        """
        return self._f_type

    @f_type.setter
    def f_type(
        self,
        f_type: str,
    ) -> None:
        if f_type is not None and f_type not in self.AVAILABLE_TYPES:
            raise ValueError(
                "MDOFunction type must be among {}; got {} instead.".format(
                    self.AVAILABLE_TYPES, f_type
                )
            )
        self._f_type = f_type

    @property
    def jac(self) -> Callable[[ndarray], ndarray]:
        """The Jacobian function to be evaluated from a given input vector.

        Raises:
            TypeError: If the Jacobian function is not callable.
        """
        return self._jac

    @jac.setter
    def jac(
        self,
        jac: Callable[[ndarray], ndarray],
    ) -> None:
        if jac is not None:
            if not callable(jac):
                raise TypeError("Jacobian function must be callable.")
        self._jac = jac

    @property
    def args(self) -> Sequence[str]:
        """The names of the inputs of the function."""
        return self._args

    @args.setter
    def args(
        self,
        args: Iterable[str] | ndarray | None,
    ) -> None:
        if args is not None:
            if isinstance(args, ndarray):
                self._args = args.tolist()
            else:
                self._args = list(args)
        else:
            self._args = None

    @property
    def expr(self) -> str:
        """The expression of the function, e.g. `"2*x"`.

        Raises:
            TypeError: If the expression is not a string.
        """
        return self._expr

    @expr.setter
    def expr(
        self,
        expr: str,
    ) -> None:
        if expr is not None and not isinstance(expr, str):
            raise TypeError(f"Expression must be a string; got {type(expr)} instead.")
        self._expr = expr

    @property
    def dim(self) -> int:
        """The dimension of the output space of the function.

        Raises:
            TypeError: If the dimension of the output space is not an integer.
        """
        return self._dim

    @dim.setter
    def dim(
        self,
        dim: int,
    ) -> None:
        if dim is not None and not int(dim) == dim:
            raise TypeError(
                f"The dimension must be an integer; got {type(int)} instead."
            )
        self._dim = dim

    @property
    def outvars(self) -> Sequence[str]:
        """The names of the outputs of the function."""
        return self._outvars

    @outvars.setter
    def outvars(
        self,
        outvars: Sequence[str],
    ) -> None:
        if outvars is not None:
            if isinstance(outvars, ndarray):
                self._outvars = outvars.tolist()
            else:
                self._outvars = list(outvars)
        else:
            self._outvars = None

    def is_constraint(self) -> bool:
        """Check if the function is a constraint.

        The type of a constraint function is either 'eq' or 'ineq'.

        Returns:
            Whether the function is a constraint.
        """
        return self.f_type in [self.TYPE_EQ, self.TYPE_INEQ]

    def __repr__(self) -> str:
        return self.special_repr or self.default_repr

    @property
    def default_repr(self) -> str:
        """The default string representation of the function."""
        str_repr = self.name
        if self.has_args():
            arguments = ", ".join(self.args)
            str_repr += f"({arguments})"

        if self.has_expr():
            str_repr += " = "
            expr = self.expr
            n_char = len(str_repr)
            # Remove empty lines with filter
            expr_spl = [_f for _f in expr.split("\n") if _f]
            str_repr += expr_spl[0]
            for repre in expr_spl[1:]:
                str_repr += "\n" + " " * n_char + repre
        return str_repr

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
        return self.dim is not None

    def has_outvars(self) -> bool:
        """Check if the outputs of the function have names.

        Returns:
            Whether the outputs of the function have names.
        """
        return self.outvars is not None

    def has_expr(self) -> bool:
        """Check if the function has an expression.

        Returns:
            Whether the function has an expression.
        """
        return self.expr is not None

    def has_args(self) -> bool:
        """Check if the inputs of the function have names.

        Returns:
            Whether the inputs of the function have names.
        """
        return self.args is not None

    def has_f_type(self) -> bool:
        """Check if the function has a type.

        Returns:
            Whether the function has a type.
        """
        return self.f_type is not None

    def __add__(self, other_f: MDOFunction) -> MDOFunction:
        """Operator defining the sum of the function and another one.

        This operator supports automatic differentiation
        if both functions have an implemented Jacobian function.

        Args:
            other_f: The other function.

        Returns:
            The sum of the function and the other one.
        """
        return ApplyOperator(
            other_f, operator=add, operator_repr="+", mdo_function=self
        )

    def __sub__(self, other_f: MDOFunction) -> MDOFunction:
        """Operator defining the difference of the function and another one.

        This operator supports automatic differentiation
        if both functions have an implemented Jacobian function.

        Args:
            other_f: The other function.

        Returns:
            The difference of the function and the other one.
        """
        return ApplyOperator(
            other_f, operator=subtract, operator_repr="-", mdo_function=self
        )

    def _min_pt(self, x_vect: ndarray) -> ndarray:
        """Evaluate the function and return its opposite value.

        Args:
            x_vect: The value of the inputs of the function.

        Returns:
            The opposite of the value of the outputs of the function.
        """
        return -self(x_vect)

    def _min_jac(self, x_vect: ndarray) -> ndarray:
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

        min_name = f"-{self.name}"
        min_self = MDOFunction(
            self._min_pt,
            min_name,
            args=self.args,
            f_type=self.f_type,
            dim=self.dim,
            outvars=self.outvars,
        )

        if self.has_jac():
            min_self.jac = self._min_jac

        if self.has_expr():
            min_self.expr = self.expr
            min_self.expr = min_self.expr.replace("+", "++")
            min_self.expr = min_self.expr.replace("-", "+")
            min_self.expr = min_self.expr.replace("++", "-")
            min_self.expr = "-" + min_self.expr
            min_self.name = self.name
            min_self.name = min_self.name.replace("+", "++")
            min_self.name = min_self.name.replace("-", "+")
            min_self.name = min_self.name.replace("++", "-")
            min_self.name = "-" + min_self.name

        return min_self

    def __truediv__(self, other):
        """Define the division operation for MDOFunction.

        This operation supports automatic differentiation
        if the different functions have an implemented Jacobian function.

        Args:
            other: The function to divide by.
        """
        return MultiplyOperator(other, self, inverse=True)

    def __mul__(self, other):
        """Define the multiplication operation for MDOFunction.

        This operation supports automatic differentiation
        if the different functions have an implemented Jacobian function.

        Args:
            other: The function to multiply by.
        """
        return MultiplyOperator(other, self)

    @staticmethod
    def _compute_operation_expression(
        operand_1: str,
        operator: str,
        operand_2: str | float | int,
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

    def offset(self, value: ndarray | Number) -> MDOFunction:
        """Add an offset value to the function.

        Args:
            value: The offset value.

        Returns:
            The offset function as an MDOFunction object.
        """
        return Offset(value, self)

    def restrict(
        self,
        frozen_indexes: ndarray,
        frozen_values: ndarray,
        input_dim: int,
        name: str | None = None,
        f_type: str | None = None,
        expr: str | None = None,
        args: Sequence[str] | None = None,
    ) -> MDOFunction:
        r"""Return a restriction of the function

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
                If None,
                create a default name
                based on the name of the current function
                and on the argument `args`.
            f_type: The type of the function after restriction.
                If None, the function will have no type.
            expr: The expression of the function after restriction.
                If None, the function will have no expression.
            args: The names of the inputs of the function after restriction.
                If None, the inputs of the function will have no names.

        Returns:
            The restriction of the function.
        """
        return FunctionRestriction(
            frozen_indexes, frozen_values, input_dim, self, name, f_type, expr, args
        )

    def linear_approximation(
        self,
        x_vect: ndarray,
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
                If None, create a name from the name of the function.
            f_type: The type of the linear approximation function.
                If None, the function will have no type.
            args: The names of the inputs of the linear approximation function,
                or a name base.
                If None, use the names of the inputs of the function.

        Returns:
            The first-order Taylor polynomial of the function at the input vector.

        Raises:
            AttributeError: If the function does not have a Jacobian function.
        """
        # Check that the function Jacobian is available
        if not self.has_jac():
            raise AttributeError(
                "Function Jacobian unavailable for linear approximation."
            )

        # Build the first-order Taylor polynomial
        coefficients = self.jac(x_vect)
        func_val = self.evaluate(x_vect)
        if isinstance(func_val, ndarray):
            # Make sure the function value is at most 1-dimensional
            func_val = func_val.flatten()
        value_at_zero = func_val - matmul(coefficients, x_vect)
        linear_approx_suffix = "_linearized"
        name = self.name + linear_approx_suffix if name is None else name
        args = args if args else self.args
        linear_approx = MDOLinearFunction(
            coefficients, name, f_type, args, value_at_zero
        )

        return linear_approx

    def convex_linear_approx(
        self,
        x_vect: ndarray,
        approx_indexes: ndarray | None = None,
        sign_threshold: float = 1e-9,
    ) -> MDOFunction:
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
                If None, consider all the inputs.
            sign_threshold: The threshold for the sign of the derivatives.

        Returns:
            The convex linearization of the function at the given input vector.
        """
        return ConvexLinearApprox(
            x_vect, self, approx_indexes=approx_indexes, sign_threshold=sign_threshold
        )

    def quadratic_approx(
        self,
        x_vect: ndarray,
        hessian_approx: ndarray,
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
                If None, use the ones of the current function.

        Returns:
            The second-order Taylor polynomial of the function at the given point.

        Raises:
            ValueError: Either if the approximated Hessian matrix is not square,
                or if it is not consistent with the dimension of the given point.
            AttributeError: If the function does not have
                an implemented Jacobian function.
        """
        # Build the second-order coefficients
        if (
            not isinstance(hessian_approx, ndarray)
            or hessian_approx.ndim != 2
            or hessian_approx.shape[0] != hessian_approx.shape[1]
        ):
            raise ValueError("Hessian approximation must be a square ndarray.")
        if hessian_approx.shape[1] != x_vect.size:
            raise ValueError(
                "Hessian approximation and vector must have same dimension."
            )
        quad_coeffs = 0.5 * hessian_approx

        # Build the first-order coefficients
        if not self.has_jac():
            raise AttributeError("Jacobian unavailable.")
        gradient = self.jac(x_vect)
        hess_dot_vect = matmul(hessian_approx, x_vect)
        linear_coeffs = gradient - hess_dot_vect

        # Buid the zero-order coefficient
        zero_coeff = matmul(0.5 * hess_dot_vect - gradient, x_vect) + self.evaluate(
            x_vect
        )

        # Build the second-order Taylor polynomial
        quad_approx = MDOQuadraticFunction(
            quad_coeffs=quad_coeffs,
            linear_coeffs=linear_coeffs,
            value_at_zero=zero_coeff,
            name=f"{self.name}_quadratized",
            args=args if args else self.args,
        )

        return quad_approx

    def check_grad(
        self,
        x_vect: ndarray,
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
            apprx = FirstOrderFD(self, step)
        elif method == "ComplexStep":
            apprx = ComplexStep(self, step)
        else:
            raise ValueError(
                "Unknown approximation method {},"
                "use 'FirstOrderFD' or 'ComplexStep'.".format(method)
            )
        apprx_grad = apprx.f_gradient(x_vect).real
        anal_grad = self._jac(x_vect).real
        if not apprx_grad.shape == anal_grad.shape:
            shape_app1d = len(apprx_grad.shape) == 1 or apprx_grad.shape[0] == 1
            shape_ex1d = len(anal_grad.shape) == 1 or anal_grad.shape[0] == 1
            shape_1d = shape_app1d and shape_ex1d
            flatten_diff = anal_grad.flatten().shape != apprx_grad.flatten().shape
            if not shape_1d or (shape_1d and flatten_diff):
                raise ValueError(
                    "Inconsistent function jacobian shape; "
                    "got: {} while expected: {}.".format(
                        anal_grad.shape, apprx_grad.shape
                    )
                )
        rel_error = self.rel_err(anal_grad, apprx_grad, error_max)
        succeed = rel_error < error_max
        if not succeed:
            LOGGER.error("Function jacobian is wrong %s", str(self))
            LOGGER.error("Error =\n%s", str(self.filt_0(anal_grad - apprx_grad)))
            LOGGER.error("Analytic jacobian=\n%s", str(self.filt_0(anal_grad)))
            LOGGER.error("Approximate step gradient=\n%s", str(self.filt_0(apprx_grad)))
            raise ValueError(f"Function jacobian is wrong {self}.")

    @staticmethod
    def rel_err(
        a_vect: ndarray,
        b_vect: ndarray,
        error_max: float,
    ) -> float:
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
        n_b = norm(b_vect)
        if n_b > error_max:
            return norm(a_vect - b_vect) / norm(b_vect)
        return norm(a_vect - b_vect)

    @staticmethod
    def filt_0(
        arr: ndarray,
        floor_value: float = 1e-6,
    ) -> ndarray:
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
    def init_from_dict_repr(**kwargs) -> MDOFunction:
        """Initialize a new function.

        This is typically used for deserialization.

        Args:
            **kwargs: The attributes from :attr:`.MDOFunction.DICT_REPR_ATTR`.

        Returns:
            A function initialized from the provided data.

        Raises:
            ValueError: If the name of an argument is not in
                :attr:`.MDOFunction.DICT_REPR_ATTR`.
        """
        allowed = MDOFunction.DICT_REPR_ATTR
        for key in kwargs:
            if key not in allowed:
                raise ValueError(
                    "Cannot initialize MDOFunction attribute: {}, "
                    "allowed ones are: {}.".format(key, ", ".join(allowed))
                )
        return MDOFunction(func=None, **kwargs)

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
        or `None`.
        The same for the method :meth:`.MDOFunction.jac`.

        Args:
            database: The database to read.
            design_space: The design space used for normalization.
            normalize: If True, the values of the inputs are unnormalized before call.
            jac: If True, a Jacobian pointer is also generated.
            x_tolerance: The tolerance on the distance between inputs.
        """
        SetPtFromDatabase(database, design_space, self, normalize, jac, x_tolerance)

    @staticmethod
    def generate_args(
        input_dim: int,
        args: Sequence[str] | None = None,
    ) -> Sequence[str]:
        """Generate the names of the inputs of the function.

        Args:
            input_dim: The dimension of the input space of the function.
            args: The initial names of the inputs of the function.
                If there is only one name,
                e.g. `["var"]`,
                use this name as a base name
                and generate the names of the inputs,
                e.g. `["var!0", "var!1", "var!2"]`
                if the dimension of the input space is equal to 3.
                If None,
                use `"x"` as a base name and generate the names of the inputs,
                i.e. `["x!0", "x!1", "x!2"]`.

        Returns:
            The names of the inputs of the function.
        """
        if args and len(args) == input_dim:
            # Keep the passed list of strings
            new_args = args
        elif args and len(args) == 1:
            # Generate the arguments strings based on the unique passed string
            new_args = MDOFunction._generate_args(args[0], input_dim)
        else:
            # Generate the arguments strings based on the default string
            args_base = "x"
            new_args = MDOFunction._generate_args(args_base, input_dim)

        return new_args

    @staticmethod
    def _generate_args(
        args_base: str,
        input_dim: int,
    ) -> list[str]:
        """Generate the names of the inputs from a base name and their indices.

        Args:
            args_base: The base name.
            input_dim: The number of scalar inputs.

        Returns:
            The names of the inputs.
        """
        index_prefix = MDOLinearFunction.INDEX_PREFIX
        return [args_base + index_prefix + str(i) for i in range(input_dim)]

    @staticmethod
    def concatenate(
        functions: Iterable[MDOFunction],
        name: str,
        f_type: str | None = None,
    ) -> MDOFunction:
        """Concatenate functions.

        Args:
            functions: The functions to be concatenated.
            name: The name of the concatenation function.
            f_type: The type of the concatenation function.
                If None, the function will have no type.

        Returns:
            The concatenation of the functions.
        """
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


class NotImplementedCallable:
    """A not implemented callable object."""

    def __call__(self, *args, **kwargs) -> NoReturn:
        """
        Raises:
            NotImplementedError: At each evaluation.
        """
        raise NotImplementedError("Function is not implemented.")


class ApplyOperator(MDOFunction):
    """Define addition/subtraction for an MDOFunction.

    Supports automatic differentiation if other_f and self have a Jacobian.
    """

    def __init__(
        self,
        other: MDOFunction | Number,
        operator: MDOFunction,
        operator_repr: str,
        mdo_function: MDOFunction,
    ) -> None:
        """Apply an operator to the function and another function.

        This operator supports automatic differentiation
        if both functions have an implemented Jacobian function.

        Args:
            other: The other function or number.
            operator: The operator as a function pointer.
            operator_repr: The representation of the operator.
            mdo_function: The original function.

        Raises:
            TypeError: If `other` is not an MDOFunction or a Number.
        """
        self.__other = other
        self.__operator = operator
        self.__operator_repr = operator_repr
        self.__mdo_function = mdo_function

        self.__is_number = isinstance(self.__other, Number)
        self.__is_func = isinstance(self.__other, MDOFunction)
        if not self.__is_number and not self.__is_func:
            raise TypeError(
                "Unsupported + operand for MDOFunction and {}.".format(
                    type(self.__other)
                )
            )

        if self.__is_func:
            add_name = (
                self.__mdo_function.name + self.__operator_repr + self.__other.name
            )

            if self.__mdo_function.has_jac() and self.__other.has_jac():
                jac = self._add_jac
            else:
                jac = None

            if self.__mdo_function.has_expr() and self.__other.has_expr():
                expr = (
                    self.__mdo_function.expr + self.__operator_repr + self.__other.expr
                )
            else:
                expr = None

            if self.__mdo_function.has_args() and self.__other.has_args():
                args = sorted(list(set(self.__mdo_function.args + self.__other.args)))
            else:
                args = None

            if self.__mdo_function.has_f_type():
                f_type = self.__mdo_function.f_type
            elif self.__other.has_f_type():
                f_type = self.__other.f_type
            else:
                f_type = None

            super().__init__(
                self._add_f_pt,
                add_name,
                jac=jac,
                expr=expr,
                args=args,
                f_type=f_type,
                dim=self.__mdo_function.dim,
                outvars=self.__mdo_function.outvars,
            )

        elif self.__is_number:
            add_name = (
                self.__mdo_function.name + self.__operator_repr + str(self.__other)
            )

            if self.__mdo_function.has_expr():
                expr = (
                    self.__mdo_function.expr + self.__operator_repr + str(self.__other)
                )
            else:
                expr = None

            super().__init__(
                self._add_f_pt,
                add_name,
                jac=self.__mdo_function.jac,
                expr=expr,
                args=self.__mdo_function.args,
                f_type=self.__mdo_function.f_type,
                dim=self.__mdo_function.dim,
                outvars=self.__mdo_function.outvars,
            )

    def _add_f_pt(self, x_vect: ndarray) -> MDOFunction:
        """Evaluate the function and apply the operator to the value of its outputs.

        Args:
            x_vect: The values of the inputs of the function.

        Returns:
            The value of the outputs of the function modified by the operator.
        """
        if self.__is_number:
            selfval = self.__mdo_function(x_vect)
            return self.__operator(selfval, self.__other)
        if self.__is_func:
            selfval = self.__mdo_function(x_vect)
            otherval = self.__other(x_vect)
            return self.__operator(selfval, otherval)

    def _add_jac(self, x_vect: ndarray) -> MDOFunction:
        """Define the Jacobian of the addition function.

        Args:
            x_vect: The design variables values.

        Returns:
            The Jacobian of self(x_vect) added to the Jacbian of other(x_vect).
        """
        self_jac = self.__mdo_function._jac(x_vect)
        other_jac = self.__other.jac(x_vect)
        return self.__operator(self_jac, other_jac)


class Concatenate(MDOFunction):
    """Wrap the concatenation of a set of functions."""

    def __init__(
        self,
        functions: Iterable[MDOFunction],
        name: str,
        f_type: str | None = None,
    ):
        """
        Args:
            functions: The functions to be concatenated.
            name: The name of the concatenation function.
            f_type: The type of the concatenation function.
                If None, the function will have no type.
        """
        self.__functions = functions
        self.__name = name
        self.__f_type = f_type

        func_output_names = [func.outvars for func in self.__functions]
        if None in func_output_names:
            output_names = None
        else:
            output_names = [
                output_name
                for output_names in func_output_names
                for output_name in output_names
            ]

        super().__init__(
            self._concat_func,
            self.__name,
            self.__f_type,
            self._concat_jac,
            dim=sum(func.dim for func in self.__functions),
            outvars=output_names,
        )

    def _concat_func(
        self,
        x_vect: ndarray,
    ) -> ndarray:
        """Concatenate the values of the outputs of the functions.

        Args:
            x_vect: The value of the inputs of the functions.

        Returns:
            The concatenation of the values of the outputs of the functions.
        """
        return concatenate([atleast_1d(func(x_vect)) for func in self.__functions])

    def _concat_jac(
        self,
        x_vect: ndarray,
    ) -> ndarray:
        """Concatenate the outputs of the Jacobian functions.

        Args:
            x_vect: The value of the inputs of the Jacobian functions.

        Returns:
            The concatenation of the outputs of the Jacobian functions.
        """
        return vstack([atleast_2d(func.jac(x_vect)) for func in self.__functions])


class MultiplyOperator(MDOFunction):
    """Wrap the multiplication of an MDOFunction.

    Supports automatic differentiation if other_f and self have a Jacobian.
    """

    def __init__(
        self,
        other: MDOFunction | Number,
        mdo_function: MDOFunction,
        inverse: bool = False,
    ) -> None:
        """Operator defining the multiplication of the function and another operand.

        This operator supports automatic differentiation
        if the different functions have an implemented Jacobian function.

        Args:
            other: The other operand.
            mdo_function: The original function.
            inverse: Whether to multiply `mdo_function` by the inverse of `other`.

        Raises:
            TypeError: If the other operand is
                neither a number nor a :class:`.MDOFunction`.
        """
        is_func = isinstance(other, MDOFunction)
        operator = "/" if inverse else "*"
        if not isinstance(other, Number) and not is_func:
            raise TypeError(
                f"Unsupported {operator} operator for MDOFunction and {type(other)}."
            )

        self.__other = other
        self.__mdo_function = mdo_function
        self.__operator = truediv if inverse else mul
        self.__is_func = is_func

        args = self.__mdo_function.args
        f_type = self.__mdo_function.f_type

        if self.__is_func:
            first_operand_name = self.__mdo_function.name
            first_operand_expr = self.__mdo_function.expr
            second_operand_name = self.__other.name
            second_operand_expr = self.__other.expr

            if args and self.__other.has_args():
                args = sorted(list(set(args + self.__other.args)))

            f_type = f_type or self.__other.f_type

        else:
            first_operand_name = self.__other
            second_operand_name = self.__mdo_function.name
            first_operand_expr = self.__other
            second_operand_expr = self.__mdo_function.expr

        out_name = f"{first_operand_name}{operator}{second_operand_name}"

        if self.__mdo_function.has_expr() and second_operand_expr is not None:
            expr = f"{first_operand_expr}{operator}{second_operand_expr}"
        else:
            expr = None

        super().__init__(
            self._func,
            out_name,
            expr=expr,
            jac=self._jac,
            args=args,
            f_type=f_type,
            dim=self.__mdo_function.dim,
            outvars=self.__mdo_function.outvars,
        )

    def _func(self, x_vect: ndarray) -> ndarray:
        """Evaluate the function and multiply its output value.

        Args:
            x_vect: The value of the inputs of the function.

        Returns:
            The product of the output value of the function with the number.
        """
        if self.__is_func:
            second_operand = self.__other(x_vect)
        else:
            second_operand = self.__other

        return self.__operator(self.__mdo_function(x_vect), second_operand)

    def _jac(self, x_vect: ndarray) -> ndarray:
        """Evaluate both functions and multiply their output values.

        Args:
            x_vect: The value of the inputs of the function.

        Returns:
            The product of the output values of the functions.
        """

        if not self.__mdo_function.has_jac():
            return

        if not self.__is_func:
            return self.__operator(self.__mdo_function._jac(x_vect), self.__other)

        if not self.__other.has_jac():
            return

        self_f = self.__mdo_function(x_vect)
        other_f = self.__other(x_vect)
        self_jac = self.__mdo_function._jac(x_vect)
        other_jac = self.__other.jac(x_vect)

        if self.__operator == mul:
            return self_jac * other_f + other_jac * self_f

        return (self_jac * other_f - other_jac * self_f) / other_jac**2


class Offset(MDOFunction):
    """Wrap an MDOFunction plus an offset value."""

    def __init__(
        self,
        value: ndarray | Number,
        mdo_function: MDOFunction,
    ) -> None:
        """
        Args:
            value: The offset value.
            mdo_function: The original MDOFunction object.
        """
        name = mdo_function.name
        self.__value = value
        self.__mdo_function = mdo_function

        expr = self.__mdo_function.expr
        if expr is None:
            expr = name

        if isinstance(value, Sized):
            operator = "+"
            operand_2 = "offset"
        else:
            if value < 0:
                operator = "-"
                operand_2 = -value
            else:
                operator = "+"
                operand_2 = value

        name = self.__mdo_function._compute_operation_expression(
            name, operator, operand_2
        )
        expr = self.__mdo_function._compute_operation_expression(
            expr, operator, operand_2
        )

        super().__init__(
            self._wrapped_function,
            name=name,
            f_type=self.__mdo_function.f_type,
            expr=expr,
            args=self.__mdo_function.args,
            dim=self.__mdo_function.dim,
            jac=self.__mdo_function.jac,
            outvars=self.__mdo_function.outvars,
        )

    def _wrapped_function(self, x_vect: ndarray) -> ndarray:
        """Wrap the function to be given to the optimizer.

        Args:
            x_vect: The design variables values.

        Returns:
            The evaluation of function at design variables plus the offset.
        """
        return self.__mdo_function(x_vect) + self.__value


class FunctionRestriction(MDOFunction):
    """Take an :class:`.MDOFunction` and apply a given restriction to its inputs."""

    def __init__(
        self,
        frozen_indexes: ndarray,
        frozen_values: ndarray,
        input_dim: int,
        mdo_function: MDOFunction,
        name: str | None = None,
        f_type: str | None = None,
        expr: str | None = None,
        args: Sequence[str] | None = None,
    ) -> None:
        """
        Args:
            frozen_indexes: The indexes of the inputs that will be frozen
            frozen_values: The values of the inputs that will be frozen.
            input_dim: The dimension of input space of the function before restriction.
            name: The name of the function after restriction.
                If None,
                create a default name
                based on the name of the current function
                and on the argument `args`.
            mdo_function: The function to restrict.
            f_type: The type of the function after restriction.
                If None, the function will have no type.
            expr: The expression of the function after restriction.
                If None, the function will have no expression.
            args: The names of the inputs of the function after restriction.
                If None, the inputs of the function will have no names.

        Raises:
            ValueError: If the `frozen_indexes` and the `frozen_values` arrays do
                not have the same shape.
        """
        # Check the shapes of the passed arrays
        if frozen_indexes.shape != frozen_values.shape:
            raise ValueError("Arrays of frozen indexes and values must have same shape")

        self.__frozen_indexes = frozen_indexes
        self.__frozen_values = frozen_values
        self.__input_dim = input_dim
        self.__mdo_function = mdo_function
        self.__name = name
        self.__f_type = f_type
        self.__expr = expr
        self.__args = args

        self._active_indexes = array(
            [i for i in range(self.__input_dim) if i not in self.__frozen_indexes]
        )

        # Build the name of the restriction
        if self.__name is None and self.__args is not None:
            self.__name = "{}_wrt_{}".format(
                self.__mdo_function.name, "_".join(self.__args)
            )
        elif name is None:
            self.__name = f"{self.__mdo_function.name}_restriction"

        if self.__mdo_function.has_jac():
            jac = self._jac
        else:
            jac = self.__mdo_function.jac

        super().__init__(
            self._func,
            self.__name,
            self.__f_type,
            expr=self.__expr,
            args=self.__args,
            jac=jac,
            dim=self.__mdo_function.dim,
            outvars=self.__mdo_function.outvars,
            force_real=self.__mdo_function.force_real,
        )

    def __extend_subvect(self, x_subvect: ndarray) -> ndarray:
        """Extend an input vector of the restriction with the frozen values.

        Args:
            x_subvect: The values of the inputs of the restriction.

        Returns:
            The extended input vector.
        """
        x_vect = empty(self.__input_dim)
        x_vect[self._active_indexes] = x_subvect
        x_vect[self.__frozen_indexes] = self.__frozen_values
        return x_vect

    def _func(self, x_subvect: ndarray) -> ndarray:
        """Evaluate the restriction.

        Args:
            x_subvect: The value of the inputs of the restriction.

        Returns:
            The value of the outputs of the restriction.
        """
        x_vect = self.__extend_subvect(x_subvect)
        value = self.__mdo_function.evaluate(x_vect)
        return value

    def _jac(self, x_subvect: ndarray) -> ndarray:
        """Compute the Jacobian matrix of the restriction.

        Args:
            x_subvect: The value of the inputs of the restriction.

        Returns:
            The Jacobian matrix of the restriction.
        """
        x_vect = self.__extend_subvect(x_subvect)
        full_jac = self.__mdo_function.jac(x_vect)
        if len(full_jac.shape) == 1:
            sub_jac = full_jac[self._active_indexes]
        else:
            sub_jac = full_jac[:, self._active_indexes]
        return sub_jac


class MDOLinearFunction(MDOFunction):
    r"""Linear multivariate function defined by

    * a matrix :math:`A` of first-order coefficients
      :math:`(a_{ij})_{\substack{i = 1, \dots m \\ j = 1, \dots n}}`
    * and a vector :math:`b` of zero-order coefficients :math:`(b_i)_{i = 1, \dots m}`

    .. math::
        F(x)
        =
        Ax + b
        =
        \begin{bmatrix}
            a_{11} & \cdots & a_{1n} \\
            \vdots & \ddots & \vdots \\
            a_{m1} & \cdots & a_{mn}
        \end{bmatrix}
        \begin{bmatrix} x_1 \\ \vdots \\ x_n \end{bmatrix}
        +
        \begin{bmatrix} b_1 \\ \vdots \\ b_m \end{bmatrix}.

    """

    def __init__(
        self,
        coefficients: ndarray,
        name: str,
        f_type: str | None = None,
        args: Sequence[str] | None = None,
        value_at_zero: ndarray | Number = 0.0,
    ):
        """
        Args:
            coefficients: The coefficients :math:`A` of the linear function.
            name: The name of the linear function.
            f_type: The type of the linear function among :attr:`.MDOFunction.AVAILABLE_TYPES`.
                If None, the linear function will have no type.
            args: The names of the inputs of the linear function.
                If None, the inputs of the linear function will have no names.
            value_at_zero: The value :math:`b` of the linear function output at zero.
        """
        # Format the passed coefficients and value at zero
        self.coefficients = coefficients
        output_dim, input_dim = self._coefficients.shape
        self.value_at_zero = value_at_zero

        # Generate the arguments strings
        new_args = MDOLinearFunction.generate_args(input_dim, args)

        # Generate the expression string
        if output_dim == 1:
            expr = self._generate_1d_expr(new_args)
        else:
            expr = self._generate_nd_expr(new_args)

        super().__init__(
            self.__fun,
            name,
            f_type=f_type,
            jac=self.__jac,
            expr=expr,
            args=new_args,
            dim=output_dim,
            outvars=[name],
        )

    def __fun(self, x_vect: ndarray) -> ndarray:
        """Return the linear combination with an offset:

        sum_{i=1}^n a_i * x_i + b

        Args:
            x_vect: The design variables values.
        """
        value = matmul(self._coefficients, x_vect) + self._value_at_zero
        if value.size == 1:
            value = value[0]
        return value

    def __jac(self, _):
        """.. note::

        If the function is scalar, the gradient of the function is returned as a
        1d-array. If the function is vectorial, the Jacobian of the function is returned
        as a 2d-array.
        """
        if self._coefficients.shape[0] == 1:
            jac = self._coefficients[0, :]
        else:
            jac = self._coefficients
        return jac

    @property
    def coefficients(self) -> ndarray:
        """The coefficients of the linear function.

        This is the matrix :math:`A` in the expression :math:`y=Ax+b`.

        Raises:
            ValueError: If the coefficients are not passed
                as a 1-dimensional or 2-dimensional ndarray.
        """
        return self._coefficients

    @coefficients.setter
    def coefficients(
        self,
        coefficients: Number | ndarray,
    ) -> None:
        if isinstance(coefficients, Number):
            self._coefficients = atleast_2d(coefficients)
        elif isinstance(coefficients, ndarray) and len(coefficients.shape) == 2:
            self._coefficients = coefficients
        elif isinstance(coefficients, ndarray) and len(coefficients.shape) == 1:
            self._coefficients = coefficients.reshape([1, -1])
        else:
            raise ValueError(
                "Coefficients must be passed as a 2-dimensional "
                "or a 1-dimensional ndarray."
            )

    @property
    def value_at_zero(self) -> ndarray:
        """The value of the function at zero.

        This is the vector :math:`b` in the expression :math:`y=Ax+b`.

        Raises:
            ValueError: If the value at zero is neither an ndarray nor a number.
        """
        return self._value_at_zero

    @value_at_zero.setter
    def value_at_zero(
        self,
        value_at_zero: Number | ndarray,
    ) -> None:
        output_dim = self.coefficients.shape[0]  # N.B. the coefficients must be set
        if isinstance(value_at_zero, ndarray) and value_at_zero.size == output_dim:
            self._value_at_zero = value_at_zero.reshape(output_dim)
        elif isinstance(value_at_zero, Number):
            self._value_at_zero = array([value_at_zero] * output_dim)
        else:
            raise ValueError("Value at zero must be an ndarray or a number.")

    def _generate_1d_expr(
        self,
        args: Sequence[str],
    ) -> str:
        """Generate the literal expression of the linear function in scalar form.

        Args:
            args: The names of the inputs of the function.

        Returns:
            The literal expression of the linear function in scalar form.
        """
        expr = ""

        # Build the expression of the linear combination
        first_non_zero = -1
        for i, coeff in enumerate(self._coefficients[0, :]):
            if coeff != 0.0:
                if first_non_zero == -1:
                    first_non_zero = i
                # Add the monomial sign
                if i == first_non_zero and coeff < 0.0:
                    # The first nonzero coefficient is negative.
                    expr += "-"  # unary minus
                elif i != first_non_zero and coeff < 0.0:
                    expr += " - "
                elif i != first_non_zero and coeff > 0.0:
                    expr += " + "
                # Add the coefficient value
                if abs(coeff) != 1.0:
                    expr += MDOFunction.COEFF_FORMAT_1D.format(abs(coeff)) + "*"
                # Add argument string
                expr += args[i]

        # Add the offset expression
        value_at_zero = self._value_at_zero[0]
        if first_non_zero == -1:
            # Constant function
            expr += MDOFunction.COEFF_FORMAT_1D.format(value_at_zero)
        elif self._value_at_zero > 0.0:
            expr += " + " + MDOFunction.COEFF_FORMAT_1D.format(value_at_zero)
        elif self._value_at_zero < 0.0:
            expr += " - " + MDOFunction.COEFF_FORMAT_1D.format(-value_at_zero)

        return expr

    def _generate_nd_expr(
        self,
        args: Sequence[str],
    ) -> str:
        """Generate the literal expression of the linear function in matrix form.

        Args:
            args: The names of the inputs of the function.

        Returns:
            The literal expression of the linear function in matrix form.
        """
        max_args_len = max(len(arg) for arg in args)
        out_dim, in_dim = self._coefficients.shape
        expr = ""
        for i in range(max(out_dim, in_dim)):
            if i > 0:
                expr += "\n"
            # matrix line
            if i < out_dim:
                line_coeffs_str = (
                    MDOFunction.COEFF_FORMAT_ND.format(coeff)
                    for coeff in self._coefficients[i, :]
                )
                expr += "[{}]".format(" ".join(line_coeffs_str))
            else:
                expr += " " + " ".join([" " * 3] * in_dim) + " "
            # vector line
            if i < in_dim:
                expr += f"[{args[i]}]"
            else:
                expr += " " * (max_args_len + 2)
            # sign
            if i == 0:
                expr += " + "
            else:
                expr += "   "
            # value at zero
            if i < out_dim:
                expr += "[{}]".format(
                    MDOFunction.COEFF_FORMAT_ND.format(self._value_at_zero[i])
                )
        return expr

    def __neg__(self) -> MDOLinearFunction:
        return MDOLinearFunction(
            -self._coefficients,
            "-" + self.name,
            self.f_type,
            self.args,
            -self._value_at_zero,
        )

    def offset(
        self,
        value: Number | ndarray,
    ) -> MDOLinearFunction:
        return MDOLinearFunction(
            self._coefficients,
            self.name,
            self.f_type,
            self.args,
            self._value_at_zero + value,
        )

    def restrict(
        self,
        frozen_indexes: ndarray,
        frozen_values: ndarray,
    ) -> MDOLinearFunction:
        """Build a restriction of the linear function.

        Args:
            frozen_indexes: The indexes of the inputs that will be frozen.
            frozen_values: The values of the inputs that will be frozen.

        Returns:
            The restriction of the linear function.

        Raises:
            ValueError: If the frozen indexes and values have different shapes.
        """
        # Check the shapes of the passed arrays
        if frozen_indexes.shape != frozen_values.shape:
            raise ValueError(
                "Arrays of frozen indexes and values must have same shape."
            )

        # Separate the frozen coefficients from the active ones
        frozen_coefficients = self.coefficients[:, frozen_indexes]
        active_indexes = array(
            [i for i in range(self.coefficients.shape[1]) if i not in frozen_indexes]
        )
        active_coefficients = self.coefficients[:, active_indexes]

        # Compute the restriction value at zero
        value_at_zero = matmul(frozen_coefficients, frozen_values) + self._value_at_zero

        # Build the function restriction
        restriction_suffix = "_restriction"
        name = self.name + restriction_suffix
        args = [self.args[i] for i in active_indexes]
        restriction = MDOLinearFunction(
            active_coefficients, name, args=args, value_at_zero=value_at_zero
        )

        return restriction


class ConvexLinearApprox(MDOFunction):
    """Wrap a convex linearization of the function."""

    def __init__(
        self,
        x_vect: ndarray,
        mdo_function: MDOFunction,
        approx_indexes: ndarray | None = None,
        sign_threshold: float = 1e-9,
    ) -> MDOFunction:
        """
        Args:
            x_vect: The input vector at which to build the convex linearization.
            mdo_function: The function to approximate.
            approx_indexes: A boolean mask
                specifying w.r.t. which inputs the function should be approximated.
                If None, consider all the inputs.
            sign_threshold: The threshold for the sign of the derivatives.

        Raises:
            ValueError: If the length of boolean array
                and the number of inputs of the functions are inconsistent.
            AttributeError: If the function does not have a Jacobian function.
        """
        self.__x_vect = x_vect
        self.__mdo_function = mdo_function
        self.__approx_indexes = approx_indexes
        self.__sign_threshold = sign_threshold

        # Check the approximation indexes
        if self.__approx_indexes is None:
            self.__approx_indexes = ones_like(x_vect, dtype=bool)
        elif (
            self.__approx_indexes.shape != self.__x_vect.shape
            or self.__approx_indexes.dtype != "bool"
        ):
            raise ValueError(
                "The approximation array must be an array of booleans with "
                "the same shape as the function argument."
            )

        # Get the function Jacobian matrix
        if not self.__mdo_function.has_jac():
            raise AttributeError(
                "Function Jacobian unavailable for convex linearization."
            )

        jac = atleast_2d(self.__mdo_function.jac(x_vect))

        # Build the coefficients matrices
        coeffs = jac[:, self.__approx_indexes]
        self.__direct_coeffs = where(coeffs > self.__sign_threshold, coeffs, 0.0)
        self.__recipr_coeffs = multiply(
            -where(-coeffs > self.__sign_threshold, coeffs, 0.0),
            self.__x_vect[self.__approx_indexes] ** 2,
        )

        super().__init__(
            self._convex_lin_func,
            f"{self.__mdo_function.name}_convex_lin",
            self.__mdo_function.f_type,
            self._convex_lin_jac,
            args=None,
            dim=self.__mdo_function.dim,
            outvars=self.__mdo_function.outvars,
            force_real=self.__mdo_function.force_real,
        )

    def __get_steps(
        self,
        x_new: ndarray,
    ) -> tuple[ndarray, ndarray]:
        """Return the steps on the direct and reciprocal variables.

        Args:
            x_new: The value of the inputs of the convex linearization function.

        Returns:
            Both the step on the direct variables
            and the step on the reciprocal variables.
        """
        step = x_new[self.__approx_indexes] - self.__x_vect[self.__approx_indexes]
        inv_step = zeros_like(step)
        nonzero_indexes = (absolute(step) > self.__sign_threshold).nonzero()
        inv_step[nonzero_indexes] = 1.0 / step[nonzero_indexes]
        return step, inv_step

    def _convex_lin_func(
        self,
        x_new: ndarray,
    ) -> ndarray:
        """Return the value of the convex linearization function.

        Args:
            x_new: The value of the inputs of the convex linearization function.

        Returns:
            The value of the outputs of the convex linearization function.
        """
        merged_vect = where(self.__approx_indexes, self.__x_vect, x_new)
        step, inv_step = self.__get_steps(x_new)
        value = (
            self.__mdo_function.evaluate(merged_vect)
            + matmul(self.__direct_coeffs, step)
            + matmul(self.__recipr_coeffs, inv_step)
        )
        if self.__mdo_function._dim == 1:
            return value[0]
        return value

    def _convex_lin_jac(
        self,
        x_new: ndarray,
    ) -> ndarray:
        """Return the Jacobian matrix of the convex linearization function.

        Args:
            x_new: The value of the inputs of the convex linearization function.

        Returns:
            The Jacobian matrix of the convex linearization function.
        """
        merged_vect = where(self.__approx_indexes, self.__x_vect, x_new)
        value = atleast_2d(self.__mdo_function.jac(merged_vect))
        _, inv_step = self.__get_steps(x_new)
        value[:, self.__approx_indexes] = self.__direct_coeffs + multiply(
            self.__recipr_coeffs, -(inv_step**2)
        )
        if self.__mdo_function._dim == 1:
            value = value[0, :]
        return value


class MDOQuadraticFunction(MDOFunction):
    r"""Scalar-valued quadratic multivariate function defined by

    * a *square* matrix :math:`A` of second-order coefficients
      :math:`(a_{ij})_{\substack{i = 1, \dots n \\ j = 1, \dots n}}`
    * a vector :math:`b` of first-order coefficients :math:`(b_i)_{i = 1, \dots n}`
    * and a scalar zero-order coefficient :math:`c`

    .. math::
        f(x)
        =
        c
        +
        \sum_{i = 1}^n b_i \, x_i
        +
        \sum_{i = 1}^n \sum_{j = 1}^n a_{ij} \, x_i \, x_j.
    """

    def __init__(
        self,
        quad_coeffs: ndarray,
        name: str,
        f_type: str | None = None,
        args: Sequence[str] = None,
        linear_coeffs: ndarray | None = None,
        value_at_zero: float | None = None,
    ):
        """
        Args:
            quad_coeffs: The second-order coefficients.
            name: The name of the function.
            f_type: The type of the linear function among :attr:`.MDOFunction.AVAILABLE_TYPES`.
                If None, the linear function will have no type.
            args: The names of the inputs of the linear function.
                If None, the inputs of the linear function will have no names.
            linear_coeffs: The first-order coefficients.
                If None, the first-order coefficients will be zero.
            value_at_zero: The zero-order coefficient.
                If None, the value at zero will be zero.
        """
        self._input_dim = 0
        self._quad_coeffs = None
        self._linear_coeffs = None
        self.quad_coeffs = quad_coeffs  # sets the input dimension
        new_args = MDOFunction.generate_args(self._input_dim, args)

        # Build the first-order term
        if linear_coeffs is not None:
            self._linear_part = MDOLinearFunction(
                linear_coeffs, f"{name}_lin", args=new_args
            )
            self.linear_coeffs = self._linear_part.coefficients
        self._value_at_zero = value_at_zero

        # Build the second-order term
        expr = self.build_expression(
            self._quad_coeffs, new_args, self._linear_coeffs, self._value_at_zero
        )
        super().__init__(
            self.__func, name, f_type, self.__grad, expr, args=new_args, dim=1
        )

    def __func(self, x_vect: ndarray) -> ndarray:
        """Compute the output of the quadratic function.

        Args:
            x_vect: The value of the inputs of the quadratic function.

        Returns:
            The value of the quadratic function.
        """
        value = multi_dot((x_vect.T, self._quad_coeffs, x_vect))
        if self._linear_coeffs is not None:
            value += self._linear_part(x_vect)
        if self._value_at_zero is not None:
            value += self._value_at_zero
        return value

    def __grad(self, x_vect: ndarray) -> ndarray:
        """Compute the gradient of the quadratic function.

        Args:
            x_vect: The value of the inputs of the quadratic function.

        Returns:
            The value of the gradient of the quadratic function.
        """
        gradient = matmul(self._quad_coeffs + self._quad_coeffs.T, x_vect)
        if self._linear_coeffs is not None:
            gradient += self._linear_part.jac(x_vect)
        return gradient

    @property
    def quad_coeffs(self) -> ndarray:
        """The second-order coefficients of the function.

        Raises:
            ValueError: If the coefficients are not passed
                as a 2-dimensional square ``ndarray``.
        """
        return self._quad_coeffs

    @quad_coeffs.setter
    def quad_coeffs(
        self,
        coefficients: ndarray,
    ) -> None:
        # Check the second-order coefficients
        if (
            not isinstance(coefficients, ndarray)
            or len(coefficients.shape) != 2
            or coefficients.shape[0] != coefficients.shape[1]
        ):
            raise ValueError(
                "Quadratic coefficients must be passed as a 2-dimensional "
                "square ndarray."
            )
        self._quad_coeffs = coefficients
        self._input_dim = self._quad_coeffs.shape[0]

    @property
    def linear_coeffs(self) -> ndarray:
        """The first-order coefficients of the function.

        Raises:
            ValueError: If the number of first-order coefficients is not consistent
                with the dimension of the input space.
        """
        if self._linear_coeffs is None:
            return zeros(self._input_dim)
        return self._linear_coeffs

    @linear_coeffs.setter
    def linear_coeffs(
        self,
        coefficients: ndarray,
    ) -> None:
        if coefficients.size != self._input_dim:
            raise ValueError(
                "The number of first-order coefficients must be equal "
                "to the input dimension."
            )
        self._linear_coeffs = coefficients

    @staticmethod
    def build_expression(
        quad_coeffs: ndarray,
        args: Sequence[str],
        linear_coeffs: linear_coeffs | None = None,
        value_at_zero: float | None = None,
    ):
        """Build the expression of the quadratic function.

        Args:
            quad_coeffs: The second-order coefficients.
            args: The names of the inputs of the function.
            linear_coeffs: The first-order coefficients.
                If None, the first-order coefficients will be zero.
            value_at_zero: The zero-order coefficient.
                If None, the value at zero will be zero.

        Returns:
            The expression of the quadratic function.
        """
        transpose_str = "'"
        expr = ""
        for index, line in enumerate(quad_coeffs.tolist()):
            arg = args[index]
            # Second-order expression
            line = quad_coeffs[index, :].tolist()
            expr += f"[{arg}]"
            expr += transpose_str if index == 0 else " "
            quad_coeffs_str = (MDOFunction.COEFF_FORMAT_ND.format(val) for val in line)
            expr += "[{}]".format(" ".join(quad_coeffs_str))
            expr += f"[{arg}]"
            # First-order expression
            if (
                linear_coeffs is not None
                and (linear_coeffs != zeros_like(linear_coeffs)).any()
            ):
                expr += " + " if index == 0 else "   "
                expr += "[{}]".format(
                    MDOFunction.COEFF_FORMAT_ND.format(linear_coeffs[0, index])
                )
                expr += transpose_str if index == 0 else " "
                expr += f"[{arg}]"
            # Zero-order expression
            if value_at_zero is not None and value_at_zero != 0.0 and index == 0:
                sign_str = "+" if value_at_zero > 0.0 else "-"
                expr += (" {} " + MDOFunction.COEFF_FORMAT_ND).format(
                    sign_str, abs(value_at_zero)
                )
            if index < quad_coeffs.shape[0] - 1:
                expr += "\n"
        return expr


class SetPtFromDatabase(MDOFunction):
    """Set a function and Jacobian from a database."""

    def __init__(
        self,
        database: Database,
        design_space: DesignSpace,
        mdo_function: MDOFunction,
        normalize: bool = False,
        jac: bool = True,
        x_tolerance: float = 1e-10,
    ):
        """
        Args:
            database: The database to read.
            design_space: The design space used for normalization.
            mdo_function: The function where the data from the database will be set.
            normalize: If True, the values of the inputs are unnormalized before call.
            jac: If True, a Jacobian pointer is also generated.
            x_tolerance: The tolerance on the distance between inputs.
        """
        self.__database = database
        self.__design_space = design_space
        self.__mdo_function = mdo_function
        self.__normalize = normalize
        self.__jac = jac
        self.__x_tolerance = x_tolerance

        self.__name = self.__mdo_function.name

        self.__mdo_function.func = self._f_from_db

        if jac:
            self.__mdo_function.jac = self._j_from_db

    def __read_in_db(
        self,
        x_n: ndarray,
        fname: str,
    ) -> ndarray:
        """Read the value of a function in the database for a given input value.

        Args:
            x_n: The value of the inputs to evaluate the function.
            fname: The name of the function.

        Returns:
            The value of the function if present in the database.

        Raises:
            ValueError: If the input value is not in the database.
        """
        if self.__normalize:
            x_db = self.__design_space.unnormalize_vect(x_n)
        else:
            x_db = x_n
        val = self.__database.get_f_of_x(fname, x_db, self.__x_tolerance)
        if val is None:
            msg = (
                "Function {} evaluation relies only on the database, "
                "and {}( x ) is not in the database for x={}."
            ).format(fname, fname, x_db)
            raise ValueError(msg)
        return val

    def _f_from_db(
        self,
        x_n: ndarray,
    ) -> ndarray:
        """Evaluate the function from the database.

        Args:
            x_n: The value of the inputs to evaluate the function.

        Returns:
            The value of the function read in the database.
        """
        return self.__read_in_db(x_n, self.__name)

    def _j_from_db(
        self,
        x_n: ndarray,
    ) -> ndarray:
        """Evaluate the Jacobian function from the database.

        Args:
            x_n: The value of the inputs to evaluate the Jacobian function.

        Returns:
            The value of the Jacobian function read in the database.
        """
        return self.__read_in_db(x_n, f"@{self.__name}")

    @property
    def expects_normalized_inputs(self) -> bool:
        return self.__normalize
