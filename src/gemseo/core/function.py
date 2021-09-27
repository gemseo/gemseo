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
#                        documentation
#        :author: Francois Gallard, Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Base class to describe a function."""
from __future__ import division, unicode_literals

import logging
from multiprocessing import Value
from numbers import Number
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    NoReturn,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from numpy import abs as np_abs
from numpy import (
    absolute,
    add,
    array,
    atleast_1d,
    atleast_2d,
    concatenate,
    empty,
    hstack,
    matmul,
    multiply,
    ndarray,
    ones_like,
    reshape,
    subtract,
    vstack,
    where,
    zeros,
    zeros_like,
)
from numpy.linalg import multi_dot, norm
from six import string_types

from gemseo.algos.database import Database
from gemseo.algos.design_space import DesignSpace
from gemseo.core.discipline import MDODiscipline
from gemseo.utils.data_conversion import DataConversion
from gemseo.utils.derivatives_approx import ComplexStep, FirstOrderFD

LOGGER = logging.getLogger(__name__)

OperandType = Union[ndarray, Number]
OperatorType = Callable[[OperandType, OperandType], OperandType]


class MDOFunction(object):
    """The standard definition of an array-based function with algebraic operations.

    :class:`MDOFunction` is the key class
    to define the objective function, the constraints and the observables
    of an :class:`.OptimizationProblem`.

    A :class:`MDOFunction` is initialized from an optional callable and a name,
    e.g. :code:`func = MDOFunction(lambda x: 2*x, "my_function")`.

    .. note::

       The callable can be set to :code:`None`
       when the user does not want to use a callable
       but a database to browse for the output vector corresponding to an input vector
       (see :meth:`set_pt_from_database`).

    The following information can also be provided at initialization:

    - the type of the function,
      e.g. :code:`f_type="obj"` if the function will be used as an objective
      (see :attr:`AVAILABLE_TYPES` for the available types),
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
    e.g. :attr:`args`.

    The original function and Jacobian function
    can be accessed with the properties :attr:`func` and :attr:`jac`.

    A :class:`MDOFunction` is callable:
    :code:`output = func(array([3.])) # expected: array([6.])`.

    Elementary operations can be performed with :class:`MDOFunction` instances:
    addition (:code:`func = func1 + func2` or :code:`func = func1 + offset`),
    subtraction (:code:`func = func1 - func2` or :code:`func = func1 - offset`),
    multiplication (:code:`func = func1 * func2` or :code:`func = func1 * factor`)
    and opposite  (:code:`func = -func1`).
    It is also possible to build a :class:`MDOFunction`
    as a concatenation of :class:`MDOFunction` objects:
    :code:`func = MDOFunction.concatenate([func1, func2, func3], "my_func_123"`).

    Moreover, a :class:`MDOFunction` can be approximated
    with either a first-order or second-order Taylor polynomial at a given input vector,
    using respectively :meth:`linear_approximation` and :meth:`quadratic_approx`;
    such an approximation is also a :class:`MDOFunction`.

    Lastly, the user can check the Jacobian function by means of approximation methods
    (see :meth:`check_grad`).
    """

    TYPE_OBJ = "obj"  # type: str
    """The type of function for objective."""

    TYPE_EQ = "eq"  # type: str
    """The type of function for equality constraint."""

    TYPE_INEQ = "ineq"  # type: str
    """The type of function for inequality constraint."""

    TYPE_OBS = "obs"  # type: str
    """The type of function for observable."""

    AVAILABLE_TYPES = [TYPE_OBJ, TYPE_EQ, TYPE_INEQ, TYPE_OBS]  # type: List[str]
    """The available types of function."""

    DICT_REPR_ATTR = [
        "name",
        "f_type",
        "expr",
        "args",
        "dim",
        "special_repr",
    ]  # type: List[str]
    """The names of the attributes to be serialized."""

    DEFAULT_ARGS_BASE = "x"  # type: str
    """The default name base for the inputs."""

    INDEX_PREFIX = "!"  # type: str
    """The character used to separate a name base and a prefix, e.g. `"x!1`."""

    COEFF_FORMAT_1D = "{:.2e}"  # type: str
    """The format to be applied to a number when represented in a vector."""
    # ensure that coefficients strings have same length

    COEFF_FORMAT_ND = "{: .2e}"  # type: str
    """The format to be applied to a number when represented in a matrix."""
    # ensure that coefficients strings have same length

    # N.B. the space character ensures same length whatever the sign of the coefficient

    def __init__(
        self,
        func,  # type: Optional[Callable[[ndarray],ndarray]]
        name,  # type: str
        f_type=None,  # type: Optional[str]
        jac=None,  # type: Optional[Callable[[ndarray],ndarray]]
        expr=None,  # type: Optional[str]
        args=None,  # type: Optional[Sequence[str]]
        dim=None,  # type: Optional[int]
        outvars=None,  # type: Optional[Sequence[str]]
        force_real=False,  # type: bool
        special_repr=None,  # type: Optional[str]
    ):  # type: (...) -> None
        """
        Args:
            func: The original function to be actually called.
                If None, the function will not have an original function.
            name: The name of the function.
            f_type: The type of the function among :attr:`AVAILABLE_TYPES`.
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
        super(MDOFunction, self).__init__()
        self._n_calls = Value("i", 0)
        # Initialize attributes
        self._f_type = None
        self._func = NotImplementedCallable()
        self._jac = NotImplementedCallable()
        self._name = None
        self._args = None
        self._expr = None
        self._dim = None
        self._outvars = None
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

    @property
    def n_calls(self):  # type: (...) -> int
        """The number of times the function has been evaluated.

        This count is both multiprocess- and multithread-safe, thanks to the locking
        process used by :meth:`evaluate`.
        """
        return self._n_calls.value

    @n_calls.setter
    def n_calls(
        self,
        value,  # type: int
    ):  # type: (...) -> None
        with self._n_calls.get_lock():
            self._n_calls.value = value

    @property
    def func(self):  # type: (...) -> Callable[[ndarray],ndarray]
        """The function to be evaluated from a given input vector."""
        return self.__counted_f

    def __counted_f(
        self,
        x_vect,  # type: ndarray
    ):  # type: (...) -> ndarray
        """Evaluate the function and store the result in :attr:`last_eval`.

        This evaluation is both multiprocess- and multithread-safe,
        thanks to a locking process.

        Args:
            x_vect: The value of the inputs of the function.

        Returns:
            The value of the outputs of the function.
        """
        with self._n_calls.get_lock():
            self._n_calls.value += 1
        val = self._func(x_vect)
        self.last_eval = val
        return val

    @func.setter
    def func(
        self,
        f_pointer,  # type: Callable[[ndarray],ndarray]
    ):  # type: (...) -> None
        self._n_calls.value = 0
        self._func = f_pointer

    def __call__(
        self,
        x_vect,  # type: ndarray
    ):  # type: (...) -> ndarray
        """Evaluate the function.

        This method can cast the result to real value
        according to the value of the attribute :attr:`force_real`.

        Args:
            x_vect: The value of the inputs of the function.

        Returns:
            The value of the outputs of the function.
        """
        val = self.evaluate(x_vect, self.force_real)
        return val

    def evaluate(
        self,
        x_vect,  # type: ndarray
        force_real=False,  # type: bool
    ):  # type: (...) -> ndarray
        """Evaluate the function and store the dimension of the output space.

        Args:
            x_vect: The value of the inputs of the function.
            force_real: If True, cast the result to real value.

        Returns:
            The value of the output of the function.
        """
        val = self.__counted_f(x_vect)
        if force_real:
            val = val.real
        self.dim = atleast_1d(val).shape[0]
        return val

    @property
    def name(self):  # type: (...) -> str
        """The name of the function.

        Raises:
            TypeError: If the name of the function is not a string.
        """
        return self._name

    @name.setter
    def name(
        self,
        name,  # type: str
    ):  # type: (...) -> None
        if name is not None and not isinstance(name, string_types):
            raise TypeError(
                "MDOFunction name must be a string; got {} instead.".format(type(name))
            )
        self._name = name

    @property
    def f_type(self):  # type: (...) -> str
        """The type of the function, among :attr:`AVAILABLE_TYPES`.

        Raises:
            ValueError: If the type of function is not available.
        """
        return self._f_type

    @f_type.setter
    def f_type(
        self,
        f_type,  # type: str
    ):  # type: (...) -> None
        if f_type is not None and f_type not in self.AVAILABLE_TYPES:
            raise ValueError(
                "MDOFunction type must be among {}; got {} instead.".format(
                    self.AVAILABLE_TYPES, f_type
                )
            )
        self._f_type = f_type

    @property
    def jac(self):  # type: (...) -> Callable[[ndarray],ndarray]
        """The Jacobian function to be evaluated from a given input vector.

        Raises:
            TypeError: If the Jacobian function is not callable.
        """
        return self._jac

    @jac.setter
    def jac(
        self,
        jac,  # type: Callable[[ndarray],ndarray]
    ):  # type: (...) -> None
        if jac is not None:
            if not callable(jac):
                raise TypeError("Jacobian function must be callable.")
        self._jac = jac

    @property
    def args(self):  # type: (...) -> Sequence[str]
        """The names of the inputs of the function."""
        return self._args

    @args.setter
    def args(
        self,
        args,  # type: Optional[Union[Iterable[str],ndarray]]
    ):  # type: (...) -> None
        if args is not None:
            if isinstance(args, ndarray):
                self._args = args.tolist()
            else:
                self._args = list(args)
        else:
            self._args = None

    @property
    def expr(self):  # type: (...) -> str
        """The expression of the function, e.g. `"2*x"`.

        Raises:
            TypeError: If the expression is not a string.
        """
        return self._expr

    @expr.setter
    def expr(
        self,
        expr,  # type: str
    ):  # type: (...) -> None
        if expr is not None and not isinstance(expr, string_types):
            raise TypeError(
                "Expression must be a string; got {} instead.".format(type(expr))
            )
        self._expr = expr

    @property
    def dim(self):  # type: (...) -> int
        """The dimension of the output space of the function.

        Raises:
            TypeError: If the dimension of the output space is not an integer.
        """
        return self._dim

    @dim.setter
    def dim(
        self,
        dim,  # type: int
    ):  # type: (...) -> None
        if dim is not None and not int(dim) == dim:
            raise TypeError(
                "The dimension must be an integer; got {} instead.".format(type(int))
            )
        self._dim = dim

    @property
    def outvars(self):  # type: (...) -> Sequence[str]
        """The names of the outputs of the function."""
        return self._outvars

    @outvars.setter
    def outvars(
        self,
        outvars,  # type: Sequence[str]
    ):  # type: (...) -> None
        if outvars is not None:
            if isinstance(outvars, ndarray):
                self._outvars = outvars.tolist()
            else:
                self._outvars = list(outvars)
        else:
            self._outvars = None

    def is_constraint(self):  # type: (...) -> bool
        """Check if the function is a constraint.

        The type of a constraint function is either 'eq' or 'ineq'.

        Returns:
            Whether the function is a constraint.
        """
        return self.f_type in [self.TYPE_EQ, self.TYPE_INEQ]

    def __repr__(self):  # type: (...) -> str
        return self.special_repr or self.default_repr

    @property
    def default_repr(self):  # type: (...) -> str
        """The default string representation of the function."""
        str_repr = self.name
        if self.has_args():
            arguments = ", ".join(self.args)
            str_repr += "({})".format(arguments)

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

    def has_jac(self):  # type: (...) -> bool
        """Check if the function has an implemented Jacobian function.

        Returns:
            Whether the function has an implemented Jacobian function.
        """
        return self.jac is not None and not isinstance(
            self._jac, NotImplementedCallable
        )

    def has_dim(self):  # type: (...) -> bool
        """Check if the dimension of the output space of the function is defined.

        Returns:
            Whether the dimension of the output space of the function is defined.
        """
        return self.dim is not None

    def has_outvars(self):  # type: (...) -> bool
        """Check if the outputs of the function have names.

        Returns:
            Whether the outputs of the function have names.
        """
        return self.outvars is not None

    def has_expr(self):  # type: (...) -> bool
        """Check if the function has an expression.

        Returns:
            Whether the function has an expression.
        """
        return self.expr is not None

    def has_args(self):  # type: (...) -> bool
        """Check if the inputs of the function have names.

        Returns:
            Whether the inputs of the function have names.
        """
        return self.args is not None

    def has_f_type(self):  # type: (...) -> bool
        """Check if the function has a type.

        Returns:
            Whether the function has a type.
        """
        return self.f_type is not None

    def apply_operator(
        self,
        other_f,  # type: Union[Number,MDOFunction]
        operator,  # type: OperatorType
        operator_repr,  # type: str
    ):  # type: (...) -> MDOFunction
        """Apply an operator to the function and another operand.

        This operator supports automatic differentiation
        if both functions have an implemented Jacobian function.

        Args:
            other_f: The other operand to which the operator applies.
            operator: The operator as a function pointer.
            operator_repr: The representation of the operator.

        Returns:
            A new function resulting from the application of the operator
            to the function and the other operand.

        Raises:
            TypeError: If the operator is unsupported for the type of the other operand.
        """
        if isinstance(other_f, MDOFunction):
            return self.__apply_operator_tofunction(other_f, operator, operator_repr)

        if isinstance(other_f, Number):
            return self.__apply_operator_to_number(other_f, operator, operator_repr)

        raise TypeError(
            "Unsupported operator for MDOFunction and {}.".format(type(other_f))
        )

    def __apply_operator_tofunction(
        self,
        other_f,  # type: MDOFunction
        operator,  # type: OperatorType
        operator_repr,  # type: str
    ):  # type: (...) -> MDOFunction
        """Apply an operator to the function and another function.

        This operator supports automatic differentiation
        if both functions have an implemented Jacobian function.

        Args:
            other_f: The other function.
            operator: The operator as a function pointer.
            operator_repr: The representation of the operator.

        Returns:
            A function resulting from the application of the operator.
        """
        assert isinstance(other_f, MDOFunction)

        def add_f_pt(
            x_vect,  # type: ndarray
        ):  # type:(...) -> ndarray
            """Evaluate the functions and apply the operator to their outputs.

            Args:
                x_vect: The values of the inputs of the functions.

            Returns:
                The value of the outputs of the functions modified by the operator.
            """
            selfval = self(x_vect)
            otherval = other_f(x_vect)
            return operator(selfval, otherval)

        add_name = self.name + operator_repr + other_f.name
        self_oper_f = MDOFunction(
            add_f_pt,
            add_name,
            args=self.args,
            f_type=self.f_type,
            dim=self.dim,
            outvars=self.outvars,
        )

        if self.has_jac() and other_f.has_jac():

            def add_jac(
                x_vect,  # type: ndarray
            ):  # type:(...) -> ndarray
                """Evaluate the Jacobian functions and apply the operator to the result.

                Args:
                    x_vect: The values of the inputs of the Jacobian functions.

                Returns:
                    The value of the outputs of the functions modified by the operator.
                """
                self_jac = self._jac(x_vect)
                other_jac = other_f.jac(x_vect)
                return operator(self_jac, other_jac)

            self_oper_f.jac = add_jac

        if self.has_expr() and other_f.has_expr():
            f_expr = self.expr + operator_repr + other_f.expr
            self_oper_f.expr = f_expr

        if self.has_args() and other_f.has_args():
            args = sorted(list(set(self.args + other_f.args)))
            self_oper_f.args = args

        if self.has_f_type():
            self_oper_f.f_type = self.f_type
        elif other_f.has_f_type():
            self_oper_f.f_type = other_f.f_type

        return self_oper_f

    def __apply_operator_to_number(
        self,
        other_val,  # type: Number
        operator,  # type: OperatorType
        operator_repr,  # type: str
    ):  # type: (...) -> MDOFunction
        """Apply an operator to the function and a number.

        This operator supports automatic differentiation
        if the function has an implemented Jacobian function.

        Args:
            other_val: The number.
            operator: The operator as a function pointer.
            operator_repr: The representation of the operator.

        Returns:
            A function resulting from the application of the operator.
        """
        assert isinstance(other_val, Number)

        def add_f_pt(
            x_vect,  # type: ndarray
        ):  # type:(...) -> ndarray
            """Evaluate the function and apply the operator to the value of its outputs.

            Args:
                x_vect: The values of the inputs of the function.

            Returns:
                The value of the outputs of the function modified by the operator.
            """
            selfval = self(x_vect)
            return operator(selfval, other_val)

        add_name = self.name + operator_repr + str(other_val)
        self_oper_f = MDOFunction(
            add_f_pt,
            add_name,
            args=self.args,
            f_type=self.f_type,
            dim=self.dim,
            outvars=self.outvars,
        )

        if self.has_jac():
            self_oper_f.jac = self.jac

        if self.has_expr():
            self_oper_f.expr = self.expr + operator_repr + str(other_val)

        if self.has_args():
            self_oper_f.args = self.args

        if self.has_f_type():
            self_oper_f.f_type = self.f_type

        return self_oper_f

    def __add__(
        self,
        other_f,  # type: MDOFunction
    ):  # type: (...) -> MDOFunction
        """Operator defining the sum of the function and another one.

        This operator supports automatic differentiation
        if both functions have an implemented Jacobian function.

        Args:
            other_f: The other function.

        Returns:
            The sum of the function and the other one.
        """
        return self.apply_operator(other_f, operator=add, operator_repr="+")

    def __sub__(
        self,
        other_f,  # type: MDOFunction
    ):  # type: (...) -> MDOFunction
        """Operator defining the difference of the function and another one.

        This operator supports automatic differentiation
        if both functions have an implemented Jacobian function.

        Args:
            other_f: The other function.

        Returns:
            The difference of the function and the other one.
        """
        return self.apply_operator(other_f, operator=subtract, operator_repr="-")

    def __neg__(self):  # type: (...) -> MDOFunction
        """Operator defining the opposite of the function.

        This operator supports automatic differentiation
        if the function has an implemented Jacobian function.

        Returns:
            The opposite of the function.
        """

        def min_pt(
            x_vect,  # type: ndarray
        ):  # type: (...) -> ndarray
            """Evaluate the function and return its opposite value.

            Args:
                x_vect: The value of the inputs of the function.

            Returns:
                The opposite of the value of the outputs of the function.
            """
            return -self(x_vect)

        min_name = "-{}".format(self.name)
        min_self = MDOFunction(
            min_pt,
            min_name,
            args=self.args,
            f_type=self.f_type,
            dim=self.dim,
            outvars=self.outvars,
        )

        if self.has_jac():

            def min_jac(
                x_vect,  # type: ndarray
            ):  # type: (...) -> ndarray
                """Evaluate the Jacobian function and return its opposite value.

                Args:
                    x_vect: The value of the inputs of the Jacobian function.

                Returns:
                    The opposite of the value of the Jacobian function.
                """
                return -self.jac(x_vect)  # pylint: disable=E1102

            min_self.jac = min_jac

        if self.has_expr():
            min_self.expr = self.expr
            min_self.expr = min_self.expr.replace("+", "++")
            min_self.expr = min_self.expr.replace("-", "+")
            min_self.expr = min_self.expr.replace("++", "-")
            min_self.expr = "-" + min_self.expr

        return min_self

    def __mul__(
        self,
        other,  # type:Union[MDOFunction, Number]
    ):  # type: (...) -> MDOFunction
        """Operator defining the multiplication of the function and another operand.

        This operator supports automatic differentiation
        if the different functions have an implemented Jacobian function.

        Args:
            other: The other operand.

        Returns:
            The multiplication of the function with a number or another function.

        Raises:
            TypeError: If the other operand is
                neither a number nor a :class:`MDOFunction`.
        """
        is_number = isinstance(other, Number)
        is_func = isinstance(other, MDOFunction)
        if not is_number and not is_func:
            raise TypeError(
                "Unuspported * operand for MDOFunction and {}.".format(type(other))
            )

        if is_func:

            def mul_f_pt(
                x_vect,  # type: ndarray
            ):  # type: (...) -> ndarray
                """Evaluate both functions and multiply their output values.

                Args:
                    x_vect: The value of the inputs of the function.

                Returns:
                    The product of the output values of the functions.
                """
                return self(x_vect) * other(x_vect)

            out_name = "{}*{}".format(self.name, other.name)

        else:  # Func is a number

            def mul_f_pt(
                x_vect,  # type: ndarray
            ):  # type: (...) -> ndarray
                """Evaluate the function and multiply its output value with a number.

                Args:
                    x_vect: The value of the inputs of the function.

                Returns:
                    The product of the output value of the function with the number.
                """
                return self(x_vect) * other

            out_name = "{}*{}".format(self.name, other)

        self_oper_f = MDOFunction(
            mul_f_pt,
            out_name,
            args=self.args,
            f_type=self.f_type,
            dim=self.dim,
            outvars=self.outvars,
        )
        if is_func:
            if self.has_jac() and other.has_jac():

                def mult_jac(
                    x_vect,  # type: ndarray
                ):  # type: (...) -> ndarray
                    """Evaluate both functions and multiply their output values.

                    Args:
                        x_vect: The value of the inputs of the function.

                    Returns:
                        The product of the output values of the functions.
                    """
                    self_f = self(x_vect)
                    other_f = other(x_vect)
                    self_jac = self._jac(x_vect)
                    other_jac = other.jac(x_vect)
                    return self_jac * other_f + other_jac * self_f

                self_oper_f.jac = mult_jac
        elif self.has_jac():

            def mult_jac(
                x_vect,  # type: ndarray
            ):  # type: (...) -> ndarray
                """Evaluate both functions and multiply their output values.

                Args:
                    x_vect: The value of the inputs of the function.

                Returns:
                    The product of the output values of the functions.
                """
                return self._jac(x_vect) * other

            self_oper_f.jac = mult_jac

        if self.has_expr():
            if is_func and other.has_expr():
                f_expr = "({})*({})".format(self.expr, other.expr)
                self_oper_f.expr = f_expr
            if is_number:
                f_expr = "({})*{}".format(self.expr, other)
                self_oper_f.expr = f_expr

        if is_func and self.has_args() and other.has_args():
            args = sorted(list(set(self.args + other.args)))
            self_oper_f.args = args

        if self.has_f_type():
            self_oper_f.f_type = self.f_type
        elif is_func and other.has_f_type():
            self_oper_f.f_type = other.f_type

        return self_oper_f

    def offset(
        self,
        value,  # type: Union[Number,ndarray]
    ):  # type: (...) -> MDOFunction
        """Add an offset value to the function.

        Args:
            value: The offset value.

        Returns:
            The offset function.
        """

        def wrapped_function(
            x_vect,  # type: ndarray
        ):  # type: (...) -> ndarray
            """The function applying the offset to the original :class:`MDOFunction`.

            Args:
                x_vect: The value of the inputs of the function.

            Returns:
                The evaluation of the function plus the offset.
            """
            return self(x_vect) + value

        expr = self.expr
        if expr is not None:
            if value < 0:
                expr += " - {}".format(-value)
            else:
                expr += " + {}".format(abs(value))
        else:
            offset_is_negative = value < 0
            if hasattr(offset_is_negative, "__len__"):
                offset_is_negative = offset_is_negative.all()
            if offset_is_negative:
                expr = "{} - {}".format(self.name, -value)
            else:
                expr = "{} + {}".format(self.name, abs(value))
        offset_func = MDOFunction(
            wrapped_function,
            name=self.name,
            f_type=self.f_type,
            expr=expr,
            args=self.args,
            dim=self.dim,
            jac=self.jac,
            outvars=self.outvars,
        )

        return offset_func
        # Nothing to do for Jacobian since it is unchanged.

    def restrict(
        self,
        frozen_indexes,  # type: ndarray
        frozen_values,  # type: ndarray
        input_dim,  # type: int
        name=None,  # type: Optional[str]
        f_type=None,  # type: Optional[str]
        expr=None,  # type: Optional[str]
        args=None,  # type: Optional[Sequence[str]]
    ):  # type:(...) -> MDOFunction
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

        Raises:
            ValueError: If the numbers of frozen indexes and values are different.
        """
        # Check the shapes of the passed arrays
        if frozen_indexes.shape != frozen_values.shape:
            raise ValueError(
                "Arrays of frozen indexes and values must have same shape."
            )

        active_indexes = array([i for i in range(input_dim) if i not in frozen_indexes])

        def extend_subvect(
            x_subvect,  # type: ndarray
        ):  # type:(...) -> ndarray
            """Extend an input vector of the restriction with the frozen values.

            Args:
                x_subvect: The values of the inputs of the restriction.

            Returns:
                The extended input vector.
            """
            x_vect = empty(input_dim)
            x_vect[active_indexes] = x_subvect
            x_vect[frozen_indexes] = frozen_values
            return x_vect

        def func(
            x_subvect,  # type: ndarray
        ):  # type: (...) -> ndarray
            """Evaluate the restriction.

            Args:
                x_subvect: The value of the inputs of the restriction.

            Returns:
                The value of the outputs of the restriction.
            """
            x_vect = extend_subvect(x_subvect)
            value = self.__call__(x_vect)
            return value

        # Build the name of the restriction
        if name is None and args is not None:
            name = "{}_wrt_{}".format(self.name, "_".join(args))
        elif name is None:
            name = "{}_restriction".format(self.name)

        restriction = MDOFunction(
            func,
            name,
            f_type,
            expr=expr,
            args=args,
            dim=self.dim,
            outvars=self.outvars,
            force_real=self.force_real,
        )

        if self.has_jac():

            def jac(
                x_subvect,  # type: ndarray
            ):  # type: (...) -> ndarray
                """Compute the Jacobian matrix of the restriction.

                Args:
                    x_subvect: The value of the inputs of the restriction.

                Returns:
                    The Jacobian matrix of the restriction.
                """
                x_vect = extend_subvect(x_subvect)
                full_jac = self.jac(x_vect)
                if len(full_jac.shape) == 1:
                    sub_jac = full_jac[active_indexes]
                else:
                    sub_jac = full_jac[:, active_indexes]
                return sub_jac

            restriction.jac = jac

        return restriction

    def linear_approximation(
        self,
        x_vect,  # type: ndarray
        name=None,  # type: Optional[str]
        f_type=None,  # type: Optional[str]
        args=None,  # type: Optional[Sequence[str]]
    ):  # type: (...) -> MDOLinearFunction
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
        func_val = self.__call__(x_vect)
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
        x_vect,  # type: ndarray
        approx_indexes=None,  # type: Optional[ndarray]
        sign_threshold=1e-9,  # type: float
    ):  # type: (...) -> MDOFunction
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

        Raises:
            ValueError: If the length of boolean array
                and the number of inputs of the functions are inconsistent.
            AttributeError: If the function does not have a Jacobian function.
        """
        # Check the approximation indexes
        if approx_indexes is None:
            approx_indexes = ones_like(x_vect, dtype=bool)
        elif approx_indexes.shape != x_vect.shape or approx_indexes.dtype != "bool":
            raise ValueError(
                "The approximation array must an array of booleans with "
                "the same shape as the function argument."
            )

        # Get the function Jacobian matrix
        if not self.has_jac():
            raise AttributeError(
                "Function Jacobian unavailable for convex linearization."
            )
        jac = atleast_2d(self.jac(x_vect))

        # Build the coefficients matrices
        coeffs = jac[:, approx_indexes]
        direct_coeffs = where(coeffs > sign_threshold, coeffs, 0.0)
        recipr_coeffs = multiply(
            -where(-coeffs > sign_threshold, coeffs, 0.0), x_vect[approx_indexes] ** 2
        )

        def get_steps(
            x_new,  # type: ndarray
        ):  # type: (...) -> Tuple[ndarray,ndarray]
            """Return the steps on the direct and reciprocal variables.

            Args:
                x_new: The value of the inputs of the convex linearization function.

            Returns:
                Both the step on the direct variables
                and the step on the reciprocal variables.
            """
            step = x_new[approx_indexes] - x_vect[approx_indexes]
            inv_step = zeros_like(step)
            nonzero_indexes = (absolute(step) > sign_threshold).nonzero()
            inv_step[nonzero_indexes] = 1.0 / step[nonzero_indexes]
            return step, inv_step

        def convex_lin_func(
            x_new,  # type: ndarray
        ):  # type: (...) -> ndarray
            """Return the value of the convex linearization function.

            Args:
                x_new: The value of the inputs of the convex linearization function.

            Returns:
                The value of the outputs of the convex linearization function.
            """
            merged_vect = where(approx_indexes, x_vect, x_new)
            step, inv_step = get_steps(x_new)
            value = (
                self.__call__(merged_vect)
                + matmul(direct_coeffs, step)
                + matmul(recipr_coeffs, inv_step)
            )
            if self._dim == 1:
                return value[0]
            return value

        def convex_lin_jac(
            x_new,  # type: ndarray
        ):  # type: (...) -> ndarray
            """Return the Jacobian matrix of the convex linearization function.

            Args:
                x_new: The value of the inputs of the convex linearization function.

            Returns:
                The Jacobian matrix of the convex linearization function.
            """
            merged_vect = where(approx_indexes, x_vect, x_new)
            value = atleast_2d(self.jac(merged_vect))
            _, inv_step = get_steps(x_new)
            value[:, approx_indexes] = direct_coeffs + multiply(
                recipr_coeffs, -(inv_step ** 2)
            )
            if self._dim == 1:
                value = value[0, :]
            return value

        # Build the convex linearization of the function
        convex_lin_approx = MDOFunction(
            convex_lin_func,
            "{}_convex_lin".format(self.name),
            self.f_type,
            convex_lin_jac,
            args=None,
            dim=self.dim,
            outvars=self.outvars,
            force_real=self.force_real,
        )

        return convex_lin_approx

    def quadratic_approx(
        self,
        x_vect,  # type: ndarray
        hessian_approx,  # type: ndarray
        args=None,  # type: Optional[Sequence[str]]
    ):  # type: (...) ->MDOQuadraticFunction
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
        zero_coeff = matmul(0.5 * hess_dot_vect - gradient, x_vect) + self.__call__(
            x_vect
        )

        # Build the second-order Taylor polynomial
        quad_approx = MDOQuadraticFunction(
            quad_coeffs=quad_coeffs,
            linear_coeffs=linear_coeffs,
            value_at_zero=zero_coeff,
            name="{}_quadratized".format(self.name),
            args=args if args else self.args,
        )

        return quad_approx

    def check_grad(
        self,
        x_vect,  # type: ndarray
        method="FirstOrderFD",  # type: str
        step=1e-6,  # type: float
        error_max=1e-8,  # type: float
    ):  # type: (...) -> None
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

        def rel_err(
            a_vect,  # type: ndarray
            b_vect,  # type: ndarray
        ):  # type: (...) -> float
            """Compute the 2-norm of the difference between two vectors.

            Normalize it with the 2-norm of the reference vector
            if the latter is greater than the maximal error.

            Args:
                a_vect: A first vector.
                b_vect: A second vector, used as a reference.

            Returns:
                The difference between two vectors,
                normalized if required.
            """
            n_b = norm(b_vect)
            if n_b > error_max:
                return norm(a_vect - b_vect) / norm(b_vect)
            return norm(a_vect - b_vect)

        def filt_0(
            arr,  # type: ndarray
            floor_value=1e-6,  # type: float
        ):  # type: (...) -> ndarray
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
        rel_error = rel_err(anal_grad, apprx_grad)
        succeed = rel_error < error_max
        if not succeed:
            LOGGER.error("Function jacobian is wrong %s", str(self))
            LOGGER.error("Error =\n%s", str(filt_0(anal_grad - apprx_grad)))
            LOGGER.error("Analytic jacobian=\n%s", str(filt_0(anal_grad)))
            LOGGER.error("Approximate step gradient=\n%s", str(filt_0(apprx_grad)))
            raise ValueError("Function jacobian is wrong {}.".format(self))

    def get_data_dict_repr(self):  # type: (...) -> Dict[str,Union[str,int,List[str]]]
        """Create a dictionary representation of the function.

        This is used for serialization.
        The pointers to the functions are removed.

        Returns:
            Some attributes of the function indexed by their names.
            See :attr:`DICT_REPR_ATTR`
        """
        repr_dict = {}
        for attr_name in self.DICT_REPR_ATTR:
            attr = getattr(self, attr_name)
            if attr is not None:
                repr_dict[attr_name] = attr
        return repr_dict

    @staticmethod
    def init_from_dict_repr(**kwargs):  # type: (...) -> MDOFunction
        """Initialize a new function.

        This is typically used for deserialization.

        Args:
            **kwargs: The attributes from :attr:`DICT_REPR_ATTR`.

        Returns:
            A function initialized from the provided data.

        Raises:
            ValueError: If the name of an argument is not in :attr:`DICT_REPR_ATTR`.
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
        database,  # type: Database
        design_space,  # type: DesignSpace
        normalize=False,  # type: bool
        jac=True,  # type: bool
        x_tolerance=1e-10,  # type: float
    ):  # type: (...) -> None
        """Set the original function and Jacobian function from a database.

        For a given input vector,
        the method :meth:`func` will return
        either the output vector stored in the database
        if the input vector is present
        or `None`.
        The same for the method :meth:`jac`.

        Args:
            database: The database to read.
            design_space: The design space used for normalization.
            normalize: If True, the values of the inputs are unnormalized before call.
            jac: If True, a Jacobian pointer is also generated.
            x_tolerance: The tolerance on the distance between inputs.
        """
        name = self.name

        def read_in_db(
            x_n,  # type: ndarray
            fname,  # type: str
        ):  # type: (...) -> ndarray
            """Read the value of a function in the database for a given input value.

            Args:
                x_n: The value of the inputs to evaluate the function.
                fname: The name of the function.

            Returns:
                The value of the function if present in the database.

            Raises:
                ValueError: If the input value is not in the database.
            """
            if normalize:
                x_db = design_space.unnormalize_vect(x_n)
            else:
                x_db = x_n
            val = database.get_f_of_x(fname, x_db, x_tolerance)
            if val is None:
                msg = (
                    "Function {} evaluation relies only on the database, "
                    "and {}( x ) is not in the database for x={}."
                ).format(fname, fname, x_db)
                raise ValueError(msg)
            return val

        def f_from_db(
            x_n,  # type: ndarray
        ):  # type: (...) -> ndarray
            """Evaluate the function from the database.

            Args:
                x_n: The value of the inputs to evaluate the function.

            Returns:
                The value of the function read in the database.
            """
            return read_in_db(x_n, name)

        def j_from_db(
            x_n,  # type: ndarray
        ):  # type: (...) -> ndarray
            """Evaluate the Jacobian function from the database.

            Args:
                x_n: The value of the inputs to evaluate the Jacobian function.

            Returns:
                The value of the Jacobian function read in the database.
            """
            return read_in_db(x_n, "@{}".format(name))

        self.func = f_from_db
        if jac:
            self.jac = j_from_db

    @staticmethod
    def generate_args(
        input_dim,  # type: int
        args=None,  # type: Optional[Sequence[str]]
    ):  # type: (...) -> Sequence[str]
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
        args_base,  # type: str
        input_dim,  # type: int
    ):  # type: (...) -> List[str]
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
        functions,  # type: Iterable[MDOFunction]
        name,  # type: str
        f_type=None,  # type: Optional[str]
    ):  # type: (...) -> MDOFunction
        """Concatenate functions.

        Args:
            functions: The functions to be concatenated.
            name: The name of the concatenation function.
            f_type: The type of the concatenation function.
                If None, the function will have no type.

        Returns:
            The concatenation of the functions.
        """

        def concat_func(
            x_vect,  # type: ndarray
        ):  # type: (...) -> ndarray
            """Concatenate the values of the outputs of the functions.

            Args:
                x_vect: The value of the inputs of the functions.

            Returns:
                The concatenation of the values of the outputs of the functions.
            """
            return concatenate([atleast_1d(func(x_vect)) for func in functions])

        def concat_jac(
            x_vect,  # type: ndarray
        ):  # type: (...) -> ndarray
            """Concatenate the outputs of the Jacobian functions.

            Args:
                x_vect: The value of the inputs of the Jacobian functions.

            Returns:
                The concatenation of the outputs of the Jacobian functions.
            """
            return vstack([atleast_2d(func.jac(x_vect)) for func in functions])

        dim = sum([func.dim for func in functions])
        outvars_list = [func.outvars for func in functions]
        if None in outvars_list:
            outvars = None
        else:
            outvars = [out_var for outvars in outvars_list for out_var in outvars]
        return MDOFunction(
            concat_func, name, f_type, concat_jac, dim=dim, outvars=outvars
        )


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
        coefficients,  # type: ndarray
        name,  # type: str
        f_type=None,  # type: Optional[str]
        args=None,  # type: Optional[Sequence[str]]
        value_at_zero=0.0,  # type: Union[ndarray,Number]
    ):
        """
        Args:
            coefficients: The coefficients :math:`A` of the linear function.
            name: The name of the linear function.
            f_type: The type of the linear function among :attr:`AVAILABLE_TYPES`.
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

        # Define the function as a linear combination with an offset
        def fun(
            x_vect,  # type: ndarray
        ):  # type: (...) -> ndarray
            value = matmul(self._coefficients, x_vect) + self._value_at_zero
            if value.size == 1:
                value = value[0]
            return value

        # Jacobian is constant
        def jac(_):
            """.. note::

            If the function is scalar, the gradient of the function is returned as a
            1d-array. If the function is vectorial, the Jacobian of the function is
            returned as a 2d-array.
            """
            if self._coefficients.shape[0] == 1:
                return self._coefficients[0, :]
            else:
                return self._coefficients

        super(MDOLinearFunction, self).__init__(
            fun,
            name,
            f_type=f_type,
            jac=jac,
            expr=expr,
            args=new_args,
            dim=output_dim,
            outvars=[name],
        )

    @property
    def coefficients(self):  # type: (...) -> ndarray
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
        coefficients,  # type: Union[Number,ndarray]
    ):  # type: (...) -> None
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
    def value_at_zero(self):  # type: (...) -> ndarray
        """The value of the function at zero.

        This is the vector :math:`b` in the expression :math:`y=Ax+b`.

        Raises:
            ValueError: If the value at zero is neither an ndarray nor a number.
        """
        return self._value_at_zero

    @value_at_zero.setter
    def value_at_zero(
        self,
        value_at_zero,  # type: Union[Number,ndarray]
    ):  # type: (...) -> None
        output_dim = self.coefficients.shape[0]  # N.B. the coefficients must be set
        if isinstance(value_at_zero, ndarray) and value_at_zero.size == output_dim:
            self._value_at_zero = value_at_zero.reshape(output_dim)
        elif isinstance(value_at_zero, Number):
            self._value_at_zero = array([value_at_zero] * output_dim)
        else:
            raise ValueError("Value at zero must be an ndarray or a number.")

    def _generate_1d_expr(
        self,
        args,  # type: Sequence[str]
    ):  # type: (...)-> str
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
        args,  # type: Sequence[str]
    ):  # type: (...)-> str
        """Generate the literal expression of the linear function in matrix form.

        Args:
            args: The names of the inputs of the function.

        Returns:
            The literal expression of the linear function in matrix form.
        """
        max_args_len = max([len(arg) for arg in args])
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
                expr += "[{}]".format(args[i])
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

    def __neg__(self):  # type: (...) -> MDOLinearFunction
        return MDOLinearFunction(
            -self._coefficients,
            "-" + self.name,
            self.f_type,
            self.args,
            -self._value_at_zero,
        )

    def offset(
        self,
        value,  # type: Union[Number,ndarray]
    ):  # type: (...) -> MDOLinearFunction
        return MDOLinearFunction(
            self._coefficients,
            self.name,
            self.f_type,
            self.args,
            self._value_at_zero + value,
        )

    def restrict(
        self,
        frozen_indexes,  # type: ndarray
        frozen_values,  # type: ndarray
    ):  # type: (...) -> MDOLinearFunction
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
        quad_coeffs,  # type: ndarray
        name,  # type: str
        f_type=None,  # type: Optional[str]
        args=None,  # type: Sequence[str]
        linear_coeffs=None,  # type: Optional[ndarray]
        value_at_zero=None,  # type: Optional[float]
    ):
        """
        Args:
            quad_coeffs: The second-order coefficients.
            name: The name of the function.
            f_type: The type of the linear function among :attr:`AVAILABLE_TYPES`.
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
            linear_part = MDOLinearFunction(
                linear_coeffs, "{}_lin".format(name), args=new_args
            )
            self.linear_coeffs = linear_part.coefficients
        self._value_at_zero = value_at_zero

        def func(
            x_vect,  # type: ndarray
        ):  # type: (...) -> ndarray
            """Compute the output of the quadratic function.

            Args:
                x_vect: The value of the inputs of the quadratic function.

            Returns:
                The value of the quadratic function.
            """
            value = multi_dot((x_vect.T, self._quad_coeffs, x_vect))
            if self._linear_coeffs is not None:
                value += linear_part(x_vect)
            if self._value_at_zero is not None:
                value += self._value_at_zero
            return value

        def grad(
            x_vect,  # type: ndarray
        ):  # type: (...) -> ndarray
            """Compute the gradient of the quadratic function.

            Args:
                x_vect: The value of the inputs of the quadratic function.

            Returns:
                The value of the gradient of the quadratic function.
            """
            gradient = matmul(self._quad_coeffs + self._quad_coeffs.T, x_vect)
            if self._linear_coeffs is not None:
                gradient += linear_part.jac(x_vect)
            return gradient

        # Build the second-order term
        expr = self.build_expression(
            self._quad_coeffs, new_args, self._linear_coeffs, self._value_at_zero
        )
        super(MDOQuadraticFunction, self).__init__(
            func, name, f_type, grad, expr, args=new_args, dim=1
        )

    @property
    def quad_coeffs(self):  # type: (...) -> ndarray
        """The second-order coefficients of the function.

        Raises:
            ValueError: If the coefficients are not passed
                as a 2-dimensional square ``ndarray``.
        """
        return self._quad_coeffs

    @quad_coeffs.setter
    def quad_coeffs(
        self,
        coefficients,  # type: ndarray
    ):  # type: (...) -> None
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
    def linear_coeffs(self):  # type: (...) -> ndarray
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
        coefficients,  # type: ndarray
    ):  # type: (...) -> None
        if coefficients.size != self._input_dim:
            raise ValueError(
                "The number of first-order coefficients must be equal "
                "to the input dimension."
            )
        self._linear_coeffs = coefficients

    @staticmethod
    def build_expression(
        quad_coeffs,  # type: ndarray
        args,  # type: Sequence[str]
        linear_coeffs=None,  # type: Optional[linear_coeffs]
        value_at_zero=None,  # type: Optional[float]
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
            expr += "[{}]".format(arg)
            expr += transpose_str if index == 0 else " "
            quad_coeffs_str = (MDOFunction.COEFF_FORMAT_ND.format(val) for val in line)
            expr += "[{}]".format(" ".join(quad_coeffs_str))
            expr += "[{}]".format(arg)
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
                expr += "[{}]".format(arg)
            # Zero-order expression
            if value_at_zero is not None and value_at_zero != 0.0 and index == 0:
                sign_str = "+" if value_at_zero > 0.0 else "-"
                expr += (" {} " + MDOFunction.COEFF_FORMAT_ND).format(
                    sign_str, abs(value_at_zero)
                )
            if index < quad_coeffs.shape[0] - 1:
                expr += "\n"
        return expr


class MDOFunctionGenerator(object):
    """Generator of :class:`MDOFunction` objects from a :class:`.MDODiscipline`.

    It creates a :class:`MDOFunction`
    evaluating some of the outputs of the discipline
    from some of its

    It uses closures to generate functions instances from a discipline execution.
    """

    def __init__(
        self,
        discipline,  # type: MDODiscipline
    ):  # type: (...) -> None
        """
        Args:
            discipline: The discipline from which the generator builds the functions.
        """
        self.discipline = discipline

    def __make_function(
        self,
        input_names_list,  # type: Sequence[str]
        output_names_list,  # type: Sequence[str]
        default_inputs,  # type: Optional[Mapping[str,ndarray]]
    ):  # type: (...) -> MDOFunction
        """Make a function from variables names and default input values.

        This method combines lambda functions
        and a local definition for function pointer.
        It uses a closure.

        Args:
            input_names_list: The names of the inputs of the function.
            output_names_list: The names of the outputs of the function.
            default_inputs: The default values of the inputs.
                If None,
                use the default values of the inputs
                specified by the discipline.

        Returns:
            The generated function.
        """

        def func(
            x_vect,  # type: ndarray
        ):  # type: (...) -> ndarray
            """A function which executes a discipline.

            Args:
                x_vect: The value of the inputs of the function.

            Returns:
                The value of the output of the function.

            Raises:
                ValueError: If an input variable is missing
                    from the default input values of the discipline.
            """
            for name in input_names_list:
                if name not in self.discipline.default_inputs:
                    raise ValueError(
                        "Discipline {} has no default_input named {}, "
                        "while input is required by MDOFunction.".format(
                            self.discipline.name, name
                        )
                    )
            defaults = self.discipline.default_inputs
            if default_inputs is not None:
                defaults.update(default_inputs)
            data = DataConversion.update_dict_from_array(
                defaults, input_names_list, x_vect
            )
            self.discipline.reset_statuses_for_run()
            computed_values = self.discipline.execute(data)
            values_array = DataConversion.dict_to_array(
                computed_values, output_names_list
            )
            if values_array.size == 1:  # Then the function is scalar
                return values_array[0]
            return values_array

        def func_jac(
            x_vect,  # type: ndarray
        ):  # type: (...) -> ndarray
            """A function which linearizes a discipline.

            Args:
                x_vect: The value of the inputs of the function.

            Returns:
                The value of the gradient of the function.
            """
            defaults = self.discipline.default_inputs
            n_dv = len(x_vect)
            data = DataConversion.update_dict_from_array(
                defaults, input_names_list, x_vect
            )
            self.discipline.linearize(data)

            grad_array = []
            for out_name in output_names_list:
                jac_loc = self.discipline.jac[out_name]
                grad_loc = DataConversion.dict_to_array(jac_loc, input_names_list)
                grad_output = hstack(grad_loc)
                if len(grad_output) > n_dv:
                    grad_output = reshape(grad_output, (grad_output.size // n_dv, n_dv))
                grad_array.append(grad_output)
            grad = vstack(grad_array).real
            if grad.shape[0] == 1:
                grad = grad.flatten()
                assert len(x_vect) == len(grad)
            return grad

        default_name = "_".join(output_names_list)
        mdo_func = MDOFunction(
            func,
            name=default_name,
            args=input_names_list,
            outvars=output_names_list,
            jac=func_jac,
        )
        return mdo_func

    def get_function(
        self,
        input_names_list,  # type: Sequence[str]
        output_names_list,  # type: Sequence[str]
        default_inputs=None,  # type: Optional[Mapping[str,ndarray]]
        differentiable=True,  # type: bool
    ):  # type: (...) -> MDOFunction
        """Build a function from a discipline input and output lists.

        Args:
            input_names_list: The names of the inputs of the discipline
                to be inputs of the function.
            output_names_list: The names of outputs of the discipline
                to be returned by the function.
            default_inputs: The default values of the inputs.
                If None,
                use the default values of the inputs
                specified by the discipline.
            differentiable: If True, then inputs and outputs are added
                to the variables to be differentiated.

        Returns:
            The function.

        Raises:
            ValueError: If a given input (or output) name is not the name
                of an input (or output) variable of the discipline.
        """
        if isinstance(input_names_list, string_types):
            input_names_list = [input_names_list]

        if isinstance(output_names_list, string_types):
            output_names_list = [output_names_list]

        if input_names_list is None:
            input_names_list = self.discipline.get_input_data_names()
        if output_names_list is None:
            output_names_list = self.discipline.get_output_data_names()

        if not self.discipline.is_all_inputs_existing(input_names_list):
            raise ValueError(
                "Some elements of {} are not inputs of the discipline {}; "
                "available inputs are: {}.".format(
                    input_names_list,
                    self.discipline.name,
                    self.discipline.get_input_data_names(),
                )
            )

        if not self.discipline.is_all_outputs_existing(output_names_list):
            raise ValueError(
                "Some elements of {} are not outputs of the discipline {}; "
                "available outputs are: {}.".format(
                    output_names_list,
                    self.discipline.name,
                    ", ".join(self.discipline.get_output_data_names()),
                )
            )

        # adds inputs and outputs to the list of variables to be
        # differentiated
        if differentiable:
            self.discipline.add_differentiated_inputs(input_names_list)
            self.discipline.add_differentiated_outputs(output_names_list)

        return self.__make_function(input_names_list, output_names_list, default_inputs)


class NotImplementedCallable(object):
    """A not implemented callable object."""

    def __call__(self, *args, **kwargs):  # type: (...) -> NoReturn
        """
        Raises:
            NotImplementedError: At each evaluation.
        """
        raise NotImplementedError("Function is not implemented.")
