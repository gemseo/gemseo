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
"""
Functions, f(x), from disciplines execution
*******************************************
"""
from __future__ import division, unicode_literals

import logging
from multiprocessing import Value
from numbers import Number

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

from gemseo.utils.data_conversion import DataConversion
from gemseo.utils.derivatives_approx import ComplexStep, FirstOrderFD

LOGGER = logging.getLogger(__name__)


class MDOFunction(object):
    """A container object to represent a function in an MDO process."""

    TYPE_OBJ = "obj"
    TYPE_EQ = "eq"
    TYPE_INEQ = "ineq"
    TYPE_OBS = "obs"
    AVAILABLE_TYPES = [TYPE_OBJ, TYPE_EQ, TYPE_INEQ, TYPE_OBS]
    DICT_REPR_ATTR = ["name", "f_type", "expr", "args", "dim", "special_repr"]
    DEFAULT_ARGS_BASE = "x"
    INDEX_PREFIX = "!"
    COEFF_FORMAT_1D = "{:.2e}"  # ensure that coefficients strings have same length
    COEFF_FORMAT_ND = "{: .2e}"  # ensure that coefficients strings have same length
    # N.B. the space character ensures same length whatever the sign of the coefficient

    def __init__(
        self,
        func,
        name,
        f_type=None,
        jac=None,
        expr=None,
        args=None,
        dim=None,
        outvars=None,
        force_real=False,
        special_repr=None,
    ):
        """Initializes the function attributes.

        :param func: the pointer to the function to be actually called
        :type func: callable
        :param name: the name of the function as a string
        :type name: str
        :param f_type: the type of function among (obj, eq, ineq, obs)
        :type f_type: str, optional
        :param jac: the jacobian
        :type jac: callable, optional
        :param expr: the function expression
        :type expr: str, optional
        :param args: the function arguments
        :type args: list(str), optional
        :param dim: the output dimension
        :type dim: int, optional
        :param outvars: the array of variable names used as inputs
            of the function
        :type outvars: list(str), optional
        :param force_real: if True, cast the results to real value
        :type force_real: bool, optional
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
    def n_calls(self):
        """Returns the number of calls to execute() which triggered the _run()
        multiprocessing safe."""
        return self._n_calls.value

    @n_calls.setter
    def n_calls(self, value):
        """Sets the number of calls to execute() which triggered the _run()
        multiprocessing safe."""
        with self._n_calls.get_lock():
            self._n_calls.value = value

    @property
    def func(self):
        """Accessor to the func property."""
        return self.__counted_f

    def __counted_f(self, x_vect):
        """Calls the function in self[self.SCIPY_FUN_TAG]

        :param x_vect: the function argument
        """
        with self._n_calls.get_lock():
            self._n_calls.value += 1
        val = self._func(x_vect)
        self.last_eval = val
        return val

    @func.setter
    def func(self, f_pointer):
        """Sets the f pointer.

        :param f_pointer: the pointer to the function to be actually called
        """
        self._n_calls.value = 0
        self._func = f_pointer

    def __call__(self, x_vect):
        """Calls the function.

        :param x_vect: the function argument
        """
        val = self.evaluate(x_vect, self.force_real)
        return val

    def evaluate(self, x_vect, force_real=False):
        """Evaluate the function.

        :param x_vect: the function argument
        :type x_vect: ndarray
        :param force_real: if True, cast the results to real value
        :type force_real: bool, optional
        :return: the function value
        :rtype: Number or ndarray
        """
        val = self.__counted_f(x_vect)
        if force_real:
            val = val.real
        self.dim = atleast_1d(val).shape[0]
        return val

    @property
    def name(self):
        """Accessor to the name of the function."""
        return self._name

    @name.setter
    def name(self, name):
        """Sets the name of the function.

        :param name: the name of the functioon
        """
        if name is not None and not isinstance(name, string_types):
            raise TypeError(
                "MDOFunction name must be a string, got " + str(type(name)) + " instead"
            )
        self._name = name

    @property
    def f_type(self):
        """Accessor to the function type."""
        return self._f_type

    @f_type.setter
    def f_type(self, f_type):
        """Sets the function type.

        :param f_type: the type of function among MDOFunction.AVAILABLE_TYPES
        """
        if f_type is not None and f_type not in self.AVAILABLE_TYPES:
            raise ValueError(
                "MDOFunction type must be among "
                + str(self.AVAILABLE_TYPES)
                + " , got "
                + str(f_type)
                + " instead"
            )
        self._f_type = f_type

    @property
    def jac(self):
        """Accessor to the jacobian."""
        return self._jac

    @jac.setter
    def jac(self, jac):
        """Sets the function jacobian.

        :param jac: pointer to a jacobian function
        """
        if jac is not None:
            if not callable(jac):
                raise TypeError("Jacobian function must be callable")
        self._jac = jac

    @property
    def args(self):
        """Accessor to the arguments list."""
        return self._args

    @args.setter
    def args(self, args):
        """Sets the function arguments.

        :param args: arguments list
        """
        if args is not None:
            if isinstance(args, ndarray):
                self._args = args.tolist()
            else:
                self._args = list(args)
        else:
            self._args = None

    @property
    def expr(self):
        """Accessor to the expression."""
        return self._expr

    @expr.setter
    def expr(self, expr):
        """Sets the function expression.

        :param expr: arguments list
        """
        if expr is not None and not isinstance(expr, string_types):
            raise TypeError(
                "Expression must be a string. "
                + " Got "
                + str(type(expr))
                + " instead."
            )
        self._expr = expr

    @property
    def dim(self):
        """Accessor to the dimension."""
        return self._dim

    @dim.setter
    def dim(self, dim):
        """Sets the function dimension.

        :param dim: int
        """
        if dim is not None and not int(dim) == dim:
            raise TypeError(
                "Dimension must be a integer. " + " Got " + str(type(int)) + " instead."
            )
        self._dim = dim

    @property
    def outvars(self):
        """Accessor to the array of input variable names used by the function."""
        return self._outvars

    @outvars.setter
    def outvars(self, outvars):
        """Sets the array of output variable names used by the function.

        :param outvars: array of variable names
        """
        if outvars is not None:
            if isinstance(outvars, ndarray):
                self._outvars = outvars.tolist()
            else:
                self._outvars = list(outvars)
        else:
            self._outvars = None

    def is_constraint(self):
        """Returns True if self.f_type is eq or ineq.

        :return: True if and only if the function is a contraint
        :rtype: bool
        """
        return self.f_type in [self.TYPE_EQ, self.TYPE_INEQ]

    def __repr__(self):
        """Self representation as a string."""
        return self.special_repr or self.default_repr

    @property
    def default_repr(self):
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

    def has_jac(self):
        """Check if MDOFunction has a jacobian.

        :returns: True if self has a jacobian
        :rtype: bool
        """
        return self.jac is not None and not isinstance(
            self._jac, NotImplementedCallable
        )

    def has_dim(self):
        """Check if MDOFunction has a dimension.

        :returns: True if self has a dimension
        :rtype: bool
        """
        return self.dim is not None

    def has_outvars(self):
        """Check if MDOFunction has an array of input variables.

        :returns: True if self has a dimension
        :rtype: bool
        """
        return self.outvars is not None

    def has_expr(self):
        """Check if MDOFunction has an expression.

        :returns: True if self has an expression
        :rtype: bool
        """
        return self.expr is not None

    def has_args(self):
        """Check if MDOFunction has an args.

        :returns: True if self has an args
        :rtype: bool
        """
        return self.args is not None

    def has_f_type(self):
        """Check if MDOFunction has an type.

        :returns: True if self has a type
        :rtype: bool
        """
        return self.f_type is not None

    def apply_operator(self, other_f, operator, operator_repr):
        """Defines addition/substraction  for MDOFunction Supports automatic
        differentiation if other_f and self have a Jacobian.

        :param other_f: param operator:
        :param operator_repr: the representation as a string
        :param operator: the operator as a function pointer
        """
        if isinstance(other_f, MDOFunction):
            return self.__apply_operator_tofunction(other_f, operator, operator_repr)

        if isinstance(other_f, Number):
            return self.__apply_operator_to_number(other_f, operator, operator_repr)

        raise TypeError(
            "Unuspported + operand for MDOFunction and " + str(type(other_f))
        )

    def __apply_operator_tofunction(self, other_f, operator, operator_repr):
        """Defines addition/substraction  for MDOFunction Supports automatic
        differentiation if other_f and self have a Jacobian.

        :param other_f: param operator:
        :param operator_repr: the representation as a string
        :param operator: the operator as a function pointer
        """
        assert isinstance(other_f, MDOFunction)

        def add_f_pt(x_vect):
            """Addition function.

            :param x_vect: design variables
            :returns: self(x_vect) added to other_f(x_vect)
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

            def add_jac(x_vect):
                """Jacobian of addition function.

                :param x_vect: design variables
                :returns: Jac self(x_vect) added to Jac other_f(x_vect)
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

    def __apply_operator_to_number(self, other_val, operator, operator_repr):
        """Defines addition/substraction  for MDOFunction Supports automatic
        differentiation if other_f and self have a Jacobian.

        :param other_val: param operator: the value to add/subs
        :param operator_repr: the representation as a string
        :param operator: the operator as a function pointer
        """
        assert isinstance(other_val, Number)

        def add_f_pt(x_vect):
            """Addition function.

            :param x_vect: design variables
            :returns: self(x_vect) added to other_f(x_vect)
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

    def __add__(self, other_f):
        """Defines addition for MDOFunction Supports automatic differentiation if
        other_f and self have a jacobian."""
        return self.apply_operator(other_f, operator=add, operator_repr="+")

    def __sub__(self, other_f):
        """Defines subtraction for MDOFunction Supports automatic differentiation if
        other_f and self have a jacobian."""
        return self.apply_operator(other_f, operator=subtract, operator_repr="-")

    def __neg__(self):
        """Defines minus operator for MDOFunction (-f) Supports automatic
        differentiation if other_f and self have a Jacobian."""

        def min_pt(x_vect):
            """Negative self function.

            :param x_vect: design variables
            :returns: self(x_vect)
            """
            return -self(x_vect)

        min_name = "-" + self.name
        min_self = MDOFunction(
            min_pt,
            min_name,
            args=self.args,
            f_type=self.f_type,
            dim=self.dim,
            outvars=self.outvars,
        )

        if self.has_jac():

            def min_jac(x_vect):
                """Jacobian of negative self.

                :param x_vect: design variables
                :returns: Jac of -self(x_vect)
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

    def __mul__(self, other):
        """Defines multiplication for MDOFunction Supports automatic linearization if
        other_f and self have a jacobian."""
        is_number = isinstance(other, Number)
        is_func = isinstance(other, MDOFunction)
        if not is_number and not is_func:
            raise TypeError(
                "Unuspported * operand for MDOFunction and " + str(type(other))
            )

        if is_func:

            def mul_f_pt(x_vect):
                """Multiplication function.

                :param x_vect: design variables
                :returns: self(x_vect) added to other_f(x_vect)
                """
                return self(x_vect) * other(x_vect)

            out_name = self.name + "*" + other.name

        else:  # Func is a number

            def mul_f_pt(x_vect):
                """Multiplication function.

                :param x_vect: design variables
                :returns: self(x_vect) added to other_f(x_vect)
                """
                return self(x_vect) * other

            out_name = self.name + "*" + str(other)

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

                def mult_jac(x_vect):
                    """Jacobian of multiplication function.

                    :param x_vect: design variables
                    :returns: Jac self(x_vect) added to Jac other_f(x_vect)
                    """
                    self_f = self(x_vect)
                    other_f = other(x_vect)
                    self_jac = self._jac(x_vect)
                    other_jac = other.jac(x_vect)
                    return self_jac * other_f + other_jac * self_f

                self_oper_f.jac = mult_jac
        elif self.has_jac():

            def mult_jac(x_vect):
                """Jacobian of multiplication function.

                :param x_vect: design variables
                :returns: Jac self(x_vect) added to Jac other_f(x_vect)
                """
                return self._jac(x_vect) * other

            self_oper_f.jac = mult_jac

        if self.has_expr():
            if is_func and other.has_expr():
                f_expr = "(" + self.expr + ")*(" + other.expr + ")"
                self_oper_f.expr = f_expr
            if is_number:
                f_expr = "(" + self.expr + ")*" + str(other)
                self_oper_f.expr = f_expr

        if is_func and self.has_args() and other.has_args():
            args = sorted(list(set(self.args + other.args)))
            self_oper_f.args = args

        if self.has_f_type():
            self_oper_f.f_type = self.f_type
        elif is_func and other.has_f_type():
            self_oper_f.f_type = other.f_type

        return self_oper_f

    def offset(self, value):
        """Adds an offset value to the function.

        :param value: the offset value
        :type value: Number or ndarray
        :returns: the offset function as an MDOFunction
        :rtype: MDOFunction
        """

        def wrapped_function(x_vect):
            """Wrapped provided function in order to give to optimizer.

            :param x_vect: design variables
            :returns: evaluation of function at design variables
                     plus the offset
            """
            return self(x_vect) + value

        expr = self.expr
        if expr is not None:
            if value < 0:
                expr += " - " + str(-value)
            else:
                expr += " + " + str(abs(value))
        else:
            offset_is_negative = value < 0
            if hasattr(offset_is_negative, "__len__"):
                offset_is_negative = offset_is_negative.all()
            if offset_is_negative:
                expr = self.name + " - " + str(-value)
            else:
                expr = self.name + " + " + str(abs(value))
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
        frozen_indexes,
        frozen_values,
        input_dim,
        name=None,
        f_type=None,
        expr=None,
        args=None,
    ):
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

        :param frozen_indexes: indexes of the variables that will be frozen
        :type frozen_indexes: ndarray
        :param frozen_values: values of the variables that will be frozen
        :type frozen_values: ndarray
        :param input_dim: dimension of the function input (before restriction)
        :type input_dim: int
        :param name: name of the restriction
        :type name: str, optional
        :param f_type: type of the restriction
        :type f_type: str, optional
        :param expr: the expression of the restriction
        :type expr: str, optional
        :param args: arguments names of the restriction
        :type args: list(str), optional
        :return: restriction of the function
        :rtype: MDOFunction
        """
        # Check the shapes of the passed arrays
        if frozen_indexes.shape != frozen_values.shape:
            raise ValueError("Arrays of frozen indexes and values must have same shape")

        active_indexes = array([i for i in range(input_dim) if i not in frozen_indexes])

        def extend_subvect(x_subvect):
            """Return the extension of a restriction input with the frozen values.

            :param x_subvect: input of the restriction
            :type x_subvect: ndarray
            :return: extended input
            :rtype: ndarray
            """
            x_vect = empty(input_dim)
            x_vect[active_indexes] = x_subvect
            x_vect[frozen_indexes] = frozen_values
            return x_vect

        def func(x_subvect):
            """Return the value of the restriction.

            :param x_subvect: input of the restriction
            :type x_subvect: ndarray
            :return: value of the restriction
            :rtype: float
            """
            x_vect = extend_subvect(x_subvect)
            value = self.__call__(x_vect)
            return value

        # Build the name of the restriction
        if name is None and args is not None:
            name = self.name + "_wrt_" + "_".join(args)
        elif name is None:
            name = self.name + "_restriction"

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

            def jac(x_subvect):
                """Return the Jacobian matrix of the restriction.

                :param x_subvect: input of the restriction
                :type x_subvect: ndarray
                :return: Jacobian matrix of the restriction
                :rtype: ndarray
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

    def linear_approximation(self, x_vect, name=None, f_type=None, args=None):
        r"""Return the first-order Taylor polynomial of the function at a given point

        :math:`\newcommand{\xref}{\hat{x}}\newcommand{\dim}{d}`
        The first-order Taylor polynomial of a (possibly vector-valued) function
        :math:`f` at a point :math:`\xref` is defined as

        .. math::
            \newcommand{\partialder}{\frac{\partial f}{\partial x_i}(\xref)}
            f(x)
            \approx
            f(\xref) + \sum_{i = 1}^{\dim} \partialder \, (x_i - \xref_i).

        :param x_vect: point defining the Taylor polynomial
        :type x_vect: ndarray
        :param name: name of the linear approximation
        :type name: str, optional
        :param f_type: the function type of the linear approximation
        :type f_type: str, optional
        :param args: names for each scalar variable, or a name base
        :type args: list(str), optional
        :returns: first-order Taylor polynomial of the function at the given point
        :rtype: MDOLinearFunction
        """
        # Check that the function Jacobian is available
        if not self.has_jac():
            raise AttributeError(
                "Function Jacobian unavailable for linear approximation"
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

    def convex_linear_approx(self, x_vect, approx_indexes=None, sign_threshold=1e-9):
        r"""Return the convex linearization of the function at a given point

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
        respect to a subset
        :math:`x_{i \in \approxinds} \subset \{x_1, \dots, x_{\dim}\}`
        of its variables rather than all of them:

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

        :param x_vect: point defining the convex linearization
        :type x_vect: ndarray
        :param approx_indexes: array of booleans specifying w.r.t. which variables
            the function should be approximated (by default, all of them)
        :type approx_indexes: ndarray, optional
        :param sign_threshold: threshold for the sign of the derivatives
        :type sign_threshold: float, optional
        :returns: convex linearization of the function at the given point
        :rtype: MDOFunction
        """
        # Check the approximation indexes
        if approx_indexes is None:
            approx_indexes = ones_like(x_vect, dtype=bool)
        elif approx_indexes.shape != x_vect.shape or approx_indexes.dtype != "bool":
            raise ValueError(
                "The approximation array must an array of booleans with "
                "the same shape as the function argument"
            )

        # Get the function Jacobian matrix
        if not self.has_jac():
            raise AttributeError(
                "Function Jacobian unavailable for convex linearization"
            )
        jac = atleast_2d(self.jac(x_vect))

        # Build the coefficients matrices
        coeffs = jac[:, approx_indexes]
        direct_coeffs = where(coeffs > sign_threshold, coeffs, 0.0)
        recipr_coeffs = multiply(
            -where(-coeffs > sign_threshold, coeffs, 0.0), x_vect[approx_indexes] ** 2
        )

        def get_steps(x_new):
            """Return the steps on the direct and reciprocal variables.

            :param x_new: argument of the convex linearization
            :type x_new: ndarray
            :returns: step on the direct variables, step on the reciprocal variables
            :rtype: ndarray, ndarray
            """
            step = x_new[approx_indexes] - x_vect[approx_indexes]
            inv_step = zeros_like(step)
            nonzero_indexes = (absolute(step) > sign_threshold).nonzero()
            inv_step[nonzero_indexes] = 1.0 / step[nonzero_indexes]
            return step, inv_step

        def convex_lin_func(x_new):
            """Return the value of the function convex linearization.

            :param x_new: argument of the convex linearization
            :type x_new: ndarray
            :returns: value of the function convex linearization
            :rtype: ndarray
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

        def convex_lin_jac(x_new):
            """Return the Jacobian of the function convex linearization.

            :param x_new: argument of the convex linearization
            :type x_new: ndarray
            :returns: Jacobian of the function convex linearization
            :rtype: ndarray
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
            self.name + "_convex_lin",
            self.f_type,
            convex_lin_jac,
            args=None,
            dim=self.dim,
            outvars=self.outvars,
            force_real=self.force_real,
        )

        return convex_lin_approx

    def quadratic_approx(self, x_vect, hessian_approx, args=None):
        r"""Return a quadratic appproximation of the (scalar-valued) function at a
        given point

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

        :param x_vect: point defining the Taylor polynomial
        :type x_vect: ndarray
        :param hessian_approx: approximation of the Hessian matrix at x_vect
        :type hessian_approx: ndarray
        :param args: names for each scalar variable, or a name base
        :type args: list(str), optional
        :returns: second-order Taylor polynomial of the function at the given point
        :rtype: MDOQuadraticFunction
        """
        # Build the second-order coefficients
        if (
            not isinstance(hessian_approx, ndarray)
            or hessian_approx.ndim != 2
            or hessian_approx.shape[0] != hessian_approx.shape[1]
        ):
            raise ValueError("Hessian approximation must be a square ndarray")
        if hessian_approx.shape[1] != x_vect.size:
            raise ValueError(
                "Hessian approximation and vector must have same dimension"
            )
        quad_coeffs = 0.5 * hessian_approx

        # Build the first-order coefficients
        if not self.has_jac():
            raise AttributeError("Jacobian unavailable")
        gradient = self.jac(x_vect)
        hess_dot_vect = matmul(hessian_approx, x_vect)
        linear_coeffs = gradient - hess_dot_vect

        # Buid the zero-order coefficient
        zero_coeff = matmul(0.5 * hess_dot_vect - gradient, x_vect) + self.__call__(
            x_vect
        )

        # Build the second-order Taylor polynomial
        quadratic_suffix = "_quadratized"
        name = self.name + quadratic_suffix
        args = args if args else self.args
        quad_approx = MDOQuadraticFunction(
            quad_coeffs=quad_coeffs,
            linear_coeffs=linear_coeffs,
            value_at_zero=zero_coeff,
            name=name,
            args=args,
        )

        return quad_approx

    def check_grad(self, x_vect, method="FirstOrderFD", step=1e-6, error_max=1e-8):
        """Checks the gradient of self.

        :param x_vect: the vector at which the function is checked
        :type x_vect: ndarray
        :param method: FirstOrderFD or ComplexStep
            (Default value = "FirstOrderFD")
        :type method: str, optional
        :param step: the step for approximation
            (Default value = 1e-6)
        :type step: float, optional
        :param error_max: Default value = 1e-8)
        :type error_max: float, optional
        """

        def rel_err(a_vect, b_vect):
            """Compute relative error.

            :param a_vect: param b_vect:
            :param b_vect:
            """
            n_b = norm(b_vect)
            if n_b > error_max:
                return norm(a_vect - b_vect) / norm(b_vect)
            return norm(a_vect - b_vect)

        def filt_0(arr, floor_value=1e-6):
            """Filtering of error.

            :param arr:
            :param floor_value:  (Default value = 1e-6)
            :param floor_value:  (Default value = 1e-6)
            """
            return where(np_abs(arr) < floor_value, 0.0, arr)

        if method == "FirstOrderFD":
            apprx = FirstOrderFD(self, step)
        elif method == "ComplexStep":
            apprx = ComplexStep(self, step)
        else:
            raise ValueError(
                "Unknwon approximation method "
                + str(method)
                + ", use FirstOrderFD or ComplexStep."
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
                    "Inconsistent function jacobian shape ! "
                    + "Got :"
                    + str(anal_grad.shape)
                    + " while expected :"
                    + str(apprx_grad.shape)
                )
        rel_error = rel_err(anal_grad, apprx_grad)
        succeed = rel_error < error_max
        if not succeed:
            LOGGER.error("Function jacobian is wrong %s", str(self))
            LOGGER.error("Error =\n%s", str(filt_0(anal_grad - apprx_grad)))
            LOGGER.error("Analytic jacobian=\n%s", str(filt_0(anal_grad)))
            LOGGER.error("Approximate step gradient=\n%s", str(filt_0(apprx_grad)))
            raise ValueError("Function jacobian is wrong " + str(self))

    def get_data_dict_repr(self):
        """Returns a dict representation of self for serialization Pointers to functions
        are removed.

        :returns: a dict with attributes names as keys
        :rtype: dict
        """
        repr_dict = {}
        for attr_name in self.DICT_REPR_ATTR:
            attr = getattr(self, attr_name)
            if attr is not None:
                repr_dict[attr_name] = attr
        return repr_dict

    @staticmethod
    def init_from_dict_repr(**kwargs):
        """Initalizes a new Function from a data dict typically used for
        deserialization.

        :param kwargs: key value pairs from DICT_REPR_ATTR
        """
        allowed = MDOFunction.DICT_REPR_ATTR
        for key in kwargs:
            if key not in allowed:
                raise ValueError(
                    "Cannot initialize MDOFunction "
                    + "attribute: "
                    + str(key)
                    + ", allowed ones are: "
                    + ", ".join(allowed)
                )
        return MDOFunction(func=None, **kwargs)

    def set_pt_from_database(
        self, database, design_space, normalize=False, jac=True, x_tolerance=1e-10
    ):
        """self.__call__(x) returns f(x) if x is in the database and self.name in the
        database keys Idem for jac.

        :param database: the database to read
        :type database: Database
        :param design_space: the design space used for normalization
        :type design_space: DesignSpace
        :param normalize: if True, x_n is unnormalized before call
        :type normalize: bool, optional
        :param jac: if True, a jacobian pointer is also generated
        :type jac: bool, optional
        :param x_tolerance: tolerance on the distance between inputs
        :type x_tolerance: float, optional
        """
        name = self.name

        def read_in_db(x_n, fname):
            """Reads fname in the database for the x_n entry.

            :param x_n: the x to evaluate fname
            :returns : f(x_n) if present in the database
            """
            if normalize:
                x_db = design_space.unnormalize_vect(x_n)
            else:
                x_db = x_n
            val = database.get_f_of_x(fname, x_db, x_tolerance)
            if val is None:
                msg = "Function " + str(fname)
                msg += " evaluation relies only on the database, and"
                msg += str(fname) + "( x ) is not in the database "
                msg += " for x=" + str(x_db)
                raise ValueError(msg)
            return val

        def f_from_db(x_n):
            """Computes f(x_n) from the database.

            :param x_n: the x to evaluate f
            """
            return read_in_db(x_n, name)

        def j_from_db(x_n):
            """Computes df(x_n)/d x_n from the database.

            :param x_n: the x to evaluate the jacobian
            """
            return read_in_db(x_n, "@" + name)

        self.func = f_from_db
        if jac:
            self.jac = j_from_db

    @staticmethod
    def generate_args(input_dim, args=None):
        """Generate the strings for a function arguments.

        :param input_dim: number of scalar input arguments
        :type input_dim: int
        :param args: the initial function arguments strings
        :type args: list(str), optional
        :returns: the arguments strings
        :rtype: list(str)
        """
        if args and len(args) == input_dim:
            # Keep the passed list of strings
            new_args = args
        elif args and len(args) == 1:
            # Generate the arguments strings based on a the unique passed string
            new_args = MDOFunction._generate_args(args[0], input_dim)
        else:
            # Generate the arguments strings based on the default string
            args_base = "x"
            new_args = MDOFunction._generate_args(args_base, input_dim)

        return new_args

    @staticmethod
    def _generate_args(args_base, input_dim):
        """Generate the arguments strings based on a string.

        :param args_base: base string for the arguments strings
        :type args_base: str
        :param input_dim: number of scalar input arguments
        :type input_dim: int
        :returns: arguments strings
        :rtype: list(str)
        """
        index_prefix = MDOLinearFunction.INDEX_PREFIX
        return [args_base + index_prefix + str(i) for i in range(input_dim)]

    @staticmethod
    def concatenate(
        functions,
        name,
        f_type=None,
    ):
        """Concatenate functions.

        :param functions: functions to be concatenated
        :type functions: list(MDOFunction)
        :param name: name of the concatenation
        :type name: str
        :param f_type: type of the concatenation function
        :type f_type: str, optional
        :return: concatenation of the functions
        :rtype: MDOFunction
        """

        def concat_func(x_vect):
            """Concatenate functions values.

            :param x_vect: common input of the functions
            :type x_vect: ndarray
            :return: concatenation of the functions values
            :rtype: ndarray
            """
            return concatenate([atleast_1d(func(x_vect)) for func in functions])

        def concat_jac(x_vect):
            """Concatenate functions Jacobian matrices.

            :param x_vect: common input of the functions
            :type x_vect: ndarray
            :return: concatenation of the functions Jacobian matrices
            :rtype: ndarray
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

    def __init__(self, coefficients, name, f_type=None, args=None, value_at_zero=0.0):
        """Initialize the function attributes.

        :param coefficients: coefficients of the linear function
        :type coefficients: ndarray or Number
        :param name: the name of the function
        :type name: str
        :param f_type: the type of function among (obj, eq, ineq)
        :type f_type: str, optional
        :param args: the function arguments
        :type args: list(str), optional
        :param value_at_zero: function value at zero
        :type value_at_zero: ndarray or Number, optional
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
        def fun(x_vect):
            """Return the linear combination with an offset:

            sum_{i=1}^n a_i * x_i + b

            :param x_vect: design variables
            :type x_vect: ndarray
            """
            value = matmul(self._coefficients, x_vect) + self._value_at_zero
            if value.size == 1:
                value = value[0]
            return value

        # Jacobian is constant
        def jac(_):
            """Return the constant Jacobian, which is the matrix of coefficients. N.B.
            if the function is scalar a 1d-array is returned (the gradient).

            :returns: function Jacobian or function gradient
            :rtype: ndarray
            """
            if self._coefficients.shape[0] == 1:
                jac = self._coefficients[0, :]
            else:
                jac = self._coefficients
            return jac

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
    def coefficients(self):
        """Get the function coefficients. N.B. shall return a 2-dimensional ndarray.

        :returns: function coefficients
        :rtype: ndarray
        """
        return self._coefficients

    @coefficients.setter
    def coefficients(self, coefficients):
        """Set the function coefficients. N.B. the coefficients shall be stored as a
        2-dimensional ndarray.

        :param coefficients: function coefficients
        :type coefficients: ndarray or Number
        """
        if isinstance(coefficients, Number):
            self._coefficients = atleast_2d(coefficients)
        elif isinstance(coefficients, ndarray) and len(coefficients.shape) == 2:
            self._coefficients = coefficients
        elif isinstance(coefficients, ndarray) and len(coefficients.shape) == 1:
            self._coefficients = coefficients.reshape([1, -1])
        else:
            raise ValueError(
                "Coefficients must be passed as a 2-dimensional or "
                "1-dimensional ndarray"
            )

    @property
    def value_at_zero(self):
        """Get the function value at zero. N.B. shall return a 1-dimensional ndarray.

        :returns: function value at zero
        :rtype: ndarray
        """
        return self._value_at_zero

    @value_at_zero.setter
    def value_at_zero(self, value_at_zero):
        """Set the function value at zero N.B. the value at zero shall be stored as a
        1-dimensional ndarray.

        :param value_at_zero: function value at zero
        :type value_at_zero: ndarray or Number
        """
        output_dim = self.coefficients.shape[0]  # N.B. the coefficients must be set
        if isinstance(value_at_zero, ndarray) and value_at_zero.size == output_dim:
            self._value_at_zero = value_at_zero.reshape(output_dim)
        elif isinstance(value_at_zero, Number):
            self._value_at_zero = array([value_at_zero] * output_dim)
        else:
            raise ValueError("Value at zero must be an ndarray or a number")

    def _generate_1d_expr(self, args):
        """Generate the literal expression of the linear function in scalar form.

        :param args: the function arguments
        :type args: list(str)
        :returns: function literal expression
        :rtype: str
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

    def _generate_nd_expr(self, args):
        """Generate the literal expression of the linear function in matrix form.

        :param args: the function arguments
        :type args: list(str)
        :returns: function literal expression
        :rtype: str
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

    def __neg__(self):
        """Define the minus operator for an MDOLinearFunction (-f).

        :returns: the opposite function
        :rtype: MDOLinearFunction
        """
        return MDOLinearFunction(
            -self._coefficients,
            "-" + self.name,
            self.f_type,
            self.args,
            -self._value_at_zero,
        )

    def offset(self, value):
        """Add an offset value to the linear function.

        :param value: the offset value
        :type value: Number or ndarray
        :returns: the offset function
        :rtype: MDOLinearFunction
        """
        return MDOLinearFunction(
            self._coefficients,
            self.name,
            self.f_type,
            self.args,
            self._value_at_zero + value,
        )

    def restrict(self, frozen_indexes, frozen_values):
        """Build a restriction of the linear function.

        :param frozen_indexes: indexes of the variables that will be frozen
        :type frozen_indexes: ndarray
        :param frozen_values: values of the variables that will be frozen
        :type frozen_values: ndarray
        :returns: the restriction of the linear function
        :rtype: MDOLinearFunction
        """
        # Check the shapes of the passed arrays
        if frozen_indexes.shape != frozen_values.shape:
            raise ValueError("Arrays of frozen indexes and values must have same shape")

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
        quad_coeffs,
        name,
        f_type=None,
        args=None,
        linear_coeffs=None,
        value_at_zero=None,
    ):
        """Initialize a quadratic function.

        :param quad_coeffs: second-order coefficients
        :type quad_coeffs: ndarray
        :param name: name of the function
        :type name: str
        :param f_type: type of the function
        :type f_type: str, optional
        :param args: arguments names of the function
        :type args: list(str), optional
        :param linear_coeffs: first-order coefficients
        :type linear_coeffs: ndarray, optional
        :param value_at_zero: zero-order coefficient
        :type value_at_zero: float, optional
        """
        self._input_dim = 0
        self._quad_coeffs = None
        self._linear_coeffs = None
        self.quad_coeffs = quad_coeffs  # sets the input dimension
        new_args = MDOFunction.generate_args(self._input_dim, args)

        # Build the first-order term
        if linear_coeffs is not None:
            linear_part = MDOLinearFunction(linear_coeffs, name + "_lin", args=new_args)
            self.linear_coeffs = linear_part.coefficients
        self._value_at_zero = value_at_zero

        def func(x_vect):
            """Compute the quadratic function value.

            :param x_vect: quadratic function argument
            :type x_vect: ndarray
            :returns: the quadratic function value
            :rtype: float
            """
            value = multi_dot((x_vect.T, self._quad_coeffs, x_vect))
            if self._linear_coeffs is not None:
                value += linear_part(x_vect)
            if self._value_at_zero is not None:
                value += self._value_at_zero
            return value

        def grad(x_vect):
            """Compute the quadratic function gradient.

            :param x_vect: quadratic function argument
            :type x_vect: ndarray
            :returns: the quadratic function gradient
            :rtype: ndarray
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
    def quad_coeffs(self):
        """Get the function second-order coefficients N.B. shall return a 2-dimensional
        ndarray.

        :returns: function second-order coefficients
        :rtype: ndarray
        """
        return self._quad_coeffs

    @quad_coeffs.setter
    def quad_coeffs(self, coefficients):
        """Set the function second-order coefficients N.B. the coefficients shall be
        stored as a 2-dimensional ndarray.

        :param coefficients: function second-order coefficients
        :type coefficients: ndarray
        """
        # Check the second-order coefficients
        if (
            not isinstance(coefficients, ndarray)
            or len(coefficients.shape) != 2
            or coefficients.shape[0] != coefficients.shape[1]
        ):
            raise ValueError(
                "Quadratic coefficients must be passed as a 2-dimensional "
                "square ndarray"
            )
        self._quad_coeffs = coefficients
        self._input_dim = self._quad_coeffs.shape[0]

    @property
    def linear_coeffs(self):
        """Get the function first-order coefficients N.B. shall return a 1-dimensional
        ndarray.

        :returns: function first-order coefficients
        :rtype: ndarray
        """
        if self._linear_coeffs is None:
            return zeros(self._input_dim)
        return self._linear_coeffs

    @linear_coeffs.setter
    def linear_coeffs(self, coefficients):
        """Set the function first-order coefficients N.B. the coefficients shall be
        stored as a 1-dimensional ndarray.

        :param coefficients: function first-order coefficients
        :type coefficients: ndarray
        """
        if coefficients.size != self._input_dim:
            raise ValueError(
                "The number of first-order coefficients must be equal "
                "to the input dimension"
            )
        self._linear_coeffs = coefficients

    @staticmethod
    def build_expression(quad_coeffs, args, linear_coeffs=None, value_at_zero=None):
        """Return the expression of the quadratic function.

        :param quad_coeffs: second-order coefficients
        :type quad_coeffs: ndarray
        :param args: arguments names of the function
        :type args: list(str)
        :param linear_coeffs: first-order coefficients
        :type linear_coeffs: ndarray, optional
        :param value_at_zero: zero-order coefficient
        :type value_at_zero: float, optional
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
    """The link between MDODisciplines and objective functions and constraints is made
    with MDOFunctionGenerator, which generates MDOFunctions from the disciplines.

    It uses closures to generate functions instances from a discipline execution.
    """

    def __init__(self, discipline):
        """Constructor.

        :param discipline: the discipline from which the generator
                           builds functions
        :type discipline: MDODiscipline
        """
        self.discipline = discipline

    def __make_function(self, input_names_list, output_names_list, default_inputs):
        """Makes a function from io and reference data Combines lambda function and a
        local def for function pointer issues Uses a closure.

        :param input_names_list: the dict keys of the input data
        :param output_names_list: the dict keys of the output data
        :param default_inputs: a default inputs dict to eventually overload
            the discipline's default inputs when evaluating the discipline
        :returns: The generated function
        """

        def func(x_vect):
            """A function which executes a discipline.

            :param x_vect: the input vector of the function
            :type x_vect: ndarray
            :returns: the selected outputs of the discipline
            :rtype: ndarray
            """
            for name in input_names_list:
                if name not in self.discipline.default_inputs:
                    raise ValueError(
                        "Discipline "
                        + str(self.discipline.name)
                        + " has no default_input named "
                        + str(name)
                        + ", while input is required"
                        + " by MDOFunction."
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

        def func_jac(x_vect):
            """A function which linearizes a discipline.

            :param x_vect: the input vector of the function
            :type x_vect: ndarray
            :returns: the selected outputs of the discipline
            :rtype: ndarray
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
        input_names_list,
        output_names_list,
        default_inputs=None,
        differentiable=True,
    ):
        """Builds a function from a discipline input and output lists.

        :param input_names_list: names of inputs of the disciplines
           to be inputs of the function
        :type input_names_list: list(str)
        :param output_names_list: names of outputs of the disciplines
            to be returned by the function
        :type output_names_list: list(str)
        :param default_inputs: a default inputs dict to eventually overload
            the discipline's default inputs when evaluating the discipline
        :type default_inputs: dict, optional
        :param differentiable: if True then inputs and outputs are added to the list
            of variables to be differentiated
        :type differentiable: bool, optional
        :returns: the function
        :rtype: MDOFunction
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
                "Some elements of "
                + str(input_names_list)
                + " are not inputs of the discipline, "
                + str(self.discipline.name)
                + " available inputs are:"
                + str(self.discipline.get_input_data_names())
            )

        if not self.discipline.is_all_outputs_existing(output_names_list):
            raise ValueError(
                "Some elements of "
                + str(output_names_list)
                + " are not outputs of the discipline, "
                + str(self.discipline.name)
                + " available outputs are: "
                + ", ".join(self.discipline.get_output_data_names())
                + "."
            )

        # adds inputs and outputs to the list of variables to be
        # differentiated
        if differentiable:
            self.discipline.add_differentiated_inputs(input_names_list)
            self.discipline.add_differentiated_outputs(output_names_list)

        return self.__make_function(input_names_list, output_names_list, default_inputs)


class NotImplementedCallable(object):
    """A not implemented callable object."""

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Function is not implemented")
