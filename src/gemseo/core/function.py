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
from __future__ import absolute_import, division, print_function, unicode_literals

from multiprocessing import Value
from numbers import Number

from future import standard_library
from future.utils import with_metaclass
from numpy import abs as np_abs
from numpy import (
    add,
    atleast_1d,
    atleast_2d,
    hstack,
    inner,
    ndarray,
    reshape,
    subtract,
    vstack,
    where,
)
from numpy.linalg import norm
from six import string_types

from gemseo.utils.data_conversion import DataConversion
from gemseo.utils.derivatives_approx import ComplexStep, FirstOrderFD
from gemseo.utils.singleton import SingleInstancePerAttributeId

standard_library.install_aliases()


from gemseo import LOGGER


class MDOFunction(object):
    """A container object to represent a function in an MDO process"""

    TYPE_OBJ = "obj"
    TYPE_EQ = "eq"
    TYPE_INEQ = "ineq"
    TYPE_OBS = "obs"
    AVAILABLE_TYPES = [TYPE_OBJ, TYPE_EQ, TYPE_INEQ, TYPE_OBS]
    DICT_REPR_ATTR = ["name", "f_type", "expr", "args", "dim"]

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
    ):
        """
        Initializes the function attributes

        :param func: the pointer to the function to be actually called
        :param name: the name of the function as a string
        :param f_type: the type of function among (obj, eq, ineq, obs)
        :param jac: the jacobian
        :param expr: the function expression
        :param args: the function arguments
        :param dim: the output dimension
        :param outvars: the array of variable names used as inputs
            of the function
        :param force_real: if True, cast the results to real value
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

    @property
    def n_calls(self):
        """
        Returns the number of calls to execute() which triggered the _run()
        multiprocessing safe
        """
        return self._n_calls.value

    @n_calls.setter
    def n_calls(self, value):
        """
        Sets the number of calls to execute() which triggered the _run()
        multiprocessing safe
        """
        with self._n_calls.get_lock():
            self._n_calls.value = value

    @property
    def func(self):
        """Accessor to the func property"""
        return self.__counted_f

    def __counted_f(self, x_vect):
        """
        Calls the function in self[self.SCIPY_FUN_TAG]

        :param x_vect: the function argument
        """
        with self._n_calls.get_lock():
            self._n_calls.value += 1
        val = self._func(x_vect)
        self.last_eval = val
        return val

    @func.setter
    def func(self, f_pointer):
        """Sets the f pointer

        :param f_pointer: the pointer to the function to be actually called
        """
        self._n_calls.value = 0
        self._func = f_pointer

    def __call__(self, x_vect):
        """
        Calls the function

        :param x_vect: the function argument
        """
        val = self.evaluate(x_vect, self.force_real)
        return val

    def evaluate(self, x_vect, force_real=False):
        """
        Evaluate the function

        :param x_vect: the function argument
        :param force_real: if True, cast the results to real value
        """
        val = self.__counted_f(x_vect)
        if force_real:
            val = val.real
        self.dim = atleast_1d(val).shape[0]
        return val

    @property
    def name(self):
        """Accessor to the name of the function"""
        return self._name

    @name.setter
    def name(self, name):
        """Sets the name of the function

        :param name: the name of the functioon
        """
        if name is not None and not isinstance(name, string_types):
            raise TypeError(
                "MDOFunction name must be a string, got " + str(type(name)) + " instead"
            )
        self._name = name

    @property
    def f_type(self):
        """Accessor to the function type"""
        return self._f_type

    @f_type.setter
    def f_type(self, f_type):
        """Sets the function type

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
        """Accessor to the jacobian"""
        return self._jac

    @jac.setter
    def jac(self, jac):
        """Sets the function jacobian

        :param jac: pointer to a jacobian function
        """
        if jac is not None:
            if not callable(jac):
                raise TypeError("Jacobian function must be callable")
        self._jac = jac

    @property
    def args(self):
        """Accessor to the arguments list"""
        return self._args

    @args.setter
    def args(self, args):
        """Sets the function arguments

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
        """Accessor to the expression"""
        return self._expr

    @expr.setter
    def expr(self, expr):
        """Sets the function expression

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
        """Accessor to the dimension"""
        return self._dim

    @dim.setter
    def dim(self, dim):
        """Sets the function dimension

        :param dim: int
        """
        if dim is not None and not int(dim) == dim:
            raise TypeError(
                "Dimension must be a integer. " + " Got " + str(type(int)) + " instead."
            )
        self._dim = dim

    @property
    def outvars(self):
        """Accessor to the array of input variable names
        used by the function
        """
        return self._outvars

    @outvars.setter
    def outvars(self, outvars):
        """Sets the array of output variable names used by the function

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
        """Returns True if self.f_type is eq or ineq"""
        return self.f_type in [self.TYPE_EQ, self.TYPE_INEQ]

    def __repr__(self):
        """
        Self representation as a string
        """
        str_repr = self.name
        if self.has_args():
            str_repr += "(" + (", ".join(self.args)) + ")"

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
        """Check if MDOFunction has a jacobian

        :returns: True if self has a jacobian
        """
        return self.jac is not None and not isinstance(
            self._jac, NotImplementedCallable
        )

    def has_dim(self):
        """Check if MDOFunction has a dimension

        :returns: True if self has a dimension
        """
        return self.dim is not None

    def has_outvars(self):
        """Check if MDOFunction has an array of input variables

        :returns: True if self has a dimension
        """
        return self.outvars is not None

    def has_expr(self):
        """Check if MDOFunction has an expression

        :returns: True if self has an expression
        """
        return self.expr is not None

    def has_args(self):
        """Check if MDOFunction has an args

        :returns: True if self has an args
        """
        return self.args is not None

    def has_f_type(self):
        """Check if MDOFunction has an type

        :returns: True if self has a type
        """
        return self.f_type is not None

    def apply_operator(self, other_f, operator, operator_repr):
        """Defines addition/substraction  for MDOFunction
        Supports automatic differentiation if other_f and self have a Jacobian

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
        """Defines addition/substraction  for MDOFunction
        Supports automatic differentiation if other_f and self have a Jacobian

        :param other_f: param operator:
        :param operator_repr: the representation as a string
        :param operator: the operator as a function pointer
        """
        assert isinstance(other_f, MDOFunction)

        def add_f_pt(x_vect):
            """Addition function

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
                """Jacobian of addition function

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
        """
        Defines addition/substraction  for MDOFunction
        Supports automatic differentiation if other_f and self have a Jacobian

        :param other_val: param operator: the value to add/subs
        :param operator_repr: the representation as a string
        :param operator: the operator as a function pointer
        """
        assert isinstance(other_val, Number)

        def add_f_pt(x_vect):
            """Addition function

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
        """
        Defines addition for MDOFunction
        Supports automatic differentiation if other_f and self have a jacobian
        """
        return self.apply_operator(other_f, operator=add, operator_repr="+")

    def __sub__(self, other_f):
        """
        Defines subtraction for MDOFunction
        Supports automatic differentiation if other_f and self have a jacobian
        """
        return self.apply_operator(other_f, operator=subtract, operator_repr="-")

    def __neg__(self):
        """
        Defines minus operator for MDOFunction (-f)
        Supports automatic differentiation if other_f and self have a Jacobian
        """

        def min_pt(x_vect):
            """Negative self function

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
                """Jacobian of negative self

                :param x_vect: design variables
                :returns: Jac of -self(x_vect)
                """
                return -self.jac(x_vect)  # pylint: disable=E1102

            min_self.jac = min_jac

        if self.has_expr():
            min_self.expr = "-" + self.expr

        return min_self

    def __mul__(self, other):
        """
        Defines multiplication for MDOFunction
        Supports automatic linearization if other_f and self have a jacobian
        """
        is_number = isinstance(other, Number)
        is_func = isinstance(other, MDOFunction)
        if not is_number and not is_func:
            raise TypeError(
                "Unuspported * operand for MDOFunction and " + str(type(other))
            )

        if is_func:

            def mul_f_pt(x_vect):
                """Multiplication function

                :param x_vect: design variables
                :returns: self(x_vect) added to other_f(x_vect)
                """
                return self(x_vect) * other(x_vect)

            out_name = self.name + "*" + other.name

        else:  # Func is a number

            def mul_f_pt(x_vect):
                """Multiplication function

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
                    """Jacobian of multiplication function

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
                """Jacobian of multiplication function

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
        """Adds an offset value to the function

        :param value: the offset value
        :returns: the offset function as an MDOFunction
        """

        def wrapped_function(x_vect):
            """Wrapped provided function in order to give to
            optimizer

            :param x_vect: design variables
            :returns: evaluation of function at design variables
                     plus the offset
            """
            return self(x_vect) + value

        expr = self.expr
        if expr is not None:
            expr += " + " + str(value)
        else:
            expr = self.name + " + " + str(value)
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

    def check_grad(self, x_vect, method="FirstOrderFD", step=1e-6, error_max=1e-8):
        """Checks the gradient of self

        :param x_vect: the vector at which the function is checked
        :param method: FirstOrderFD or ComplexStep
            (Default value = "FirstOrderFD")
        :param step: the step for approximation
            (Default value = 1e-6)
        :param error_max: Default value = 1e-8)
        """

        def rel_err(a_vect, b_vect):
            """Compute relative error

            :param a_vect: param b_vect:
            :param b_vect:

            """
            n_b = norm(b_vect)
            if n_b > error_max:
                return norm(a_vect - b_vect) / norm(b_vect)
            return norm(a_vect - b_vect)

        def filt_0(arr, floor_value=1e-6):
            """Filtering of error

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
            if not (shape_1d) or (shape_1d and flatten_diff):
                raise ValueError(
                    "Inconsistent function jacobian shape ! "
                    + "Got :"
                    + str(anal_grad.shape)
                    + " while expected :"
                    + str(apprx_grad.shape)
                )
        succeed = rel_err(anal_grad, apprx_grad) < error_max
        if not succeed:
            LOGGER.error("Function jacobian is wrong %s", str(self))
            LOGGER.error("Error =\n%s", str(filt_0(anal_grad - apprx_grad)))
            LOGGER.error("Analytic jacobian=\n%s", str(filt_0(anal_grad)))
            LOGGER.error("Approximate step gradient=\n%s", str(filt_0(apprx_grad)))
            raise ValueError("Function jacobian is wrong " + str(self))

    def get_data_dict_repr(self):
        """Returns a dict representation of self for serialization
        Pointers to functions are removed

        :returns: a dict with attributes names as keys
        """
        repr_dict = {}
        for attr_name in self.DICT_REPR_ATTR:
            attr = getattr(self, attr_name)
            if attr is not None:
                repr_dict[attr_name] = attr
        return repr_dict

    @staticmethod
    def init_from_dict_repr(**kwargs):
        """Initalizes a new Function from a data dict
        typically used for deserialization

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
        """
        self.__call__(x) returns f(x) if x is in the database
        and self.name in the database keys
        Idem for jac

        :param database: the database to read
        :param design_space: the design space used for normalization
        :param normalize: if True, x_n is unnormalized before call
        :param jac: if True, a jacobian pointer is also generated
        """
        name = self.name

        def read_in_db(x_n, fname):
            """
            Reads fname in the database for the x_n entry

            :param x_n: the x to evaluate fname
            :returns : f(x_n) if present in the fatabase
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
            """
            Computes f(x_n) from the database

            :param x_n: the x to evaluate f
            """
            return read_in_db(x_n, name)

        def j_from_db(x_n):
            """
            Computes df(x_n)/d x_n from the database

            :param x_n: the x to evaluate the jacobian
            """
            return read_in_db(x_n, "@" + name)

        self.func = f_from_db
        if jac:
            self.jac = j_from_db


class MDOLinearFunction(MDOFunction):
    """Linear multivariate function with coefficients (a_i)
    f(x) = sum_{i=1}^n a_i * x_i


    """

    def __init__(self, coefficients, name, f_type=None, args=None):
        """
        Initialize the function attributes

        :param coefficients: coefficients of the linear function
        :param name: the name of the function as a string
        :param f_type: the type of function among (obj, eq, ineq)
        :param args: the function arguments
        """
        self.coefficients = coefficients

        # define the function as a linear combination
        def fun(x_vect):
            """sum_{i=1}^n a_i * x_i

            :param x_vect: design variables
            """
            return inner(coefficients, x_vect)

        # Jacobian is constant
        def jac(_):
            """Constant Jacobian is the matrix of coefficients

            :param _: design variables

            """
            return atleast_2d(coefficients)

        expr = self._generate_expr()
        args = ["x"]

        super(MDOLinearFunction, self).__init__(
            fun, name, f_type=f_type, jac=jac, expr=expr, args=args, dim=1
        )

    def _generate_expr(self):
        """Generate the literal expression of the linear function"""
        expr = ""
        first_non_zero = -1
        for i in range(len(self.coefficients)):
            if self.coefficients[i] != 0.0:
                if first_non_zero == -1:
                    first_non_zero = i
                # sign
                if self.coefficients[i] == -1.0:
                    if i == first_non_zero:
                        expr += "-"  # unary minus
                elif i > 0 and first_non_zero < i:
                    if self.coefficients[i] < 0.0:
                        expr += " - "
                    else:
                        expr += " + "
                # coefficient
                if abs(self.coefficients[i]) != 1.0:
                    expr += str(abs(self.coefficients[i])) + "*"
                # variable
                expr += "x_" + str(i)
        return expr


class MDOFunctionGenerator(with_metaclass(SingleInstancePerAttributeId, object)):
    """The link between MDODisciplines and objective
    functions and constraints is made with MDOFunctionGenerator,
    which generates MDOFunctions from the disciplines.

    It uses closures to generate functions instances from a discipline
    execution.
    """

    def __init__(self, discipline):
        """
        Constructor

        :param discipline: the discipline from which the generator
                           builds functions
        """
        self.discipline = discipline

    def __make_function(self, input_names_list, output_names_list, default_inputs):
        """
        Makes a function from io and reference data
        Combines lambda function and a local def for function pointer issues
        Uses a closure.

        :param input_names_list: the dict keys of the input data
        :param output_names_list: the dict keys of the output data
        :param default_inputs: a default inputs dict to eventually overload
            the discipline's default inputs when evaluating the discipline
        :returns: The generated function
        """

        def func(x_vect):
            """A function which executes a discipline

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
            """A function which linearizes a discipline

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

    def get_function(self, input_names_list, output_names_list, default_inputs=None):
        """Builds a function from a discipline input and output lists.

        :param input_names_list: names of inputs of the disciplines
           to be inputs of the function
        :param output_names_list: names of outputs of the disciplines
            to be returned by the function
        :param default_inputs: a default inputs dict to eventually overload
            the discipline's default inputs when evaluating the discipline
        :returns: the function
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
        self.discipline.add_differentiated_inputs(input_names_list)
        self.discipline.add_differentiated_outputs(output_names_list)

        return self.__make_function(input_names_list, output_names_list, default_inputs)


class NotImplementedCallable(object):
    """
    A not implemented callable object
    """

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Function is not implemented")
