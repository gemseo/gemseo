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
#    INITIAL AUTHORS - API and implementation and/or documentation
#       :author: Damien Guenot
#       :author: Francois Gallard, Charlie Vanaret, Benoit Pauwels
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
    Optimization problem
    ====================

    The :class:`.OptimizationProblem` class is used to define
    the optimization problem from a :class:`.DesignSpace` defining:

    - Initial guess :math:`x_0`
    - Bounds :math:`l_b \\leq x \\leq u_b`

    (Possible vector) objective function is defined as a :class:`.MDOFunction`
    and set using the :code:`objective` attribute.
    If the optimization problem looks for the maximum of this objective
    function, the :meth:`.OptimizationProblem.change_objective_sign` changes
    the objective function sign because the optimization drivers seek to
    minimize this objective function.

    Equality and inequality constraints are also :class:`.MDOFunction` s
    provided to the :class:`.OptimizationProblem` by means of its
    :meth:`.OptimizationProblem.add_constraint` method.

    The :class:`.OptimizationProblem` allows to evaluate the different
    functions for a given design parameters vector
    (see :meth:`.OptimizationProblem.evaluate_functions`). Note that
    this evaluation step relies on an automated scaling of function wrt bounds
    so that optimizers and DOE algorithms work with scaled inputs
    between 0 and 1 for all variables.
    The :class:`.OptimizationProblem`  has also a :class:`.Database` storing
    the calls to all functions so that no function is called twice
    with the same inputs. Concerning the derivatives computation,
    the :class:`.OptimizationProblem` automates the generation of finite
    differences or complex step wrappers on functions,
    when analytical gradient is not available.

    Lastly, various getters and setters are available, as well as methods
    to export the :class:`.Database` to an HDF file or to a :class:`.Dataset`
    for future post-processing.
"""

from __future__ import absolute_import, division, unicode_literals

from functools import reduce
from numbers import Number

import h5py
from future import standard_library
from numpy import abs as np_abs
from numpy import all as np_all
from numpy import any as np_any
from numpy import argmin, array, concatenate, inf
from numpy import isnan as np_isnan
from numpy import issubdtype, ndarray
from numpy import number as np_number
from numpy import string_ as np_string
from numpy import where
from numpy.linalg import norm
from six import string_types

from gemseo.algos.database import Database
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt_result import OptimizationResult
from gemseo.algos.stop_criteria import DesvarIsNan, FunctionIsNan
from gemseo.core.dataset import Dataset
from gemseo.core.function import MDOFunction
from gemseo.utils.data_conversion import DataConversion
from gemseo.utils.derivatives_approx import ComplexStep, FirstOrderFD
from gemseo.utils.py23_compat import PY3, string_array

standard_library.install_aliases()
from gemseo import LOGGER


class OptimizationProblem(object):
    """An optimization problem class to store:

    - A (possibly vector) objective function
    - Constraints, equality and inequality
    - Initial guess x_0
    - Bounds l_bounds<= x <= u_bounds
    - The database of calls to all functions so that no function is called
      twice with the same inputs

    It also has an automated scaling of function wrt bounds
    so that optimizers and DOE algorithms work with scaled inputs
    between 0 and 1 for all variables.

    It automates the generation of finite differences or complex
    step wrappers on functions, when analytical gradient
    is not available
    """

    LINEAR_PB = "linear"
    NON_LINEAR_PB = "non-linear"
    AVAILABLE_PB_TYPES = [LINEAR_PB, NON_LINEAR_PB]

    USER_GRAD = "user"
    COMPLEX_STEP = "complex_step"
    FINITE_DIFFERENCES = "finite_differences"
    NO_DERIVATIVES = "no_derivatives"
    DIFFERENTIATION_METHODS = [
        USER_GRAD,
        COMPLEX_STEP,
        FINITE_DIFFERENCES,
        NO_DERIVATIVES,
    ]
    DESIGN_VAR_NAMES = "x_names"
    DESIGN_VAR_SIZE = "x_size"
    DESIGN_SPACE_ATTRS = ["u_bounds", "l_bounds", "x_0", DESIGN_VAR_NAMES, "dimension"]
    FUNCTIONS_ATTRS = ["objective", "constraints"]
    OPTIM_DESCRIPTION = [
        "minimize_objective",
        "fd_step",
        "differentiation_method",
        "pb_type",
        "ineq_tolerance",
        "eq_tolerance",
    ]

    OPT_DESCR_GROUP = "opt_description"
    DESIGN_SPACE_GROUP = "design_space"
    OBJECTIVE_GROUP = "objective"
    SOLUTION_GROUP = "solution"
    CONSTRAINTS_GROUP = "constraints"

    HDF5_FORMAT = "hdf5"
    GGOBI_FORMAT = "ggobi"

    def __init__(
        self,
        design_space,
        pb_type=NON_LINEAR_PB,
        input_database=None,
        differentiation_method=USER_GRAD,
        fd_step=1e-7,
    ):
        """
        Constructor for the optimization problem

        :param pb_type: the type of optimization problem among
               OptimizationProblem.AVAILABLE_PB_TYPES
        :param input_database: a file to eventually load the database
        :param differentiation_method: the default differentiation method
               for the functions of the optimization problem
        :param fd_step: finite differences or complex step step
        """
        self.objective = None
        self.nonproc_objective = None
        self.constraints = []
        self.nonproc_constraints = []
        self.observables = []
        self.new_iter_observables = []
        self.nonproc_observables = []
        self.nonproc_new_iter_observables = []
        self.minimize_objective = True
        self.fd_step = fd_step
        self.differentiation_method = differentiation_method
        self.pb_type = pb_type
        self.ineq_tolerance = 1e-4
        self.eq_tolerance = 1e-2
        self.__functions_are_preprocessed = False
        self.database = Database(input_hdf_file=input_database)
        self.solution = None
        self.design_space = design_space
        self.__x0 = None
        self.stop_if_nan = True
        self.__store_listeners = []
        self.__newiter_listeners = []
        self.preprocess_options = {}

    def add_constraint(self, cstr_func, value=None, cstr_type=None, positive=False):
        """Add constraints (equality and inequality) from MDOFunction

        :param cstr_func: constraints as an MDOFunction
        :param value: the target value for the constraint
            by default cstr(x)<= 0 or cstr(x)= 0 otherwise cstr(x)<=value
        :param cstr_type: constraint type (equality or inequality)
            (Default value = None)
        :param positive: positive/negative inequality constraint
            (Default value = False)
        """
        self.check_format(cstr_func)
        if value is not None:
            cstr_func = cstr_func.offset(-value)
        if positive:
            cstr_func = -cstr_func  # Operator overloading in MDOFunction
        if cstr_type is not None:
            cstr_func.f_type = cstr_type
        else:
            if not cstr_func.is_constraint():
                msg = "Constraint type must be provided, either in the "
                msg += "function attributes or to the add_constraint method"
                raise ValueError(msg)
        self.constraints.append(cstr_func)

    def add_eq_constraint(self, cstr_func, value=None):
        """Add equality constraints to the optimization problem

        :param cstr_func: MDOFunction constraints
        :param value: the target value for the constraint
            by default, cstr(x)=0 otherwise cstr(x)=value
        """
        self.add_constraint(cstr_func, value, cstr_type=MDOFunction.TYPE_EQ)

    def add_ineq_constraint(self, cstr_func, value=None, positive=False):
        """Add inequality constraints to the optimization problem

        :param cstr_func: MDOFunction constraints
        :param value: the target value for the constraint
            by default, cstr(x)<= 0 otherwise cstr(x)<=value
        :param positive: if True, the constraint should be
            cstr(x)>= value, by default  cstr(x)<= value
        """
        self.add_constraint(
            cstr_func, value, cstr_type=MDOFunction.TYPE_INEQ, positive=positive
        )

    def add_observable(self, obs_func, new_iter=True):
        """Adds observable as an MDOFunction.

        :param obs_func: observable as an MDOFunction
        :type obs_func: MDOFunction
        :param new_iter: if True, the observable will be called at each new
            iterate
        :type new_iter: bool
        """
        self.check_format(obs_func)
        obs_func.f_type = MDOFunction.TYPE_OBS
        self.observables.append(obs_func)
        if new_iter:
            self.new_iter_observables.append(obs_func)

    def get_eq_constraints(self):
        """Accessor for all equality constraints


        :returns: a list of equality constraints
        """

        def filter_eq(cstr):
            """A filter for equality constraints

            :param cstr: constraint function
            :returns: True if the function is an equality constraint
            """
            return cstr.f_type == MDOFunction.TYPE_EQ

        return list(filter(filter_eq, self.constraints))

    def get_ineq_constraints(self):
        """Accessor for all equality constraints

        :returns: a list of equality constraints
        """

        def filter_ineq(cstr):
            """A filter for equality constraints

            :param cstr: constraint function
            :returns: True if the function is an inequality constraint
            """
            return cstr.f_type == MDOFunction.TYPE_INEQ

        return list(filter(filter_ineq, self.constraints))

    def get_observable(self, name):
        """
        Returns the required observable.

        :param name: name of the observable
        :type name: str
        """
        try:
            observable = next(obs for obs in self.observables if obs.name == name)
        except StopIteration:
            raise ValueError("Observable {} cannot be found.".format(name))

        return observable

    def get_ineq_constraints_number(self):
        """Computes the number of inequality constraints

        :returns: the number of inequality constraints
        """
        return len(self.get_ineq_constraints())

    def get_eq_constraints_number(self):
        """Computes the number of equality constraints

        :returns: the number of equality constraints
        """
        return len(self.get_eq_constraints())

    def get_constraints_number(self):
        """Computes the number of equality constraints

        :returns: the number of equality constraints

        """
        return len(self.constraints)

    def get_constraints_names(self):
        """
        Get all constraints names as a list

        :returns: the list of constraints names
        """
        names = [constraint.name for constraint in self.constraints]
        return names

    def get_nonproc_constraints(self):
        """Returns the list of nonprocessed constraints."""
        return self.nonproc_constraints

    def get_design_variable_names(self):
        """Returns a list of all design variables names"""
        return self.design_space.variables_names

    def get_all_functions(self):
        """Returns a list of all functions of the MDO problem
        optimization constraints and objective
        """
        return [self.objective] + self.constraints + self.observables

    def get_all_functions_names(self):
        """
        Get all constraints and objective names

        :returns: a list of names of all functions of the MDO problem
            optimization constraints and objective
        """
        return [func.name for func in self.get_all_functions()]

    def get_objective_name(self):
        """
        Get objective function name

        :returns: the name of the actual objective function
        """
        return self.objective.name

    def get_nonproc_objective(self):
        """Returns the nonprocessed objective."""
        return self.nonproc_objective

    def has_nonlinear_constraints(self):
        """Checks if the problem has constraints

        :returns: True if the problem has equality or inequality constraints
        """
        return len(self.constraints) > 0

    def __notify_store_listeners(self):
        """
        Notifies the listeners that a new store has been made in the
        database
        """
        for func in self.__store_listeners:
            func()

    def __notify_newiter_listeners(self, xvect=None):
        """
        Notifies the listeners that a new iteration is ongoing

        :param xvect: design parameters
        :type xvect: ndarray
        """

        for func in self.__newiter_listeners:
            func()

        if xvect is not None:
            for obs in self.new_iter_observables:
                obs(xvect)

    def has_constraints(self):
        """Checks if the problem has equality or inequality constraints

        :returns: True if the problem has  constraints
        """
        return self.has_eq_constraints() or self.has_ineq_constraints()

    def has_eq_constraints(self):
        """Checks if the problem has equality constraints

        :returns: True if the problem has equality constraints
        """
        return len(self.get_eq_constraints()) > 0

    def has_ineq_constraints(self):
        """Checks if the problem has inequality constraints

        :returns: True if the problem has inequality constraints

        """
        return len(self.get_ineq_constraints()) > 0

    def get_x0_normalized(self):
        """Accessor for the normalized x_0

        :returns: x0-lb)/(ub-lb)
        """
        dspace = self.design_space
        return dspace.normalize_vect(dspace.get_current_x())

    def get_dimension(self):
        """
        Get the total number of design variables

        :returns: the dimension of the design space
        """
        return self.design_space.dimension

    @property
    def dimension(self):
        """dimension property, ie dimension of the design space"""
        return self.design_space.dimension

    @staticmethod
    def check_format(input_function):
        """Checks that the input_function is an istance of MDOFunction

        :param input_function: function
        """
        if not isinstance(input_function, MDOFunction):
            raise TypeError(
                "Optimization problem functions must be "
                + "instances of gemseo.core.MDOFunction"
            )

    def get_eq_cstr_total_dim(self):
        """Returns the total number of equality constraints dimensions
        that is the sum of all outputs dimensions of
        all constraints

        :returns: total number of equality constraints
        """
        return self.__count_cstr_total_dim(MDOFunction.TYPE_EQ)

    def get_ineq_cstr_total_dim(self):
        """Returns the total number of inequality constraints dimensions
        that is the sum of all outputs dimensions of
        all constraints

        :returns: total number of inequality constraints dimensions
        """
        return self.__count_cstr_total_dim(MDOFunction.TYPE_INEQ)

    def __count_cstr_total_dim(self, cstr_type):
        """
        Returns the total number of equality or inequality constraints
        dimensions that is the sum of all outputs dimensions of
        all constraints

        :param cstr_type: the type of contraint, TYPE_INEQ or TYPE_EQ
        :returns: total number of constraints dimensions
        """
        n_cstr = 0
        for constraint in self.constraints:
            if not constraint.has_dim():
                raise ValueError(
                    "Constraint dimension not available yet,"
                    + " please call function "
                    + str(constraint)
                    + " once."
                )
            if constraint.f_type == cstr_type:
                n_cstr += constraint.dim
        return n_cstr

    def get_active_ineq_constraints(self, x_vect, tol=1e-6):
        """
        Returns active constraints names and indices

        :param x_vect: vector of x values, not normalized
        """
        self.design_space.check_membership(x_vect)
        normalize = self.preprocess_options.get("normalize", False)
        if normalize:
            x_vect = self.design_space.normalize_vect(x_vect)

        act_funcs = {}

        for func in self.get_ineq_constraints():
            val = np_abs(func(x_vect))
            act_funcs[func] = where(val <= tol, True, False)

        return act_funcs

    def __wrap_in_database(self, orig_func, normalized=True, is_observable=False):
        """
        Wraps the function to test if it is already in the database
        and store its evaluation

        :param  orig_func: the MDOFunction to be wrapped
        :param  normalized: if True, the input of orig_func are assumed
            normalized; otherwise they are assumed non-normalized
        :param is_observable: if True, new_iter_listeners are not called when
            function is called (avoid recursive call)
        :returns: the wrapped function as an MDOFunction
        """
        fname = orig_func.name
        normalize_vect = self.design_space.normalize_vect
        unnormalize_vect = self.design_space.unnormalize_vect

        def unnormalize_gradient(x_vect):
            """Unnormalize gradient

            :param x_vect: gradient
            :return: unnormalized gradient
            """
            return normalize_vect(x_vect, minus_lb=False)

        def normalize_gradient(x_vect):
            """Normalize gradient

            :param x_vect: gradient
            :return: normalized gradient
            """
            return unnormalize_vect(x_vect, minus_lb=False, no_check=True)

        def wrapped_function(x_vect):
            """Wrapped provided function in order to give to
            optimizer

            :param x_vect: design variable
            :returns: evaluation of function at x_vect
            """
            if np_any(np_isnan(x_vect)):
                raise DesvarIsNan(
                    "Design Variables contain a NaN value !" + str(x_vect)
                )
            if normalized:
                xn_vect = x_vect
                xu_vect = unnormalize_vect(xn_vect)
            else:
                xu_vect = x_vect
                xn_vect = normalize_vect(xu_vect)
            # try to retrieve the evaluation
            value = None
            if not self.database.get(xu_vect, False) and not is_observable:
                if normalized:
                    self.__notify_newiter_listeners(xn_vect)
                else:
                    self.__notify_newiter_listeners(xu_vect)
            else:
                value = self.database.get_f_of_x(fname, xu_vect)
            if value is None:
                # if not evaluated yet, evaluate
                if normalized:
                    value = orig_func(xn_vect)
                else:
                    value = orig_func(xu_vect)
                if self.stop_if_nan and np_any(np_isnan(value)):
                    raise FunctionIsNan(
                        "Function " + str(fname) + " is NaN for x=" + str(xu_vect)
                    )
                values_dict = {fname: value}
                # store (x, f(x)) in database
                self.database.store(xu_vect, values_dict)
                self.__notify_store_listeners()

            return value

        db_func = MDOFunction(
            wrapped_function,
            name=fname,
            f_type=orig_func.f_type,
            expr=orig_func.expr,
            args=orig_func.args,
            dim=orig_func.dim,
            outvars=orig_func.outvars,
        )

        if orig_func.has_jac():

            def dwrapped_function(x_vect):
                """Wrapped provided gradient in order to give to
                optimizer

                :param x_vect: design variable
                :returns: evaluation of gradient at x_vect
                """
                if np_any(np_isnan(x_vect)):
                    raise FunctionIsNan(
                        "Design Variables contain a NaN " + "value !" + str(x_vect)
                    )
                if normalized:
                    xn_vect = x_vect
                    xu_vect = unnormalize_vect(xn_vect)
                else:
                    xu_vect = x_vect
                    xn_vect = normalize_vect(xu_vect)
                # try to retrieve the evaluation
                jac_u = None
                if not self.database.get(xu_vect, False):
                    self.__notify_newiter_listeners()
                else:
                    jac_u = self.database.get_f_of_x(Database.GRAD_TAG + fname, xu_vect)

                if jac_u is not None:
                    jac_n = normalize_gradient(jac_u)
                else:
                    # if not evaluated yet, evaluate
                    if normalized:
                        jac_n = orig_func.jac(xn_vect).real
                        jac_u = unnormalize_gradient(jac_n)
                    else:
                        jac_u = orig_func.jac(xu_vect).real
                        jac_n = normalize_gradient(jac_u)
                    if np_any(np_isnan(jac_n)) and self.stop_if_nan:
                        raise FunctionIsNan(
                            "Function "
                            + str(fname)
                            + " 's Jacobian is NaN for x="
                            + str(xu_vect)
                        )
                    values_dict = {Database.GRAD_TAG + fname: jac_u}
                    # store (x, j(x)) in database
                    self.database.store(xu_vect, values_dict)
                    self.__notify_store_listeners()
                if normalized:
                    jac = jac_n
                else:
                    jac = jac_u
                return jac

            db_func.jac = dwrapped_function

        return db_func

    def add_callback(self, callback_func, each_new_iter=True, each_store=False):
        """Adds a callback function after each store operation or new iteration

        :param callback_func: a function called after the
            function if None nothing
        :param each_new_iter: if True, callback at every iteration
        :param each_store: if True, callback at every call to store()
            in the database
        """
        if each_store:
            self.add_store_listener(callback_func)
        if each_new_iter:
            self.add_new_iter_listener(callback_func)

    def add_store_listener(self, listener_func):
        """
        When an item is stored to the database, calls the listener
        functions

        :param listener_func : function to be called
        :param args: optional arguments of function
        """
        if not callable(listener_func):
            raise TypeError("Listener function is not callable")
        self.__store_listeners.append(listener_func)

    def add_new_iter_listener(self, listener_func):
        """
        When a new iteration stored to the database, calls the listener
        functions

        :param listener_func : function to be called
        :param args: optional arguments of function
        """
        if not callable(listener_func):
            raise TypeError("Listener function is not callable")
        self.__newiter_listeners.append(listener_func)

    def clear_listeners(self):
        """
        Clears all the new_iter and store listeners
        """
        self.__store_listeners = []
        self.__newiter_listeners = []

    def evaluate_functions(
        self,
        x_vect=None,
        eval_jac=False,
        eval_obj=True,
        normalize=True,
        no_db_no_norm=False,
    ):
        """Compute objective and constraints at x_normed
        Some libraries require the number of constraints as an input parameter
        which is unknown by formulation/scenario. Evaluation of initial point
        allows to get this mandatory informations.
        Also used for DOE to evaluate samples

        :param x_normed: the normalized vector at which the
            point must be evaluated if None, x_0 is used (Default value = None)
        :param eval_jac: if True, the jacobian is also evaluated
            (Default value = False)
        :param eval_obj: if True, the objective is evaluated
            (Default value = True)
        :param no_db_no_norm: if True, dont use preprocessed functions,
            so we have no database, nor normalization
        """
        if no_db_no_norm and normalize:
            raise ValueError(
                "Can't use no_db_no_norm " + "and normalize options together"
            )
        if normalize:
            if x_vect is None:
                x_vect = self.get_x0_normalized()
            else:
                # Checks proposed x wrt bounds
                x_u_r = self.design_space.unnormalize_vect(x_vect)
                self.design_space.check_membership(x_u_r)
        else:
            if x_vect is None:
                x_vect = self.design_space.get_current_x()
            else:
                # Checks proposed x wrt bounds
                self.design_space.check_membership(x_vect)

        if no_db_no_norm:
            if eval_obj:
                functions = self.nonproc_constraints + [self.nonproc_objective]
            else:
                functions = self.nonproc_constraints
        else:
            if eval_obj:
                functions = self.constraints + [self.objective]
            else:
                functions = self.constraints

        outputs = {}
        for func in functions:
            try:
                outputs[func.name] = func(x_vect)
            except ValueError:
                LOGGER.error("Failed to evaluate function %s", str(func.name))
                raise

        jacobians = {}
        if eval_jac:
            for func in functions:
                try:
                    jacobians[func.name] = func.jac(x_vect)
                except ValueError:
                    msg = "Failed to evaluate " "jacobian of {}".format(str(func.name))
                    LOGGER.error(msg)
                    raise

        return outputs, jacobians

    def preprocess_functions(self, normalize=True, use_database=True, round_ints=True):
        """Preprocesses all the functions: objective and constraints
        to wrap them with the database and eventually
        the gradients by complex step or FD
        :param normalize: if True, the function is normalized
        :type normalize: bool
        :param use_database: if True, the function is wrapped in the database
        :type use_database: bool
        :param round_ints: if True, rounds integer variables
        :type round_ints: bool
        """
        # Avoids multiple wrappings of functions when multiple executions
        # are performed, in bi level scenarios for instance
        if not self.__functions_are_preprocessed:
            self.preprocess_options = {
                "normalize": normalize,
                "use_database": use_database,
                "round_ints": round_ints,
            }
            # Preprocess the constraints
            for icstr, cstr in enumerate(self.constraints):
                non_p_cstr = self.__preprocess_func(
                    cstr, normalize=False, use_database=False, round_ints=round_ints
                )
                self.nonproc_constraints.append(non_p_cstr)
                p_cstr = self.__preprocess_func(
                    cstr, normalize=normalize, use_database=use_database
                )
                self.constraints[icstr] = p_cstr
            # Preprocess the observables
            for iobs, obs in enumerate(self.observables):
                non_p_obs = self.__preprocess_func(
                    obs,
                    normalize=False,
                    use_database=False,
                    round_ints=round_ints,
                    is_observable=True,
                )
                self.nonproc_observables.append(non_p_obs)
                p_obs = self.__preprocess_func(
                    obs,
                    normalize=normalize,
                    use_database=use_database,
                    is_observable=True,
                )
                self.observables[iobs] = p_obs
            for iobs, obs in enumerate(self.new_iter_observables):
                non_p_obs = self.__preprocess_func(
                    obs,
                    normalize=False,
                    use_database=False,
                    round_ints=round_ints,
                    is_observable=True,
                )
                self.nonproc_new_iter_observables.append(non_p_obs)
                p_obs = self.__preprocess_func(
                    obs,
                    normalize=normalize,
                    use_database=use_database,
                    is_observable=True,
                )
                self.new_iter_observables[iobs] = p_obs
            # Preprocess the objective
            np_o = self.__preprocess_func(
                self.objective,
                normalize=False,
                use_database=False,
                round_ints=round_ints,
            )
            self.nonproc_objective = np_o
            self.objective = self.__preprocess_func(
                self.objective,
                normalize=normalize,
                use_database=use_database,
                round_ints=round_ints,
            )
            self.objective.f_type = MDOFunction.TYPE_OBJ
            self.__functions_are_preprocessed = True
            self.check()

    def __preprocess_func(
        self,
        function,
        normalize=True,
        use_database=True,
        round_ints=True,
        is_observable=False,
    ):
        """
        Wraps the function to: differentiate it and store its call in
        the database. Only the computed gradients are stored in the database,
        not the eventual finite differences or complex step
        perturbed evaluations

        :param function: the scaled and derived MDOFunction
        :param normalize: if True, the function is normalized
        :param use_database: if True, the function is wrapped in the database
        :param round_ints: if True, rounds integer variables before call
        :param is_observable: if True, new_iter_listeners are not called when
            function is called (avoid recursive call)
        :returns: the preprocessed function
        """
        self.check_format(function)
        # First differentiate it so that the finite differences evaluations
        # are not stored in the database, which would be the case in the other
        # way round
        # Also, store non normalized values in the database for further
        # exploitation
        function = self.__normalize_and_round(function, normalize, round_ints)
        self.__add_fd_jac(function)

        # Cast to real value, the results can be a complex number (ComplexStep)
        function.force_real = True
        if use_database:
            function = self.__wrap_in_database(function, normalize, is_observable)
        return function

    def __normalize_and_round(self, orig_func, normalize, round_ints):
        """
        Create a function that takes a scaled input vector
        instead of the original input vector

        :param orig_func: the function as an MDOFunction
        :param normalize: if True, the function is normalized
        :param round_ints: if True, rounds integer variables before call

        :returns: the normalized and differentiated function
        """
        if (not normalize) and (not round_ints):
            return orig_func

        unnormalize_vect = self.design_space.unnormalize_vect

        def normalize_gradient(x_vect):
            """Normalize gradient

            :param x_vect: gradient
            :return: normalized gradient
            """
            return unnormalize_vect(x_vect, minus_lb=False, no_check=True)

        round_int_vars = self.design_space.round_vect

        def f_wrapped(x_vect):
            """Unnormalize design vector for function evaluation

            :param x_vect: normalized design vector
            :returns: function value at x_vect
            """
            if normalize:
                x_vect = unnormalize_vect(x_vect)
            if round_ints:
                x_vect = round_int_vars(x_vect)
            return orig_func(x_vect)

        normed_func = MDOFunction(
            f_wrapped,
            name=orig_func.name,
            f_type=orig_func.f_type,
            expr=orig_func.expr,
            args=orig_func.args,
            dim=orig_func.dim,
            outvars=orig_func.outvars,
        )

        def df_wrapped(x_vect):
            """Unnormalize design vector for gradient evaluation

            :param x_vect: normalized design vector
            :returns: gradient value at xn
            """
            if not orig_func.has_jac():
                raise ValueError(
                    "Selected user gradient "
                    + " but function "
                    + str(orig_func)
                    + " has no Jacobian matrix !"
                )
            if normalize:
                x_vect = unnormalize_vect(x_vect)
            if round_ints:
                x_vect = round_int_vars(x_vect)
            g_u = orig_func.jac(x_vect)
            if normalize:
                return normalize_gradient(g_u)
            return g_u

        normed_func.jac = df_wrapped

        return normed_func

    def __add_fd_jac(self, function):
        """
        Adds a pointer to the jacobian of the function
        generated either by COMPLEX_STEP or FINITE_DIFFERENCES

        :param function: the function to be derivated
        """
        if self.differentiation_method == self.COMPLEX_STEP:
            c_s = ComplexStep(function.evaluate, self.fd_step)
            function.jac = c_s.f_gradient

        if self.differentiation_method == self.FINITE_DIFFERENCES:
            f_d = FirstOrderFD(function, self.fd_step)
            function.jac = f_d.f_gradient

    def check(self):
        """Checks if the optimization problem is ready for run"""
        if self.objective is None:
            raise ValueError("Missing objective function" + " in OptimizationProblem")
        self.__check_pb_type()
        self.design_space.check()
        self.__check_differentiation_method()
        self.check_format(self.objective)
        self.__check_functions()

    def __check_functions(self):
        """
        Checks that the constraints are well declared
        """
        for cstr in self.constraints:
            self.check_format(cstr)
            if not cstr.is_constraint():
                raise ValueError(
                    "Constraint type is not eq or ineq !, got "
                    + str(cstr.f_type)
                    + " instead "
                )
        self.check_format(self.objective)

    def __check_pb_type(self):
        """
        Checks that the pb_type is among self.AVAILABLE_PB_TYPES
        """
        if self.pb_type not in self.AVAILABLE_PB_TYPES:
            raise TypeError(
                "Unknown problem type "
                + str(self.pb_type)
                + ", available problem types are "
                + str(self.AVAILABLE_PB_TYPES)
            )

    def __check_differentiation_method(self):
        """
        Checks that the differentiation method is in allowed ones
        """
        if self.differentiation_method not in self.DIFFERENTIATION_METHODS:
            raise ValueError(
                "Differentiation method "
                + str(self.differentiation_method)
                + " is not among the supported ones : "
                + str(self.DIFFERENTIATION_METHODS)
            )

        if self.differentiation_method == self.COMPLEX_STEP:
            if self.fd_step == 0:
                raise ValueError("ComplexStep step is null !")
            if self.fd_step.imag != 0:
                LOGGER.warning(
                    "Complex step method has an imaginary "
                    "step while required a pure real one."
                    " Auto setting the real part"
                )
                self.fd_step = self.fd_step.imag
        elif self.differentiation_method == self.FINITE_DIFFERENCES:
            if self.fd_step == 0:
                raise ValueError("Finite differences step is null !")
            if self.fd_step.imag != 0:
                LOGGER.warning(
                    "Finite differences method has a complex "
                    "step while required a pure real one."
                    " Auto setting the imaginary part to 0"
                )
                self.fd_step = self.fd_step.real

    def change_objective_sign(self):
        """Changes the objective function sign, when it needs to be maximized
        for instance
        """
        self.minimize_objective = not self.minimize_objective
        self.objective = -self.objective
        # Use MDOFunction Operator overloading

    def _satisfied_constraint(self, cstr_type, value):
        """Determine if an evaluation satisfies a constraint within
        a given tolerance

        :param cstr_type: type of the constraint
        :param value: evaluation of the constraint
        """
        if cstr_type == MDOFunction.TYPE_EQ:
            return np_all(np_abs(value) <= self.eq_tolerance)
        return np_all(value <= self.ineq_tolerance)

    def is_point_feasible(self, out_val, constraints=None):
        """
        Returns True if the point is feasible

        :param out_val: dict of values, containing objective function
            and eventually constraints.
            Warning: if the constraint value is not present,
            the constraint will be considered satisfied
        :param constraints: the list of constraints (MDOFunctions)
            to check. If None, takes all constraints of the problem
        """
        if constraints is None:
            constraints = self.get_ineq_constraints() + self.get_eq_constraints()
        feasible = True
        for constraint in constraints:
            # look for the evaluation of the constraint
            eval_cstr = out_val.get(constraint.name, None)
            # if evaluation exists, check if it is satisfied
            if eval_cstr is None or not self._satisfied_constraint(
                constraint.f_type, eval_cstr
            ):
                feasible = False
                break
        return feasible

    def get_feasible_points(self):
        """Return the list of feasible points within a given tolerance
        eq_tolerance and  ineq_tolerance are taken from sel attrs
        """
        x_history = []
        f_history = []
        constraints = self.get_ineq_constraints() + self.get_eq_constraints()

        for x_vect, out_val in self.database.items():
            feasible = self.is_point_feasible(out_val, constraints=constraints)

            # if all constraints are satisfied, store the vector
            if feasible:
                x_history.append(x_vect.unwrap())
                f_history.append(out_val)
        return x_history, f_history

    def get_violation_criteria(self, x_vect):
        """
        Computes a violation measure associated to an iteration
        For each constraints, when it is violated, add the absolute
        distance to zero, in L2 norm

        if 0, all constraints are satisfied

        :param x_vect: vector of design variables
        :returns: True if feasible, and the violation criteria
        """
        f_violation = 0.0
        is_pt_feasible = True
        constraints = self.get_ineq_constraints() + self.get_eq_constraints()
        out_val = self.database.get(x_vect)
        for constraint in constraints:
            # look for the evaluation of the constraint
            eval_cstr = out_val.get(constraint.name, None)
            # if evaluation exists, check if it is satisfied
            if eval_cstr is None:
                break
            if not self._satisfied_constraint(constraint.f_type, eval_cstr):
                is_pt_feasible = False
                if constraint.f_type == MDOFunction.TYPE_INEQ:
                    if isinstance(eval_cstr, ndarray):
                        viol_inds = where(eval_cstr > self.ineq_tolerance)
                        f_violation += (
                            norm(eval_cstr[viol_inds] - self.ineq_tolerance) ** 2
                        )
                    else:
                        f_violation += (eval_cstr - self.ineq_tolerance) ** 2
                else:
                    f_violation += norm(abs(eval_cstr) - self.eq_tolerance) ** 2
        return is_pt_feasible, f_violation

    def get_best_infeasible_point(self):
        """Return the best infeasible point within a given tolerance """
        x_history = []
        f_history = []
        is_feasible = []
        viol_criteria = []
        for x_vect, out_val in self.database.items():
            is_pt_feasible, f_violation = self.get_violation_criteria(x_vect)
            is_feasible.append(is_pt_feasible)
            viol_criteria.append(f_violation)
            x_history.append(x_vect.unwrap())
            f_history.append(out_val)

        is_opt_feasible = False
        if viol_criteria:
            best_i = int(argmin(array(viol_criteria)))
            is_opt_feasible = is_feasible[best_i]
        else:
            best_i = 0

        opt_f_dict = {}
        if len(f_history) <= best_i:
            f_opt = None
            x_opt = None
        else:
            f_opt = f_history[best_i].get(self.objective.name)
            x_opt = x_history[best_i]
            opt_f_dict = f_history[best_i]
        return x_opt, f_opt, is_opt_feasible, opt_f_dict

    def __get_optimum_infeas(self):
        """Return the optimum solution within a given feasibility
        tolerances, when there is no feasible point

        :returns: tuple, best evaluation iteration and solution
        """
        msg = (
            "Optimization found no feasible point ! "
            + " The least infeasible point is selected."
        )
        LOGGER.warning(msg)
        x_opt, f_opt, _, f_history = self.get_best_infeasible_point()
        c_opt = {}
        c_opt_grad = {}
        constraints = self.get_ineq_constraints() + self.get_eq_constraints()
        for constraint in constraints:
            c_opt[constraint.name] = f_history.get(constraint.name)
            f_key = Database.GRAD_TAG + constraint.name
            c_opt_grad[constraint.name] = f_history.get(f_key)
        return x_opt, f_opt, c_opt, c_opt_grad

    def __get_optimum_feas(self, feas_evals, feas_pts):
        """Return the optimum solution within a given feasibility
        tolerances, when there is a feasible point

        :param fea_evals: feasible evaluation list
        :parm feas_pts: feasible points list
        :returns: tuple, best evaluation iteration and solution
        """
        f_opt, x_opt = inf, None
        c_opt = {}
        c_opt_grad = {}
        obj_name = self.objective.name
        constraints = self.get_ineq_constraints() + self.get_eq_constraints()
        for (i, out_val) in enumerate(feas_evals):
            obj_eval = out_val.get(obj_name)
            if obj_eval is None or isinstance(obj_eval, Number) or obj_eval.size == 1:
                tmp_objeval = obj_eval
            else:
                tmp_objeval = norm(obj_eval)
            if tmp_objeval is not None and tmp_objeval < f_opt:
                f_opt = tmp_objeval
                x_opt = feas_pts[i]
                for constraint in constraints:
                    c_name = constraint.name
                    c_opt[c_name] = feas_evals[i].get(c_name)
                    c_key = Database.GRAD_TAG + c_name
                    c_opt_grad[constraint.name] = feas_evals[i].get(c_key)
        return x_opt, f_opt, c_opt, c_opt_grad

    def get_optimum(self):
        """Return the optimum solution within a given feasibility
        tolerances

        :returns: tuple, best evaluation iteration and solution
        """
        if not self.database:
            raise ValueError("Optimization history is empty")
        feas_pts, feas_evals = self.get_feasible_points()

        if not feas_pts:
            is_feas = False
            x_opt, f_opt, c_opt, c_opt_d = self.__get_optimum_infeas()
        else:
            is_feas = True
            x_opt, f_opt, c_opt, c_opt_d = self.__get_optimum_feas(feas_evals, feas_pts)
        return f_opt, x_opt, is_feas, c_opt, c_opt_d

    def _get_msg(self, max_ds_size=40):
        """
        Get logging message representing self

        :param max_ds_size: maximum design space dimension
            to print
        """
        msg = "Optimization problem:\n"
        msg += "      Minimize: "
        msg += str(self.objective) + "\n"
        variables_names = self.design_space.variables_names
        msg += "With respect to: \n"
        msg += "    " + str(", ".join(variables_names)) + "\n"

        if self.has_constraints():
            msg += "Subject to constraints: \n"
            for eq_c in self.get_eq_constraints():
                cst_exprs = [_f for _f in str(eq_c).split("\n") if _f]
                for expr_spl in cst_exprs:
                    msg += expr_spl + " = 0\n"

            for ieq_c in self.get_ineq_constraints():
                cst_exprs = [_f for _f in str(ieq_c).split("\n") if _f]
                for expr_spl in cst_exprs:
                    msg += expr_spl + " <= 0\n"
        if self.design_space.dimension <= max_ds_size:
            msg += str(self.design_space)
        return msg

    def __str__(self):
        """
        Gets a string representing the optimization problem
        only the formulation constraints are displayed

        :returns: the opt pb representation
        """

        return self._get_msg()

    def log_me(self, max_ds_size=40):
        """Logs a representation of the optimization problem characteristics
        logs self.__repr__ message

        :param max_ds_size: maximum design space dimension
            to print
        """
        msg = self._get_msg(max_ds_size)
        for line in msg.split("\n"):
            LOGGER.info(line)

    @staticmethod
    def __store_h5data(group, data_array, dataset_name, dtype=None):
        """
        Stores an array in a hdf5 file group

        :param group: the group pointer
        :param data_array: the data to stor
        :param dataset_name: name of the dataset to store the array
        """
        if data_array is None:
            return
        if isinstance(data_array, ndarray):
            data_array = data_array.real
        if isinstance(data_array, string_types):
            data_array = string_array([data_array])
        if isinstance(data_array, list):
            all_str = reduce(
                lambda x, y: x or y,
                (isinstance(data, string_types) for data in data_array),
            )
            if all_str:
                data_array = string_array(data_array)
                dtype = data_array.dtype
        group.create_dataset(dataset_name, data=data_array, dtype=dtype)

    @staticmethod
    def __store_attr_h5data(obj, group):
        """
        Stores an object that has a get_data_dict_repr attribute
        in the hdf5 dataset

        :param obj: the object to store
        :param group: the hdf5 group
        """
        data_dict = obj.get_data_dict_repr()
        for attr_name, attr in data_dict.items():
            dtype = None
            is_arr_n = isinstance(attr, ndarray) and issubdtype(attr.dtype, np_number)
            if isinstance(attr, string_types):
                attr = attr.encode("ascii", "ignore")
            elif isinstance(attr, bytes):
                attr = attr.decode()
            elif hasattr(attr, "__iter__") and not is_arr_n:
                if PY3:
                    attr = [
                        att.encode("ascii", "ignore")
                        if isinstance(att, string_types)
                        else att
                        for att in attr
                    ]
                dtype = h5py.special_dtype(vlen=str)
            OptimizationProblem.__store_h5data(group, attr, attr_name, dtype)

    def export_hdf(self, file_path, append=False):
        """Export optimization problem to hdf file.

        :param file_path: file to store the data
        :param append: if True, data is appended to the file if not empty
            (Default value = False)
        """
        LOGGER.info("Export optimization problem to file: %s", str(file_path))
        if append:
            mode = "a"
        else:
            mode = "w"
        h5file = h5py.File(file_path, mode)

        if not append or self.OPT_DESCR_GROUP not in h5file:
            opt_group = h5file.require_group(self.OPT_DESCR_GROUP)
            for attr_name in self.OPTIM_DESCRIPTION:
                attr = getattr(self, attr_name)
                self.__store_h5data(opt_group, attr, attr_name)

            obj_group = h5file.require_group(self.OBJECTIVE_GROUP)
            self.__store_attr_h5data(self.objective, obj_group)

            if self.constraints:
                cstr_group = h5file.require_group(self.CONSTRAINTS_GROUP)
                for cstr in self.constraints:
                    name = cstr.name
                    c_subgroup = cstr_group.require_group(name)
                    self.__store_attr_h5data(cstr, c_subgroup)

            if hasattr(self.solution, "get_data_dict_repr"):
                sol_group = h5file.require_group(self.SOLUTION_GROUP)
                self.__store_attr_h5data(self.solution, sol_group)

        no_designspace = DesignSpace.DESIGN_SPACE_GROUP not in h5file
        h5file.close()

        self.database.export_hdf(file_path, append=True)
        # Design space shall remain the same in append mode
        if not append or no_designspace:
            self.design_space.export_hdf(file_path, append=True)

    @staticmethod
    def import_hdf(file_path, x_tolerance=0.0):
        """Imports optimization history from hdf file

        :param file_path: file to deserialize
        :returns: the read optimization problem
        """
        LOGGER.info("Import optimization problem from file: %s", str(file_path))
        design_space = DesignSpace(file_path)
        opt_pb = OptimizationProblem(design_space, input_database=file_path)
        h5file = h5py.File(file_path, "r")
        try:
            group = h5file[opt_pb.OPT_DESCR_GROUP]
            for attr_name, attr in group.items():
                val = attr.value
                val = val.decode() if isinstance(val, bytes) else val
                setattr(opt_pb, attr_name, val)

            if opt_pb.SOLUTION_GROUP in h5file:
                group = h5file[opt_pb.SOLUTION_GROUP]
                data_dict = {attr_name: attr.value for attr_name, attr in group.items()}
                data_dict = OptimizationProblem.__cast_bytes_dict(data_dict)
                attr = OptimizationResult.init_from_dict_repr(**data_dict)
                opt_pb.solution = attr

            group = h5file[opt_pb.OBJECTIVE_GROUP]
            data_dict = {attr_name: attr.value for attr_name, attr in group.items()}
            data_dict = OptimizationProblem.__cast_bytes_dict(data_dict)
            attr = MDOFunction.init_from_dict_repr(**data_dict)
            # The generated functions can be called at the x stored in
            # the database
            attr.set_pt_from_database(
                opt_pb.database, design_space, jac=True, x_tolerance=x_tolerance
            )
            opt_pb.objective = attr

            if opt_pb.CONSTRAINTS_GROUP in h5file:
                group = h5file[opt_pb.CONSTRAINTS_GROUP]
                for cstr_name in group.keys():
                    sub_group = group[cstr_name]
                    data_dict = {
                        attr_name: attr.value for attr_name, attr in sub_group.items()
                    }
                    data_dict = OptimizationProblem.__cast_bytes_dict(data_dict)
                    attr = MDOFunction.init_from_dict_repr(**data_dict)
                    opt_pb.constraints.append(attr)
        except KeyError as err:
            h5file.close()
            raise KeyError(
                "Invalid database hdf5 file, missing dataset. " + err.args[0]
            )

        h5file.close()
        return opt_pb

    def export_to_dataset(self, name, by_group=True, categorize=True, opt_naming=True):
        """Export the optimization problem to a :class:`.Dataset`.

        :param str name: dataset name.
        :param bool by_group: if True, store the data by group. Otherwise,
            store them by variables. Default: True
        :param bool categorize: distinguish between the different groups of
            variables. Default: True.
        :parma bool opt_naming: use an optimization naming. Default: True.
        """
        dataset = Dataset(name, by_group)

        # Set the different groups
        in_grp = out_grp = dataset.DEFAULT_GROUP
        cache_output_as_input = True
        if categorize:
            if opt_naming:
                in_grp = dataset.DESIGN_GROUP
                out_grp = dataset.FUNCTION_GROUP
            else:
                in_grp = dataset.INPUT_GROUP
                out_grp = dataset.OUTPUT_GROUP
            cache_output_as_input = False

        # Add database inputs
        inputs_names = self.design_space.variables_names
        sizes = self.design_space.variables_sizes
        inputs = array(self.database.get_x_history())
        data = DataConversion.array_to_dict(inputs, inputs_names, sizes)
        for input_name, value in data.items():
            dataset.add_variable(input_name, value, in_grp)

        # Add database outputs
        all_data_names = self.database.get_all_data_names()
        outputs_names = list(
            set(all_data_names) - set(inputs_names) - set([self.database.ITER_TAG])
        )
        fct = []
        for func_name in outputs_names:
            func = self.database.get_func_history(func_name)
            fct.append(func.reshape((inputs.shape[0], -1)))
            sizes.update({func_name: fct[-1].shape[1]})
        outputs = concatenate(fct, axis=1)
        data = DataConversion.array_to_dict(outputs, outputs_names, sizes)
        for output_name, value in data.items():
            dataset.add_variable(
                output_name, value, out_grp, cache_as_input=cache_output_as_input
            )
        return dataset

    @staticmethod
    def __cast_bytes_dict(orig_dict):
        """
        Casts a dict of bytes to string values, if the values are string arrays
        also casts the output to string

        :param orig_dict : initial dict
        :returns: the casted dict
        """
        casted = {
            key: val.decode() if isinstance(val, bytes) else val
            for key, val in orig_dict.items()
        }

        for key, value in casted.items():
            if isinstance(value, ndarray) and value.dtype.type is np_string:
                if value.size == 1:
                    casted[key] = value[0]
                else:
                    casted[key] = value.tolist()
        return casted
