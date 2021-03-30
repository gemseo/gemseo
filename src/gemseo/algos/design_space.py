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
#                           documentation
#        :author: Charlie Vanaret, Benoit Pauwels, Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Design space
============

A design space is used to represent the optimization's unknowns,
a.k.a. design variables. A :class:`.DesignSpace` describes
this design space at a given state, in terms of names, sizes, types, bounds
and current values of the design variables. Variables can easily be added to
the :class:`.DesignSpace` using the :meth:`.DesignSpace.add_variable` method
or removed using the :meth:`.DesignSpace.remove_variable` method. We can also
filter the design variables using the :meth:`.DesignSpace.filter` method.
Getters and setters are also available to get or set the value of a given
variable property. Lastly, an instance of :class:`.DesignSpace` can be stored
in a txt or HDF file.
"""

from __future__ import absolute_import, division, unicode_literals

from copy import deepcopy
from os.path import exists

import h5py
from future import standard_library
from numpy import abs as np_abs
from numpy import (
    array,
    atleast_1d,
    complex128,
    concatenate,
    empty,
    equal,
    finfo,
    float64,
    genfromtxt,
    inf,
    int32,
    isnan,
    logical_or,
    mod,
    ndarray,
    ones,
    ones_like,
)
from numpy import round_ as np_round
from numpy import string_, vectorize, where, zeros_like
from six import string_types

from gemseo.algos.opt_result import OptimizationResult
from gemseo.third_party.prettytable import PrettyTable
from gemseo.utils.py23_compat import string_array, to_unicode_list

standard_library.install_aliases()


from gemseo import LOGGER


class DesignSpace(object):
    """
    Class that describes the design space at a given state:
    the names/sizes/types/bounds of the variables and the
    initial solution of the optimization problem
    """

    FLOAT = "float"
    INTEGER = "integer"
    AVAILABLE_TYPES = [FLOAT, INTEGER]
    MINIMAL_FIELDS = ["name", "lower_bound", "upper_bound"]
    TABLE_NAMES = ["name", "lower_bound", "value", "upper_bound", "type"]

    DESIGN_SPACE_GROUP = "design_space"
    NAMES_GROUP = "names"
    LB_GROUP = "l_b"
    UB_GROUP = "u_b"
    VAR_TYPE_GROUP = "var_type"
    VALUE_GROUP = "value"
    SIZE_GROUP = "size"
    # separator that denotes a vector's components
    SEP = "!"

    def __init__(self, hdf_file=None):
        """
        Constructor
        """
        self.variables_names = []
        self.dimension = 0
        self.variables_sizes = {}
        self.variables_types = {}
        self.normalize = {}
        self._lower_bounds = {}
        self._upper_bounds = {}
        # These attributes are stored for faster computation of normalization
        # and unnormalization
        self._norm_factor = None
        self.__lower_bounds_array = None
        self.__upper_bounds_array = None
        self.__int_vars_indices = None
        self.__norm_data_is_computed = False
        self.__norm_inds = None
        self.__to_zero = None
        self.__bound_tol = 100.0 * finfo(float64).eps
        self._current_x = {}
        if hdf_file is not None:
            self.import_hdf(hdf_file)

    def remove_variable(self, name):
        """Remove a variable (and bounds and types) from the design space

        :param name: name of the variable to remove
        """
        self.__norm_data_is_computed = False
        size = self.variables_sizes[name]
        self.dimension -= size
        self.variables_names.remove(name)
        del self.variables_sizes[name]
        del self.variables_types[name]
        del self.normalize[name]
        if name in self._lower_bounds:
            del self._lower_bounds[name]
        if name in self._upper_bounds:
            del self._upper_bounds[name]
        if name in self._current_x:
            del self._current_x[name]

    def filter(self, keep_variables):
        """Filters the design space to keep a sublist of variables

        :param keep_variables: the list of variables to keep
        """
        if isinstance(keep_variables, string_types):
            keep_variables = [keep_variables]
        for name in deepcopy(self.variables_names):
            if name not in keep_variables:
                self.remove_variable(name)
        for name in keep_variables:
            if name not in self.variables_names:
                raise ValueError('Variable "' + str(name) + '" is not known')
        return self

    def filter_dim(self, variable, keep_dimensions):
        """Filters the design space to keep a sublist of dimensions
        for a given variable

        :param variable: the variable
        :param keep_dimensions: the list of dimension to keep
        """
        self.__norm_data_is_computed = False
        removed_dimensions = list(
            set(range(self.variables_sizes[variable])) - set(keep_dimensions)
        )
        bad_dimensions = list(
            set(keep_dimensions) - set(range(self.variables_sizes[variable]))
        )
        size = len(removed_dimensions)
        self.dimension -= size
        self.variables_sizes[variable] -= size
        types = []
        for dimension in keep_dimensions:
            if dimension in bad_dimensions:
                self.remove_variable(variable)
                raise ValueError(
                    "Dimension "
                    + str(dimension)
                    + ' of variable "'
                    + str(variable)
                    + '" is not known'
                )
            types.append(self.variables_types[variable][dimension])
            self.variables_types[variable] = array(types)
        idx = keep_dimensions
        self.normalize[variable] = self.normalize[variable][idx]
        if variable in self._lower_bounds:
            self._lower_bounds[variable] = self._lower_bounds[variable][idx]
        if variable in self._upper_bounds:
            self._upper_bounds[variable] = self._upper_bounds[variable][idx]
        if variable in self._current_x:
            self._current_x[variable] = self._current_x[variable][idx]
        return self

    def add_variable(
        self, name, size=1, var_type=FLOAT, l_b=None, u_b=None, value=None
    ):
        """Add a variable to the design space

        :param name: param size:  (Default value = 1)
        :param var_type: Default value = FLOAT)
        :param l_b: Default value = None)
        :param u_b: Default value = None)
        :param value: Default value = None)
        :param size:  (Default value = 1)
        """
        if name in self.variables_names:
            raise ValueError("Variable " + name + " already exists")
        if size <= 0 or int(size) != size:
            raise ValueError("The size of " + name + " should be a positive integer.")

        # name and size
        self.variables_names.append(name)
        self.dimension += size
        self.variables_sizes[name] = size
        # type
        self._add_type(name, size, var_type)

        # bounds
        self._add_bound(name, size, l_b, is_lower=True)
        self._add_bound(name, size, u_b, is_lower=False)
        self._check_variable_bounds(name)

        # normalization policy
        self._add_norm_policy(name)

        if value is not None:
            array_value = atleast_1d(value)
            self._check_value(array_value, name)
            if len(array_value) == 1 and size > 1:
                array_value = array_value * ones(size)
            self._current_x[name] = array_value
            self._check_current_x_value(name)

    def _add_type(self, name, size, var_type=None):
        """Add a type to a variable

        :param name: name of the variable
        :param size: size of the variable
        :param var_type: type in self.AVAILABLE_TYPES,
            or None, which is then FLOAT by default
        """
        if isinstance(var_type, bytes):
            var_type = var_type.decode()
        if var_type is None:
            var_type = self.FLOAT
        if hasattr(var_type, "__iter__") and not isinstance(var_type, string_types):
            if len(var_type) != size:
                raise ValueError(
                    "The list of types for  variable "
                    + name
                    + " should be of size "
                    + str(size)
                )
            # a type for each component
            var_type = [
                v_type.decode() if isinstance(v_type, bytes) else v_type
                for v_type in var_type
            ]
            for v_type in var_type:
                if v_type not in self.AVAILABLE_TYPES:
                    msg = 'The type "{0}" of {1} is not known'.format(v_type, name)
                    raise ValueError(msg)
            self.variables_types[name] = array(var_type)

        else:
            # same type for all components
            if var_type not in self.AVAILABLE_TYPES:
                raise ValueError('Type "' + str(var_type) + '" is not known')
            var_type_array = array([var_type] * size)
            self.variables_types[name] = var_type_array

        self.__norm_data_is_computed = False

    def _add_norm_policy(self, name):
        """Adds a normalization policy to a variable.
        Unbounded variables are not normalized.
        Bounded variables (both from above and from below) are normalized.

        :param name: variable name
        """
        # Check that the variable is in the design space:
        if name not in self.variables_names:
            raise ValueError("Variable " + name + " is not in the design " + "space.")
        # Check that the variable size is set:
        size = self.get_size(name)
        if size is None:
            raise ValueError("The size of variable " + name + " is not set.")
        # Check that the variables types are set:
        variables_types = self.variables_types.get(name, None)
        if variables_types is None:
            raise ValueError(
                "The components types of variable " + name + " are not set."
            )
        # Set the normalization policy:
        normalize = empty(size)
        for i in range(size):
            var_type = variables_types[i]
            if var_type in (DesignSpace.INTEGER, DesignSpace.FLOAT):
                if (
                    self._lower_bounds[name][i] == -inf
                    or self._upper_bounds[name][i] == inf
                ):
                    # Unbounded variables are not normalized:
                    normalize[i] = False
                elif self._lower_bounds[name][i] == self._upper_bounds[name][i]:
                    # Constant variables are not normalized:
                    normalize[i] = False
                else:
                    normalize[i] = True
            else:
                msg = "The normalization policy for type {0}" + " is not implemented."
                raise ValueError(msg.format(var_type))
        self.normalize[name] = normalize

    @staticmethod
    def __is_integer(value):
        """  Checks that all values are integers  """
        are_none = equal(value, None)
        are_int = equal(mod(value.astype("f"), 1), 0)
        return logical_or(are_none, are_int)

    @staticmethod
    def __is_numeric(value):
        """ Checks that a value is numeric """
        res = (value is None) or hasattr(value, "real")
        try:
            if not res:
                float(value)
            return True
        except TypeError:
            return False

    @staticmethod
    def __isnot_nan(value):
        """ Checks that a value is not nan"""
        return (value is None) or ~isnan(value)

    def _check_value(self, value, name):
        """
        Checks that a variable value is valid

        :param value: a numpy array
        """
        all_indices = set(range(len(value)))
        # OK if the variable value is one-dimensional
        if len(value.shape) > 1:
            raise ValueError(
                "Value "
                + str(value)
                + " of variable "
                + str(name)
                + " has dimension greater than 1"
                + "while a float or a 1d iterable object "
                + "(array, list, tuple, ...) "
                + "while a scalar was expected."
            )

        # OK if all components are None
        if all(equal(value, None)):
            return True

        test = vectorize(self.__is_numeric)(value)
        indices = all_indices - set(list(where(test)[0]))
        for idx in indices:
            raise ValueError(
                "Value "
                + str(value[idx])
                + " of variable "
                + str(name)
                + " is not numerizable."
            )

        test = vectorize(self.__isnot_nan)(value)
        indices = all_indices - set(list(where(test)[0]))
        for idx in indices:
            raise ValueError(
                "Value " + str(value[idx]) + " of variable " + str(name) + " is nan."
            )

        # Check if some components of an integer variable are not integer.
        if self.variables_types[name][0] == self.INTEGER:
            indices = all_indices - set(list(where(self.__is_integer(value))[0]))
            for idx in indices:
                raise ValueError(
                    "Component value "
                    + str(value[idx])
                    + " of variable "
                    + str(name)
                    + " is not an integer "
                    + "while variable is of type integer"
                    + "(index: "
                    + str(idx)
                    + ")."
                )

    def _add_bound(self, name, size, bound, is_lower=True):
        """Add a lower or upper bound to a variable

        :param name: name of the variable
        :param bound: lower or upper bound (array)
        :param size: size of the variable
        :param is_lower: if True, bound is a lower bound
        """
        self.__norm_data_is_computed = False
        if is_lower:
            bound_dict = self._lower_bounds
        else:
            bound_dict = self._upper_bounds
        infinity = inf * ones(size)
        if bound is None:
            if is_lower:
                bound_array = -infinity
            else:
                bound_array = infinity
            bound_dict.update({name: bound_array})
            return
        bound_array = atleast_1d(bound)
        self._check_value(bound_array, name)
        if hasattr(bound, "__iter__"):
            # iterable structure
            if len(bound_array) != size:
                bound_str = "lower" if is_lower else "upper"
                raise ValueError(
                    "The "
                    + bound_str
                    + " bounds of "
                    + name
                    + " should be of size "
                    + str(size)
                )
            if is_lower:
                bound_array = where(equal(bound_array, None), -infinity, bound_array)
            else:
                bound_array = where(equal(bound_array, None), infinity, bound_array)
            bound_dict.update({name: bound_array})
        else:
            # scalar: same lower bound for all components
            bound_array = bound * ones(size)
            bound_dict.update({name: bound_array})
        return

    def _check_variable_bounds(self, name):
        """Check that the bounds are compatible and are the same size

        :param name: name of the variable
        """
        l_b = self._lower_bounds.get(name, None)
        u_b = self._upper_bounds.get(name, None)
        inds = where(u_b < l_b)[0]
        if inds.size != 0:
            raise ValueError(
                "The bounds of variable "
                + name
                + str(inds)
                + " are not valid : "
                + str(l_b[inds])
                + "!<"
                + str(u_b[inds])
            )

    def _check_current_x_value(self, name):
        """Check that the current x values are between bounds

        :param name: name of the variable
        """
        l_b = self._lower_bounds.get(name, None)
        u_b = self._upper_bounds.get(name, None)
        c_x = self._current_x.get(name, None)
        not_none = ~equal(c_x, None)
        inds = where(
            logical_or(
                c_x[not_none] < l_b[not_none] - 1e-14,
                c_x[not_none] > u_b[not_none] + 1e-14,
            )
        )[0]
        for idx in inds:
            raise ValueError(
                "The current value of variable "
                + name
                + "!"
                + str(c_x[idx])
                + " is not between "
                + "the lower bound "
                + str(l_b[idx])
                + " and "
                + "the upper bound "
                + str(u_b[idx])
            )

    def has_current_x(self):
        """
        Tests if current_x is defined

        :returns: True if current_x is defined
        """
        if self._current_x is None or len(self._current_x) != len(self.variables_names):
            return False
        for val in self._current_x.values():
            if val is not None:
                return True
        return False

    def check(self):
        """Check the state of the design space"""
        if not self.variables_names:
            raise ValueError("Design space is empty !")

        for name in self.variables_names:
            self._check_variable_bounds(name)

        if self.has_current_x():
            self._check_current_x()

    def check_membership(self, x_vect, variables_names=None):
        """Checks whether the input variables satisfy the design space
        requirements.

        :param x_vect: design variables
        :type x_vect: dict or array
        :param variables_names: names of the variables to be checked

        """
        # Convert the input vector into a dictionary if necessary:
        if isinstance(x_vect, dict):
            x_dict = x_vect
        elif isinstance(x_vect, ndarray):
            if x_vect.size != self.dimension:
                raise ValueError(
                    "The dimension of the input array ("
                    + str(x_vect.size)
                    + ") should be "
                    + str(self.dimension)
                    + "."
                )
            x_dict = self.array_to_dict(x_vect)
        else:
            raise TypeError(
                "The input vector should be an array or a "
                + "dictionary. Got "
                + str(type(x_vect))
                + " instead."
            )
        # Check the membership of the input vector to the design space:
        variables_names = variables_names or self.variables_names
        for name in variables_names:
            if x_dict[name] is None:
                continue
            size = self.variables_sizes[name]
            l_b = self._lower_bounds.get(name, None)
            u_b = self._upper_bounds.get(name, None)

            if atleast_1d(x_dict[name]).size != size:
                raise ValueError(
                    "The component "
                    + name
                    + " of the given"
                    + " array should have size "
                    + str(size)
                    + "."
                )
            for i in range(size):
                x_real = atleast_1d(x_dict[name])[i].real
                if l_b is not None and x_real < l_b[i] - self.__bound_tol:
                    msg = "The component " + name + self.SEP + str(i)
                    msg += " of the given array (" + str(x_real) + ") is "
                    msg += "lower than the lower bound (" + str(l_b[i]) + ")"
                    msg += " by {:.1e}.".format(l_b[i] - x_real)
                    raise ValueError(msg)
                if u_b is not None and u_b[i] + self.__bound_tol < x_real:
                    msg = "The component " + name + self.SEP + str(i)
                    msg += " of the given array (" + str(x_real) + ") is "
                    msg += "greater than the upper bound ("
                    msg += str(u_b[i]) + ")"
                    msg += " by {:.1e}.".format(x_real - u_b[i])
                    raise ValueError(msg)
                if (
                    self.variables_types[name][0] == self.INTEGER
                ) and not self.__is_integer(x_real):
                    msg = "The component " + name + self.SEP + str(i)
                    msg += " of the given array is not an integer "
                    msg += " while variable is of type integer !"
                    msg += " value = " + str(x_real)
                    raise ValueError(msg)

    def get_active_bounds(self, x_vec=None, tol=1e-8):
        """Determine which bound constraints of the current point are active

        :param x_vec: the point at which we check the bounds
        :param tol: tolerance of comparison of a scalar with a bound
            (Default value = 1e-8)
        """
        if x_vec is None:
            x_dict = self._current_x
            self.check_membership(self.get_current_x())
        elif isinstance(x_vec, ndarray):
            x_dict = self.array_to_dict(x_vec)
        elif isinstance(x_vec, dict):
            x_dict = x_vec
        else:
            raise TypeError(
                "Expected dict or array for x_vec argument,"
                + " got "
                + str(type(x_vec))
            )

        active_l_b = {}
        active_u_b = {}
        for name in self.variables_names:
            l_b = self._lower_bounds.get(name)
            l_b = where(equal(l_b, None), -inf, l_b)
            u_b = self._upper_bounds.get(name)
            u_b = where(equal(u_b, None), inf, u_b)
            x_vec_i = x_dict[name]
            # lower bound saturated
            albi = where(np_abs(x_vec_i - l_b) <= tol, True, False)
            active_l_b[name] = albi
            # upper bound saturated
            aubi = where(np_abs(x_vec_i - u_b) <= tol, True, False)
            active_u_b[name] = aubi
        return active_l_b, active_u_b

    def _check_current_x(self, variables_names=None):
        """Checks the current point.

        :param variables_names: Default value = None)
        """
        if sorted(set(self.variables_names)) != sorted(set(self._current_x.keys())):
            raise ValueError(
                "Expected current_x variables :"
                + str(self.variables_names)
                + ", got "
                + str(list(self._current_x.keys()))
            )
        self.check_membership(self._current_x, variables_names)

    def get_current_x(self, variables_names=None):
        """Gets the current point in the design space.

        :param variables_names: names of the required variables, optional
        :type variables_names: list(str)
        :returns: the x vector as array
        :rtype: ndarray
        """
        try:
            x_arr = self.dict_to_array(self._current_x, all_var_list=variables_names)
            return x_arr
        except KeyError as err:
            raise KeyError("DesignSpace has no current_x for " + err.args[0])

    def get_indexed_var_name(self, variable_name):
        """
        Retuns a list of the variables names with their indices
        such as [x!0,x!1,y,z!0,z!1]
        """
        size = self.variables_sizes[variable_name]
        if size == 1:
            return variable_name
        return [variable_name + self.SEP + str(i) for i in range(size)]

    def get_indexed_variables_names(self):
        """
        Retuns a list of the variables names with their indices
        such as [x!0,x!1,y,z!0,z!1]
        """
        var_ind_names = []
        for var in self.variables_names:
            vnames = self.get_indexed_var_name(var)
            if isinstance(vnames, string_types):
                var_ind_names.append(vnames)
            else:
                var_ind_names += vnames
        return var_ind_names

    def __update_normalization_vars(self):
        """
        Computes inner attributes used to compute
        the normalization/unnormalization
        """

        self.__lower_bounds_array = self.get_lower_bounds()
        self.__upper_bounds_array = self.get_upper_bounds()
        self._norm_factor = self.__upper_bounds_array - self.__lower_bounds_array
        norm_array = self.dict_to_array(self.normalize)
        self.__norm_inds = where(norm_array)[0]
        # In case lb=ub
        self.__to_zero = where(self._norm_factor == 0.0)[0]

        var_ind_list = []
        for var in self.variables_names:
            # Store the mask of int variables
            to_one = self.variables_types[var] == self.INTEGER
            var_ind_list.append(to_one)
        self.__int_vars_indices = concatenate(var_ind_list)
        self.__norm_data_is_computed = True

    def normalize_vect(self, x_vect, minus_lb=True):
        """Normalizes a vector of the design space.
        Unbounded variables are not normalized.

        :param x_vect: design variables
        :type x_vect: ndarray
        :param minus_lb: if True, remove lower bounds at normalization
        :return: normalized vector
        """
        if not self.__norm_data_is_computed:
            self.__update_normalization_vars()
        # Normalize the relevant components:
        norm_vect = array(x_vect, copy=True)
        # In case lb=ub
        norm_inds = self.__norm_inds

        if len(x_vect.shape) == 1:
            if minus_lb:
                norm_vect[norm_inds] -= self.__lower_bounds_array[norm_inds]
            norm_vect[norm_inds] /= self._norm_factor[norm_inds]
            # In case lb=ub put value to 0
        elif len(x_vect.shape) == 2:
            if minus_lb:
                norm_vect[:, norm_inds] -= self.__lower_bounds_array[norm_inds]
            norm_vect[:, norm_inds] /= self._norm_factor[norm_inds]
        else:
            raise ValueError("The array to be normalized must be 1d or 2d.")

        to_zero = self.__to_zero
        if to_zero.size > 0:
            norm_vect[to_zero] = 0.0
        return norm_vect

    def unnormalize_vect(self, x_vect, minus_lb=True, no_check=False):
        """Unnormalizes a normalized vector of the design space.

        :param x_vect: design variables
        :type x_vect: ndarray
        :param minus_lb: if True, remove lower bounds at normalization
        :param no_check: if True, dont check that values are in [0,1]
        :return: normalized vector
        """
        if not self.__norm_data_is_computed:
            self.__update_normalization_vars()
        norm_inds = self.__norm_inds
        l_bounds = self.__lower_bounds_array
        if not no_check:
            # Get the indexes of the components to be unnormalized:
            # Check whether the input components are between 0 and 1:
            if (x_vect < -1e-12).any() or (x_vect > 1 + 1e-12).any():
                msg = "All components of the "
                msg += "normalized vector should be between 0 and 1."
                lb_viol = x_vect[x_vect < -1e-12]
                if lb_viol.size != 0:
                    msg += " lower bounds violated : " + str(lb_viol)
                ub_viol = x_vect[x_vect > 1 + 1e-12]
                if ub_viol.size != 0:
                    msg += " upper bounds violated : " + str(ub_viol)

                LOGGER.warning(msg)
        # Unnormalize the relevant components:
        unnorm_vect = array(x_vect, copy=True)

        n_dims = len(x_vect.shape)
        if n_dims == 1:
            unnorm_vect[norm_inds] *= self._norm_factor[norm_inds]
            if minus_lb:
                unnorm_vect[norm_inds] += l_bounds[norm_inds]
        elif n_dims == 2:
            unnorm_vect[:, norm_inds] *= self._norm_factor[norm_inds]
            if minus_lb:
                unnorm_vect[:, norm_inds] += l_bounds[norm_inds]
        else:
            raise ValueError(
                "The array to be unnormalized must be 1d or 2d"
                + ", got "
                + str(n_dims)
                + "d !"
            )
        inds_fixed = self.__to_zero
        if inds_fixed.size > 0:
            if n_dims == 1:
                unnorm_vect[inds_fixed] = l_bounds[inds_fixed]
            else:
                unnorm_vect[:, inds_fixed] = l_bounds[inds_fixed]

        r_xvec = self.round_vect(unnorm_vect)
        return r_xvec

    def round_vect(self, x_vect):
        """
        Rounds the vector where variables are of integer type

        :param x_vect: design variables to round
        """
        if not self.__norm_data_is_computed:
            self.__update_normalization_vars()
        int_vars = self.__int_vars_indices
        if not int_vars.any():
            return x_vect
        n_dims = len(x_vect.shape)
        rounded_x_vect = x_vect.copy()
        if n_dims == 1:
            rounded_x_vect[int_vars] = np_round(rounded_x_vect[int_vars])
        elif n_dims == 2:
            rounded_x_vect[:, int_vars] = np_round(rounded_x_vect[:, int_vars])
        else:
            raise ValueError(
                "The array to be unnormalized must be 1d or 2d"
                + ", got "
                + str(n_dims)
                + "d !"
            )

        return rounded_x_vect

    def get_current_x_normalized(self):
        """Returns the current point normalized.

        :returns: the x vector as array normalized by the bounds
        """
        try:
            current_x = self.get_current_x()
        except KeyError as err:
            raise KeyError("Cannot compute normalized current x since " + err.args[0])
        return self.normalize_vect(current_x)

    def get_current_x_dict(self):
        """Get the current point in the design space

        :returns: the x vector as a dict, keys are the variable names
            values are the variable vales as np array
        """
        return self._current_x

    def set_current_x(self, current_x):
        """Set the current point

        :param current_x: the current design vector
        """
        if isinstance(current_x, dict):
            self._current_x = current_x
        elif isinstance(current_x, ndarray):
            if current_x.size != self.dimension:
                raise ValueError(
                    "Invalid current_x, dimension mismatch "
                    + str(self.dimension)
                    + " != "
                    + str(current_x.size)
                )
            self._current_x = self.array_to_dict(current_x)
        elif isinstance(current_x, OptimizationResult):
            if current_x.x_opt.size != self.dimension:
                raise ValueError(
                    "Invalid x_opt, dimension mismatch "
                    + str(self.dimension)
                    + " != "
                    + str(current_x.x_opt.size)
                )
            x_array = self.array_to_dict(current_x.x_opt)
            self._current_x = x_array
        else:
            raise TypeError(
                "The current should be an array or a dict. "
                + "Got "
                + str(type(current_x))
                + " instead."
            )
        self._check_current_x()

    def set_current_variable(self, name, current_value):
        """Set the current value of a single variable

        :param name: name of the variable
        :param current_value: current value of the variable
        """
        if name in self.variables_names:
            self._current_x[name] = current_value
        else:
            raise ValueError("Variable " + str(name) + " is not in the design space !")

    def get_size(self, name):
        """Get the size of a variable
        Return None if the variable is not known

        :param name: name of the variable
        """
        return self.variables_sizes.get(name, None)

    def get_type(self, name):
        """Get the type of a variable
        Return None if the variable is not known

        :param name: name of the variable
        """
        return self.variables_types.get(name, None)

    def get_lower_bound(self, name):
        """Gets the lower bound of a variable.

        :param name: variable name
        :returns: variable lower bound (possibly infinite)

        """
        return self._lower_bounds.get(name)

    def get_upper_bound(self, name):
        """Gets the upper bound of a variable.

        :param name: variable name
        :returns: variable upper bound (possibly infinite)
        """
        return self._upper_bounds.get(name)

    def get_lower_bounds(self, variables_names=None):
        """Generates an array of the variables' lower bounds.

        :param variables_names: names of the variables of which the lower
            bounds are required
        """
        if self.__norm_data_is_computed and variables_names is None:
            return self.__lower_bounds_array
        return self.dict_to_array(self._lower_bounds, all_var_list=variables_names)

    def get_upper_bounds(self, variables_names=None):
        """Generates an array of the variables' upper bounds.

        :param variables_names: names of the variables of which the upper
            bounds are required
        """
        if self.__norm_data_is_computed and variables_names is None:
            return self.__upper_bounds_array
        return self.dict_to_array(self._upper_bounds, all_var_list=variables_names)

    def set_lower_bound(self, name, lower_bound):
        """Set a new lower bound for variable name

        :param name: name of the variable
        :param lower_bound: lower bound
        """
        if name not in self.variables_names:
            raise ValueError("Variable " + name + " does not exist")

        self._add_bound(name, self.variables_sizes[name], lower_bound, is_lower=True)
        self._add_norm_policy(name)

    def set_upper_bound(self, name, upper_bound):
        """Set a new upper bound for variable name

        :param name: name of the variable
        :param upper_bound: upper bound
        """
        if name not in self.variables_names:
            raise ValueError("Variable " + name + " does not exist")

        self._add_bound(name, self.variables_sizes[name], upper_bound, is_lower=False)
        self._add_norm_policy(name)

    def array_to_dict(self, x_array):
        """Split the current point into a dictionary with variables names

        :param x_array: x array to be converted to a dict of array
        """
        x_dict = {}
        current_index = 0
        # order given by self.variables_names
        for name in self.variables_names:
            size = self.variables_sizes[name]
            x_dict[name] = x_array[current_index : current_index + size]
            current_index += size
        return x_dict

    @staticmethod
    def __get_common_dtype(x_dict):
        """
        If dict has a value complex array, returns numpy.complex128
        if dict has real values and mixed floats/int, returns numpy.float64
        if dict has only int values, returns numpy.int32

        :param x_dict : dictionary of variables
        """
        has_float = False
        has_int = False
        for val_arr in x_dict.values():
            if not isinstance(val_arr, ndarray):
                raise TypeError("x_dict values must be ndarray")
            if val_arr.dtype == complex128:
                return complex128
            if val_arr.dtype == int32:
                has_int = True
            if val_arr.dtype == float64:
                has_float = True
        if has_float:
            return float64
        if has_int:
            return int32
        return float64

    def dict_to_array(self, x_dict, all_vars=True, all_var_list=None):
        """Aggregate a point as dictionary into array

        :param x_dict: point as dictionary
        :param all_vars: if True, all variables shall be in x_dict
        :param all_var_list: list of whole set of variables,
            if None, use self.variables_names
        """
        dtype = self.__get_common_dtype(x_dict)
        if all_var_list is None:
            all_var_list = self.variables_names
        if all_vars:
            array_list = [array(x_dict[name], dtype=dtype) for name in all_var_list]
        else:
            array_list = [
                array(x_dict[name], dtype=dtype)
                for name in all_var_list
                if name in x_dict
            ]
        return concatenate(array_list)

    def get_pretty_table(self, fields=None):
        """Builds a PrettyTable object from the design space data

        :param fields: list of fields to export, by default all
        :returns:  the pretty table object
        """
        if fields is None:
            fields = self.TABLE_NAMES
        table = PrettyTable(fields)
        table.float_format = "%.16g"
        for name in self.variables_names:
            size = self.variables_sizes[name]
            l_b = self._lower_bounds.get(name)
            u_b = self._upper_bounds.get(name)
            var_type = self.variables_types[name]
            curr = self._current_x.get(name)
            for i in range(size):
                data = {
                    "name": name,
                    "value": None,
                    "lower_bound": float("-inf"),
                    "upper_bound": float("inf"),
                    "type": var_type[i],
                }
                if l_b is not None and l_b[i] is not None:
                    data["lower_bound"] = l_b[i]
                if u_b is not None and u_b[i] is not None:
                    data["upper_bound"] = u_b[i]
                if curr is not None:
                    data["value"] = curr[i]
                table.add_row([data[key] for key in fields])

        table.align["name"] = "l"
        table.align["type"] = "l"
        return table

    def export_hdf(self, file_path, append=False):
        """Export to hdf file.

        :param file_path: path to file to write
        :param append: if True, appends the data in the file
        """
        if append:
            mode = "a"
        else:
            mode = "w"
        h5file = h5py.File(file_path, mode)
        design_vars_grp = h5file.require_group(self.DESIGN_SPACE_GROUP)
        design_vars_grp.create_dataset(
            self.NAMES_GROUP, data=array(self.variables_names, dtype=string_)
        )
        for name in self.variables_names:
            var_grp = design_vars_grp.require_group(name)
            var_grp.create_dataset(self.SIZE_GROUP, data=self.variables_sizes[name])
            l_b = self._lower_bounds.get(name)
            if l_b is not None:
                var_grp.create_dataset(self.LB_GROUP, data=l_b)
            u_b = self._upper_bounds.get(name)
            if u_b is not None:
                var_grp.create_dataset(self.UB_GROUP, data=u_b)
            var_type = self.variables_types[name]
            if var_type is not None:
                data_array = string_array(var_type)
                dtype = data_array.dtype
                var_grp.create_dataset(
                    self.VAR_TYPE_GROUP, data=data_array, dtype=dtype
                )
            value = self._current_x.get(name)
            if value is not None:
                var_grp.create_dataset(self.VALUE_GROUP, data=self.__to_real(value))
        h5file.close()

    def import_hdf(self, file_path):
        """Imports design space from hdf file

        :param file_path:
        """
        if not exists(file_path):
            raise ValueError("Input hdf file does not exist ! " + str(file_path))

        h5file = h5py.File(file_path, "r")
        try:
            design_vars_grp = h5file[self.DESIGN_SPACE_GROUP]
            variables_names = design_vars_grp[
                self.NAMES_GROUP
            ].value  # pylint: disable=E1101
            for name in variables_names:
                name = name.decode()
                var_group = design_vars_grp[name]
                l_b = self.__read_opt_attr_array(var_group, self.LB_GROUP)
                u_b = self.__read_opt_attr_array(var_group, self.UB_GROUP)
                var_type = self.__read_opt_attr_array(var_group, self.VAR_TYPE_GROUP)
                value = self.__read_opt_attr_array(var_group, self.VALUE_GROUP)
                size = var_group[self.SIZE_GROUP].value
                self.add_variable(name, size, var_type, l_b, u_b, value)
        except KeyError as err:
            h5file.close()
            raise KeyError(
                "Invalid design space hdf5 file "
                + str(file_path)
                + " missing dataset. "
                + str(err.args[0])
            )
        h5file.close()
        self.check()

    @staticmethod
    def __read_opt_attr_array(var_group, dataset_name):
        """
        Reads an array in a group, can be optional
        If data does not exists, returns None

        :param var_group : the variable group
        :param dataset_name : name of the data
        :returns: the data as np array, or None
        """
        inval = var_group.get(dataset_name)
        if inval is not None:
            inval = array(inval)
        return inval

    @staticmethod
    def __to_real(data):
        """   Convert complex to real numpy array        """
        return array(array(data, copy=False).real, dtype=float64)

    def to_complex(self):
        """  Casts the current value to complex  """
        for name, val in self._current_x.items():
            self._current_x[name] = array(val, dtype=complex128)

    def export_to_txt(self, output_file, fields=None, header_char="", **table_options):
        """Exports the design space to a text file

        :param output_file: output file path
        :param fields: list of fields to export, by default all
        """
        table = self.get_pretty_table(fields)
        table.border = False
        for option, val in table_options.items():
            table.__setattr__(option, val)
        with open(output_file, "w") as outf:
            table_str = header_char + table.get_string()
            outf.write(table_str)

    @staticmethod
    def read_from_txt(input_file, header=None):
        """Parses a csv file to read the DesignSpace

        :param input_file: returns: s: the design space
        :param header: fields list, or by default, read in the file
        :returns:  the design space
        """
        float_data = genfromtxt(input_file, dtype="float")
        str_data = genfromtxt(input_file, dtype="str")
        if header is None:
            header = to_unicode_list(str_data[0, :].tolist())
            start_read = 1
        else:
            start_read = 0
        if not set(DesignSpace.MINIMAL_FIELDS).issubset(set(header)):
            raise ValueError(
                "Malformed DesignSpace input file "
                + str(input_file)
                + " does not contain minimal "
                + "variables in header :"
                + str(DesignSpace.MINIMAL_FIELDS)
                + ", got instead : "
                + str(header)
            )
        col_map = {field: i for i, field in enumerate(header)}
        var_names = to_unicode_list(str_data[start_read:, 0].tolist())
        unique_names = []
        prev_name = None
        for name in var_names:  # set([]) does not preserve order !
            if name not in unique_names:
                unique_names.append(name)
                prev_name = name
            elif prev_name != name:
                raise ValueError(
                    "Malformed DesignSpace input file "
                    + str(input_file)
                    + " contains some variables ("
                    + name
                    + ") in a non-consecutive order "
                )

        k = start_read
        design_space = DesignSpace()
        lower_bounds_field = DesignSpace.MINIMAL_FIELDS[1]
        upper_bounds_field = DesignSpace.MINIMAL_FIELDS[2]
        value_field = DesignSpace.TABLE_NAMES[2]
        var_type_field = DesignSpace.TABLE_NAMES[-1]
        for name in unique_names:
            size = var_names.count(name)
            l_b = float_data[k : k + size, col_map[lower_bounds_field]]
            u_b = float_data[k : k + size, col_map[upper_bounds_field]]
            if value_field in col_map:
                value = float_data[k : k + size, col_map[value_field]]
            else:
                value = None
            if var_type_field in col_map:
                var_type = str_data[k : k + size, col_map[var_type_field]].tolist()
            else:
                var_type = None
            design_space.add_variable(name, size, var_type, l_b, u_b, value)
            k += size
        design_space.check()
        return design_space

    def log_me(self):
        """Logs a representation of the design_space characteristics
        as a table

        """
        msg = str(self)
        for line in msg.split("\n"):
            LOGGER.info(line)

    def __str__(self, *args, **kwargs):
        desc = "Design Space: "
        desc += "\n" + str(self.get_pretty_table().get_string())
        return desc

    def project_into_bounds(self, x_c, normalized=False):
        """
        Projects x_c onto the bounds, using a simple
        coordinate wise approach

        :param x_c: x vector (np array)
        :returns: projected x_c
        """
        if not self.__norm_data_is_computed:
            self.__update_normalization_vars()
        if not normalized:
            l_b = self.__lower_bounds_array
            u_b = self.__upper_bounds_array
        else:
            l_b = zeros_like(x_c)
            u_b = ones_like(x_c)
        x_p = array(x_c)
        l_inds = where(x_c < l_b)
        x_p[l_inds] = l_b[l_inds]

        u_inds = where(x_c > u_b)
        x_p[u_inds] = u_b[u_inds]
        return x_p
