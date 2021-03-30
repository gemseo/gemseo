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
#        :author: Francois Gallard
#        :author: Damien Guenot
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
#        :author: Benoit Pauwels - Stacked data management
#               (e.g. iteration index)
"""
A database of function calls and design variables
*************************************************
"""

from __future__ import absolute_import, division, unicode_literals

from ast import literal_eval
from builtins import isinstance
from hashlib import sha1
from itertools import chain, islice
from xml.etree.ElementTree import parse as parse_element

import h5py
from future import standard_library
from numpy import array, atleast_2d, concatenate, float64, ndarray, string_, uint8
from numpy.linalg import norm
from six import string_types

from gemseo.utils.ggobi_export import save_data_arrays_to_xml
from gemseo.utils.py23_compat import OrderedDict  # automatically dict from py36

standard_library.install_aliases()


from gemseo import LOGGER


class Database(object):
    """Class to store evaluations of functions,
    such as DOE or optimization histories

    Avoids multiple calls of the same functions,
    useful when simulations are costly

    It is also used to store inputs and retrieve them
    for optimization graphical post processing and plots
    generation

    Can be serialized to HDF5 for portability and cold post processing


    """

    missing_value_tag = "NA"
    KEYSSEPARATOR = "__KEYSSEPARATOR__"
    GRAD_TAG = "@"
    ITER_TAG = "Iter"

    def __init__(self, input_hdf_file=None):
        """
        Constructor
        """
        self.__dict = OrderedDict()
        self.__max_iteration = 0
        # Call functions when store is called

        if input_hdf_file is not None:
            self.import_hdf(input_hdf_file)

    def __setitem__(self, key, value, dict_setitem=dict.__setitem__):
        """
        Sets an item of the dictionary

        :param key: the key of the item
        :param value: the value of the item
        :param dict_setitem: the set item default method
        """
        if not isinstance(key, (ndarray, HashableNdarray)):
            raise TypeError(
                "Optimization history keys must be" + " design variables numpy arrays"
            )
        if not isinstance(value, dict):
            raise TypeError("Optimization history values must" + " be data dictionary")
        if isinstance(key, HashableNdarray):
            self.__dict[key] = value
        else:
            self.__dict[HashableNdarray(key, True)] = value

    @staticmethod
    def __get_hashed_key(x_vect):
        """
        Gets the HashableNdarray from x_vect

        :param x_vect: the x design vector or a HashableNdarray
        :returns : the HashableNdarray of x_vect
        """
        if not isinstance(x_vect, (ndarray, HashableNdarray)):
            raise TypeError(
                "Optimization history keys must be" + " design variables numpy arrays"
            )
        if isinstance(x_vect, ndarray):
            return HashableNdarray(x_vect)
        return x_vect

    def __getitem__(self, x_vect):
        hashed = Database.__get_hashed_key(x_vect)
        return self.__dict[hashed]

    def __delitem__(self, x_vect):
        hashed = Database.__get_hashed_key(x_vect)
        del self.__dict[hashed]

    def setdefault(self, key, default):
        """
        Sets a default database entry.
        """
        if not isinstance(key, (ndarray, HashableNdarray)):
            raise TypeError(
                "Optimization history keys must be" + " design variables numpy arrays"
            )
        if not isinstance(default, dict):
            raise TypeError("Optimization history values must" + " be data dictionary")
        return self.__dict.setdefault(key, default)

    def __len__(self):
        return len(self.__dict)

    def keys(self):
        """Database keys generator."""
        for key in self.__dict.keys():
            yield key

    def values(self):
        """Database values generator."""
        for value in self.__dict.values():
            yield value

    def items(self):
        """Database items generator."""
        for key, val in self.__dict.items():
            yield key, val

    def get_value(self, x_vect):
        """Accessor for the values

        :param x_vect: the design variables

        """
        return self[x_vect]

    def get_max_iteration(self):
        """Maximum iteration number"""
        return self.__max_iteration

    def get_x_history(self):
        """Get list of x ordered by calls

        :returns: the list of x np arrays
        """
        history = []
        for x_vect in self.__dict.keys():
            history.append(x_vect.unwrap())
        return history

    def get_index_of(self, x_vect):
        """
        Returns the index of a particular x
        :param x_vect: x numpy array
        :returns: the index of x_vect, or throws a key error
        """
        hashed = HashableNdarray(x_vect)
        for i, key in enumerate(self.__dict.keys()):
            if key == hashed:
                return i
        raise KeyError(x_vect)

    def get_x_by_iter(self, iteration):
        """Return design variables at a specified iteration

        :param iteration: the iteration number
        :returns: the numpy array of x at iteration
        """
        nkeys = len(self.__dict)
        if nkeys == 0:
            raise ValueError("The database is empty!")
        if iteration < 0:
            iteration = nkeys + iteration
        if iteration >= nkeys or (iteration < 0 and -iteration > nkeys):
            raise ValueError(
                "iteration should be lower than"
                + " maximum iteration = "
                + str(len(self) - 1)
                + " got instead : "
                + str(iteration)
            )
        for i, key in enumerate(self.__dict.keys()):
            if i == iteration:
                return key.unwrap()
        return None  # pep8 requirement

    def clear(self):
        """
        Clears the database
        """
        self.__dict.clear()

    def clean_from_iterate(self, iterate):
        """
        Delete the iterates after a given iterate number

        :param iterate: the iterate number
        """

        def gen_todel():
            for iterate_number, x_vect in enumerate(self.__dict.keys()):
                # remove iterations beyond limit iterate number
                if iterate < iterate_number:
                    yield x_vect

        # Copies only the keys after iterate
        to_del = list(gen_todel())
        for key in to_del:
            del self.__dict[key]
        self.__max_iteration = len(self)

    def remove_empty_entries(self):
        """
        Removes empty entries, when x is associated to
        an empty dict
        """
        empt = [
            k
            for k, v in self.items()
            if len(v) == 0 or (len(v) == 1 and list(v.keys())[0] == self.ITER_TAG)
        ]
        for k in empt:
            del self[k]

    def filter(self, data_list_to_keep):
        """
        Keeps only the values in the data list
        :param data_list_to_keep: the list of data names to keep
        """
        data_list_to_keep = set(data_list_to_keep)
        for val in self.values():
            keys_to_del = set(val.keys()) - data_list_to_keep
            for key in keys_to_del:
                del val[key]

    def get_func_history(self, funcname, x_hist=False):
        """Return function values history.
        Can also return history of design variables

        :param funcname: the function name
        :param x_hist: if True, returns variables history as well
            (Default value = False)
        :returns: the function history list
        """
        outf_l = []
        x_history = []
        for x_vect, out_val in self.items():
            val = out_val.get(funcname)
            if val is not None:
                if isinstance(val, ndarray) and val.size == 1:
                    val = val[0]
                outf_l.append(val)
                if x_hist:
                    x_history.append(x_vect.unwrap())
        outf = array(outf_l)
        if x_hist:
            return outf, x_history

        return outf

    def get_func_grad_history(self, funcname, x_hist=False):
        """Return gradient values history
        Can also return history of design variables

        :param funcname: the function name
        :param x_hist: if True, returns variables history as well
            (Default value = False)
        :returns: the jacobian history list
        """
        return self.get_func_history(funcname=self.GRAD_TAG + funcname, x_hist=x_hist)

    def is_func_grad_history_empty(self, funcname):
        """Check if history is empty

        :param funcname: the function name
        :returns: True if history is empty
        """
        return len(self.get_func_grad_history(funcname, x_hist=False)) == 0

    def contains_x(self, x_vect):
        """Tests if history has a design variables x stored

        :param x_vect: the design variables to test
        :returns: True if x_vect is in self
        """
        return HashableNdarray(x_vect) in self.__dict

    def get_f_of_x(self, fname, x_vect, dist_tol=0.0):
        """If x in self, get associated "fname" value, if it exists

        :param fname: the function name
        :param x_vect: the design variables
        :returns: the values associated to x with name fname
        """
        if dist_tol == 0.0:
            vals = self.get(x_vect)
            if vals is not None:
                return vals.get(fname)  # Returns None if not in self
        else:
            for x_key, vals in self.items():
                x_v = x_key.unwrap()
                if norm(x_v - x_vect) <= dist_tol * norm(x_v):
                    return vals.get(fname)
        return None

    def get(self, x_vect, default=None):
        """
        Return the value for key if key is in the dictionary, else default.
        """
        if not isinstance(x_vect, (HashableNdarray, ndarray)):
            raise TypeError(
                "Optimization history keys must be" + " design variables numpy arrays"
            )
        if isinstance(x_vect, ndarray):
            x_vect = HashableNdarray(x_vect)
        return self.__dict.get(x_vect, default)

    def pop(self, k):
        """
        D.pop(k[,d]) -> v, remove specified key and return the corresponding
        value.
        If key is not found, d is returned if given, otherwise KeyError is
        raised
        """
        return self.__dict.pop(k)

    def contains_dataname(self, data_name, skip_grad=False):
        """Tests if history has a value named data_name stored

        :param data_name: the name of the data
        :param skip_grad: do not account for gradient names
        :returns: True if data_name is in self
        """
        return data_name in self.get_all_data_names(skip_grad=skip_grad)

    def store(self, x_vect, values_dict, add_iter=True):
        """Stores the values associated to the variables x

        :param x_vect: design variables vector
        :param values_dict: values to be stored
        :param add_iter: add iteration information
            (Default value = True)
        """
        if self.contains_x(x_vect):
            curr_val = self.get_value(x_vect)
            # No new keys = already computed = new iteration
            # otherwise just calls to other functions
            curr_val.update(values_dict)
        elif add_iter:
            self.__max_iteration += 1
            # include the iteration index
            new_values_dict = dict(
                values_dict, **{self.ITER_TAG: [self.__max_iteration]}
            )
            self.__setitem__(x_vect, new_values_dict)
        else:
            self.__max_iteration += 1
            # do not include the iteration index but still update it
            self.__setitem__(x_vect, values_dict)

    def get_all_data_names(self, skip_grad=True, skip_iter=False):
        """Return data variables (design, functions, gradient, ...
        Gradient variables can be skipped

        :param skip_grad: do not list gradient names (Default value = True)
        :param skip_iter: do not add Iter in the list

        :returns: the list of data names in the database
        """
        names = set()
        for value in self.__dict.values():
            for key in value.keys():
                if skip_grad and key.startswith(self.GRAD_TAG):
                    continue
                names.add(key)
        if skip_iter and self.ITER_TAG in names:
            names.remove(self.ITER_TAG)
        return sorted(names)

    def _format_history_names(self, functions, stacked_data):
        """Formats the functions names to be displayed in the history.

        :param functions: param stacked_data:
        :param stacked_data:
        """
        if functions is None:
            functions = self.get_all_data_names()
        if stacked_data is None:
            if self.ITER_TAG in functions:
                stacked_data = [self.ITER_TAG]
            else:
                stacked_data = iter([])
        elif not set(stacked_data).issubset(functions):
            raise ValueError(
                "The names of the data to be unstacked ("
                + str(stacked_data)
                + ")"
                + " must be included in the names of the data"
                + " to be returned ("
                + str(functions)
                + ")."
            )
        elif self.ITER_TAG in functions and self.ITER_TAG not in stacked_data:
            stacked_data.append(self.ITER_TAG)
        return functions, stacked_data

    def get_complete_history(
        self,
        functions=None,
        add_missing_tag=False,
        missing_tag="NA",
        all_iterations=False,
        stacked_data=None,
    ):
        """Return complete history of optimization:
        design variables, functions,
        gradients.

        :param functions: functions names to get (Default value = None)
        :param add_missing_tag: add "missing_tag" when data is not available
            for this iteration (Default value = False)
        :param missing_tag: the missing tag to add (Default value = 'NA')
        :param all_iterations: if True, points called at several
            iterations will be duplicated in the history (each duplicate
            corresponding to a different calling index); otherwise each point
            will appear only once (with the latest calling index)
            (Default value = False)
        :param stacked_data: list of names corresponding to data stored as
            lists. For example the iterations indexes are stored in a list.
            Other examples of
            stacked data may be penalization parameters or trust region radii.
            (Default value = None)
        :returns: function history and x history as lists
        """
        functions, stacked_data = self._format_history_names(functions, stacked_data)
        f_history = []
        x_history = []
        for x_vect, out_val in self.items():
            # If duplicates are not to be considered, or if no iteration index
            # is specified, then only one entry (the last one) will be written:
            if not all_iterations or self.ITER_TAG not in out_val:
                first_index = -1
                last_index = -1
            # Otherwise all the entries will be written:
            else:
                first_index = 0
                # N.B. if the list of indexes is empty, then no entry will be
                # written.
                last_index = len(out_val[self.ITER_TAG]) - 1
            # Add an element to the history for each duplicate required:
            for duplicate_ind in range(first_index, last_index + 1):
                out_vals = []
                for funcname in functions:
                    if funcname in out_val:
                        if funcname not in stacked_data:
                            out_vals.append(out_val[funcname])
                        # If the data 'funcname' is stacked and there remains
                        # entries to unstack, then unstack the next entry:
                        elif duplicate_ind < len(out_val[funcname]):
                            val = out_val[funcname][duplicate_ind]
                            out_vals.append(val)
                    elif add_missing_tag:
                        out_vals.append(missing_tag)
                if out_vals:
                    f_history.append(out_vals)
                    x_history.append(x_vect.unwrap())
        return f_history, x_history

    @staticmethod
    def __to_real(data):
        """
        Convert complex to real numpy array
        """
        return array(array(data, copy=False).real, dtype=float64)

    def export_hdf(self, file_path="optimization_history.h5", append=False):
        """Export optimization history to hdf file.

        :param file_path: path to file to write
            (Default value = 'optimization_history.h5')
        :param append: if True, appends the data in the file
            (Default value = False)
        """
        if append:
            mode = "a"
        else:
            mode = "w"
        h5file = h5py.File(file_path, mode)
        design_vars_grp = h5file.require_group("x")
        keys_group = h5file.require_group("k")
        values_group = h5file.require_group("v")
        iterated = self.items()
        i = 0
        if append and design_vars_grp:
            iterated = islice(iterated, len(design_vars_grp), len(self.__dict))
            i = len(design_vars_grp)

        for key, val in iterated:
            design_vars_grp.create_dataset(str(i), data=key.unwrap())
            keys_data = array(list(val.keys()), dtype=string_)
            locvalues_scalars = []
            argrp = None
            for ind, locval in enumerate(val.values()):
                if isinstance(locval, (ndarray, list)):
                    if argrp is None:
                        argrp = values_group.require_group("arr_" + str(i))
                    argrp.create_dataset(str(ind), data=self.__to_real(locval))
                else:
                    locvalues_scalars.append(locval)
            keys_group.create_dataset(str(i), data=keys_data)
            values_group.create_dataset(str(i), data=self.__to_real(locvalues_scalars))
            i += 1

        h5file.close()

    def import_hdf(self, filename="optimization_history.h5"):
        """Imports a database from hdf file

        :param filename: Default value = 'optimization_history.h5')
        """
        h5file = h5py.File(filename, "r")
        try:
            design_vars_grp = h5file["x"]
            keys_group = h5file["k"]
            values_group = h5file["v"]
            ndata = len(design_vars_grp)  # keys , and subdict
            for idata in range(ndata):
                x_vect = design_vars_grp[str(idata)]
                keys = keys_group[str(idata)]
                keys = [k.decode() for k in keys]
                vec_dict = {}
                if "arr_" + str(idata) in values_group:
                    argrp = values_group["arr_" + str(idata)]
                    vec_dict = {keys[int(k)]: array(v) for k, v in argrp.items()}
                locvalues_scalars = values_group[str(idata)]
                scalar_keys = (k for k in keys if k not in vec_dict)
                scalar_dict = dict(
                    ((k, v) for k, v in zip(scalar_keys, locvalues_scalars))
                )
                scalar_dict.update(vec_dict)

                self.store(array(x_vect), scalar_dict, add_iter=False)
        except KeyError as err:
            h5file.close()
            raise KeyError(
                "Invalid database hdf5 file, missing dataset. " + err.args[0]
            )

    @staticmethod
    def set_dv_names(n_dv):
        """Create a list of default design variables names

        :param n_dv: number of design variables in problem
        :returns: a list of design variables names
        """
        return ["x_" + str(i) for i in range(1, n_dv + 1)]

    def _format_design_variables_names(self, design_variables_names, dimension):
        """Formats the design variables names to be displayed in the history.

        :param design_variables_names: param dimension:
        :param dimension: number of components
        """
        if design_variables_names is None:
            design_variables_names = self.set_dv_names(dimension)
        elif isinstance(design_variables_names, string_types):
            design_variables_names = [
                design_variables_names,
            ]
        elif not isinstance(design_variables_names, list) and not isinstance(
            design_variables_names, tuple
        ):
            raise TypeError(
                "design_variables_names must be a list or a "
                + "tuple: a "
                + str(type(design_variables_names))
                + " is provided"
            )
        return design_variables_names

    def get_history_array(
        self,
        functions=None,
        design_variables_names=None,
        add_missing_tag=False,
        missing_tag="NA",
        add_dv=True,
        all_iterations=False,
        stacked_data=None,
    ):
        """Return history of optimization process

        :param functions: functions names to export (Default value = None)
        :param design_variables_names: names of the design variables
            (Default value = None)
        :param missing_tag: missing tag to add (Default value = 'NA')
        :param add_dv: if True, adds the design variables to the
            returned array (Default value = True)
        :param add_missing_tag: add "missing_tag" when data is not available
            for this iteration (Default value = False)
        :param missing_tag: the missing tag to add (Default value = 'NA')
        :param all_iterations: if True, points called at several
            iterations will be duplicated in the history (each duplicate
            corresponding to a different calling index); otherwise each point
            will appear only once (with the latest calling index)
            (Default value = False)
        :param stacked_data: list of names corresponding to data stored as
            lists. For example the iterations indexes are stored in a list.
            Other examples of
            stacked data may be penalization parameters or trust region radii.
            (Default value = None)
        :returns: function history and x history as lists
        """
        if functions is None:
            functions = self.get_all_data_names()
        f_history, x_history = self.get_complete_history(
            functions, add_missing_tag, missing_tag, all_iterations, stacked_data
        )
        design_variables_names = self._format_design_variables_names(
            design_variables_names, len(x_history[0])
        )
        flat_vals = []
        fdict = OrderedDict()
        for f_val_i in f_history:
            flat_vals_i = []
            for f_val, f_name in zip(f_val_i, functions):
                if isinstance(f_val, list):
                    f_val = array(f_val)
                if isinstance(f_val, ndarray) and len(f_val) > 1:
                    flat_vals_i = flat_vals_i + f_val.tolist()
                    fdict[f_name] = [
                        f_name + "_" + str(i + 1) for i in range(len(f_val))
                    ]
                else:
                    flat_vals_i.append(f_val)
                    if f_name not in fdict:
                        fdict[f_name] = [f_name]
            flat_vals.append(flat_vals_i)
        flat_names = sorted(list(chain(*fdict.values())))

        x_flat_vals = []
        xdict = OrderedDict()
        for x_val_i in x_history:
            x_flat_vals_i = []
            for x_val, x_name in zip(x_val_i, design_variables_names):
                if isinstance(x_val, ndarray) and len(x_val) > 1:
                    x_flat_vals_i = x_flat_vals_i + x_val.tolist()
                    xdict[x_name] = [
                        x_name + "_" + str(i + 1) for i in range(len(x_val))
                    ]
                else:
                    x_flat_vals_i.append(x_val)
                    if x_name not in xdict:
                        xdict[x_name] = [x_name]
            x_flat_vals.append(x_flat_vals_i)

        x_flat_names = list(chain(*xdict.values()))
        if add_dv:
            variables_names = flat_names + x_flat_names
        else:
            variables_names = flat_names

        f_history = array(flat_vals).real
        x_history = array(x_flat_vals).real
        if add_dv:
            f2d = atleast_2d(f_history)
            x2d = atleast_2d(x_history)
            if f2d.shape[0] == 1:
                f2d = f2d.T
            if x2d.shape[0] == 1:
                x2d = x2d.T
            values_array = concatenate((f2d, x2d), axis=1)
        else:
            values_array = f_history
        return values_array, variables_names, functions

    def export_to_ggobi(
        self, functions=None, file_path="opt_hist.xml", design_variables_names=None
    ):
        """Export history to xml file format for ggobi tool

        :param functions: Default value = None)
        :param file_path: Default value = "opt_hist.xml")
        :param design_variables_names: Default value = None)
        """
        values_array, variables_names, functions = self.get_history_array(
            functions, design_variables_names, add_missing_tag=True, missing_tag="NA"
        )
        LOGGER.info("Export to ggobi for functions: %s", str(functions))
        LOGGER.info("Export to ggobi file: %s", str(file_path))
        save_data_arrays_to_xml(
            variables_names=variables_names,
            values_array=values_array,
            file_path=file_path,
        )

    def import_from_opendace(self, database_file):
        """Reads an opendace xml database

        :param database_file: the path to the database file
        """
        tree = parse_element(database_file)
        for link in tree.getroot().iter("link"):
            data = {}
            for information in link:
                for x_ydyddy in information:
                    data[x_ydyddy.tag] = literal_eval(x_ydyddy.text)
            x_vect = array(data.pop("x"))
            data_reformat = data["y"]
            for key, value in data["dy"].items():
                data_reformat["@" + key[1:]] = array(value)
            self.store(x_vect, data_reformat)


class HashableNdarray(object):

    """HashableNdarray wrapper for ndarray objects.

    Instances of ndarray are not HashableNdarray,
    meaning they cannot be added to
    sets, nor used as keys in dictionaries. This is by design - ndarray
    objects are mutable, and therefore cannot reliably implement the
    __hash__() method.

    The HashableNdarray class allows a way around this limitation.
    It implements the required methods for HashableNdarray
    objects in terms of an encapsulated
    ndarray object. This can be either a copied instance (which is safer)
    or the original object (which requires the user to be careful enough
    not to modify it).


    """

    def __init__(self, wrapped, tight=False):
        """
        Creates a new HashableNdarray object encapsulating an ndarray.

        :param wrapped:The wrapped ndarray.
        :param tight: If True, a copy of the input ndaray is created.
        """
        self.__tight = tight
        self.wrapped = array(wrapped) if tight else wrapped
        self.__hash = int(sha1(wrapped.view(uint8)).hexdigest(), 16)

    def __eq__(self, other):
        return all(self.wrapped == other.wrapped)

    def __hash__(self):
        return self.__hash

    def unwrap(self):
        """Returns the encapsulated ndarray.
        If the wrapper is "tight", a copy of the encapsulated ndarray is

        """
        if self.__tight:
            return array(self.wrapped)

        return self.wrapped

    def __str__(self):
        return str(array(self.wrapped))

    def __repr__(self):
        return str(self)
