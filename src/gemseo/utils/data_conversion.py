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
#                         documentation
#        :author: Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Convert arrays/dict of arrays data
**********************************
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from builtins import str
from copy import deepcopy
from functools import reduce as f_reduce

from future import standard_library
from numpy import array, hstack, ndarray, zeros

standard_library.install_aliases()


class DataConversion(object):
    """This class contains static methods that convert:
    - data dict into numpy array
    - numpy array into data dict by updating an existing dict
    """

    FLAT_JAC_SEP = "!d$_$d!"

    @staticmethod
    def dict_to_array(data_dict, data_names):
        """Convert a dict into an array

        :param data_dict: the data dictionary
        :param data_names: the data keys to concatenate
        :returns: an array
        :rtype: ndarray
        """
        if not data_names:
            return array([])
        values_list = [data_dict[name] for name in data_names]
        return hstack(values_list)

    @staticmethod
    def list_of_dict_to_array(data_list, data_names, group=None):
        """Convert a list of dict into an array

        :param data_list: the data list
        :param data_names: the data keys to concatenate
        :param group: groupe of keys to concatenate
        :returns: an array
        :rtype: ndarray
        """
        dict_to_a = DataConversion.dict_to_array
        if group is not None:
            out_array = array(
                [
                    dict_to_a(data_list[element][group], data_names)
                    for element, _ in enumerate(data_list)
                ]
            )
        else:
            out_array = array(
                [
                    dict_to_a(data_list[element], data_names)
                    for element, _ in enumerate(data_list)
                ]
            )
        return out_array

    @staticmethod
    def array_to_dict(data_array, data_names, data_sizes):
        """Convert an array into a dict

        :param data_array: the array
        :param data_names: list of names (keys of the resulting dict)
        :param data_sizes: dict of (name, size)
        :returns: a dict
        :rtype: dict
        """
        current_position = 0
        array_dict = {}
        if len(data_array.shape) == 2:
            for name in data_names:
                array_dict[name] = data_array[
                    :, current_position : current_position + data_sizes[name]
                ]
                current_position += data_sizes[name]
        elif len(data_array.shape) == 1:
            for name in data_names:
                array_dict[name] = data_array[
                    current_position : current_position + data_sizes[name]
                ]
                current_position += data_sizes[name]
        else:
            raise ValueError("Invalid data dimension >2 !")
        return array_dict

    @staticmethod
    def jac_2dmat_to_dict(flat_jac, outputs, inputs, data_sizes):
        """
        Converts a full 2D jacobian to a dict of dict sparse format

        :param flat_jac : the flat 2D jacobian
        :param inputs: derive outputs wrt inputs
        :param outputs: outputs to be derived
        :param data_sizes: dict of (name, size) for names in inputs and outputs
        """
        curr_out = 0
        jac_dict = {}
        for out in outputs:
            jac_dict[out] = {}
            out_size = data_sizes[out]
            curr_in = 0
            for inpt in inputs:
                inpt_size = data_sizes[inpt]
                jac_dict[out][inpt] = flat_jac[
                    curr_out : curr_out + out_size, curr_in : curr_in + inpt_size
                ]
                curr_in += inpt_size
            curr_out += out_size
        return jac_dict

    @staticmethod
    def jac_3dmat_to_dict(jac, outputs, inputs, data_sizes):
        """Converts a 3D jacobian (list of 2D jacobians) to a dict of dict
        sparse format.

        :param jac : list of 2D jacobians
        :param inputs: derive outputs wrt inputs
        :param outputs: outputs to be derived
        :param data_sizes: dict of (name, size) for names in inputs and outputs
        """
        curr_out = 0
        jac_dict = {}
        for out in outputs:
            jac_dict[out] = {}
            out_size = data_sizes[out]
            curr_in = 0
            for inpt in inputs:
                inpt_size = data_sizes[inpt]
                jac_dict[out][inpt] = jac[
                    :, curr_out : curr_out + out_size, curr_in : curr_in + inpt_size
                ]
                curr_in += inpt_size
            curr_out += out_size
        return jac_dict

    @staticmethod
    def dict_jac_to_2dmat(jac_dict, outputs, inputs, data_sizes):
        """
        Converts a dict of dict sparse format jacobian to a full 2D jacobian

        :param jac_dict : the jacobian dict of dict
        :param inputs: derive outputs wrt inputs
        :param outputs: outputs to be derived
        :param data_sizes: dict of (name, size) for names in inputs and outputs
        """
        n_outs = sum((data_sizes[out] for out in outputs))
        n_inpts = sum((data_sizes[inpt] for inpt in inputs))
        flat_jac = zeros((n_outs, n_inpts))
        curr_out = 0
        for out in outputs:
            out_size = data_sizes[out]
            curr_in = 0
            for inpt in inputs:
                inpt_size = data_sizes[inpt]
                flat_jac[
                    curr_out : curr_out + out_size, curr_in : curr_in + inpt_size
                ] = jac_dict[out][inpt]
                curr_in += inpt_size
            curr_out += out_size
        return flat_jac

    @staticmethod
    def dict_jac_to_dict(jac_dict):
        """
        Converts a full 2D jacobian to a flat dict

        :param jac_dict : the jacobian dict of dict
        """

        flat_jac = {}
        for out_name, jac_dict_loc in jac_dict.items():
            for inpt_name, jac_mat in jac_dict_loc.items():
                flat_name = DataConversion.flat_jac_name(out_name, inpt_name)
                flat_jac[flat_name] = jac_mat
        return flat_jac

    @staticmethod
    def flat_jac_name(out_name, inpt_name):
        """
        get the flat jacobian name from the full jacobian name
        :param out_name: name of output
        :param inpt_name: name of input
        """
        return out_name + DataConversion.FLAT_JAC_SEP + inpt_name

    @staticmethod
    def dict_to_jac_dict(flat_jac_dict):
        """
        Converts a flat dict to full 2D jacobian

        :param flat_jac_dict : the jacobian dict
        """

        jac = {}
        sep = DataConversion.FLAT_JAC_SEP
        all_outs = set((key.split(sep)[0]) for key in flat_jac_dict)
        all_ins = set((key.split(sep)[1]) for key in flat_jac_dict)

        for out_name in all_outs:
            jac[out_name] = {}
            for inpt_name in all_ins:
                flat_name = DataConversion.flat_jac_name(out_name, inpt_name)
                jac[out_name][inpt_name] = flat_jac_dict[flat_name]
        return jac

    @staticmethod
    def update_dict_from_array(reference_input_data, data_names, values_array):
        """Updates a data dictionary from values array
        The order of the data in the array follows the order of the
        data names

        :param reference_input_data: the base input data dict
        :param data_names: the dict keys to be updated
        :param values_array: the data array to update the dictionary
        :returns: the updated data dict
        """
        if not isinstance(values_array, ndarray):
            raise TypeError(
                "Values array must be a numpy.ndarray, "
                + "got instead :"
                + str(type(values_array))
            )
        data = deepcopy(reference_input_data)
        if not data_names:
            return data
        i_min = 0
        for key in data_names:
            value_ref = reference_input_data.get(key)
            if value_ref is None:
                raise ValueError("Reference data has no item named: " + str(key))
            value_ref = array(reference_input_data[key], copy=False)
            shape_ref = value_ref.shape
            value_flatten = value_ref.flatten()  # copy is made here
            i_max = i_min + len(value_flatten)
            if len(values_array) < i_max:
                raise ValueError(
                    "Inconsistent input array size  of values array "
                    + str(values_array)
                    + " with reference data shape "
                    + str(shape_ref)
                    + "for data named:"
                    + str(key)
                )
            value_flatten[:] = values_array[i_min:i_max]
            data[key] = value_flatten.reshape(shape_ref)
            i_min = i_max
        if i_max != values_array.size:
            raise ValueError(
                "Inconsistent data shapes !\n"
                + "Could not use the whole data array of shape "
                + str(values_array.shape)
                + " (only reached max index ="
                + str(i_max)
                + "),\nwhile updating data dictionary keys "
                + str(data_names)
                + "\n of shapes : "
                + str([(k, reference_input_data[k].shape) for k in data_names])
            )
        return data

    @staticmethod
    def deepcopy_datadict(data_dict, keys=None):
        """Performs a deepcopy of a data dict
        treats numpy arrays specially using array.copy()
        instead of deepcopy

        :param data_dict: data dict to copy
        """
        dict_cp = {}
        if keys is None:
            for key, val in data_dict.items():
                if isinstance(val, ndarray):
                    dict_cp[key] = val.copy()
                else:
                    dict_cp[key] = deepcopy(val)
        else:
            common_keys = set(keys) & set(data_dict.keys())
            for key in common_keys:
                val = data_dict[key]
                if isinstance(val, ndarray):
                    dict_cp[key] = val.copy()
                else:
                    dict_cp[key] = deepcopy(val)
        return dict_cp

    @staticmethod
    def __set_reduce(s_1, s_2):
        """
        Returns a set of unique elements of merged s_1 and s_2
        """
        set(s_1)
        set(s_2)
        return set(s_1) | set(s_2)

    @staticmethod
    def get_all_inputs(disciplines, recursive=False):
        """
        Lists all the inputs of the disciplines
        Merges the input data from the disicplines grammars

        :param disciplines: the list of disciplines to search
        :param recursive: if True, searches for the inputs of the
            sub disciplines (when some disciplines are scenarios)
        :returns: the list of input data
        """

        sub_d = [disc for disc in disciplines if not disc.is_scenario()]
        if recursive:
            sub_sc = [disc for disc in disciplines if disc.is_scenario()]

            # Take all sub disciplines
            flat_sub_d = list(
                f_reduce(
                    DataConversion.__set_reduce, (scen.disciplines for scen in sub_sc)
                )
            )
            # Take disciplines plus sub disciplines
            flat_sub_d = flat_sub_d + sub_d
            return DataConversion.get_all_inputs(flat_sub_d, recursive=False)

        data = f_reduce(
            DataConversion.__set_reduce, (disc.get_input_data_names() for disc in sub_d)
        )
        return list(data)

    @staticmethod
    def get_all_outputs(disciplines, recursive=False):
        """
        Lists all the outputs of the disciplines
        Merges the output data from the disciplines grammars

        :param disciplines: the list of disciplines to search
        :param recursive: if True, searches for the outputs of the
            sub disciplines (when some disciplines are scenarios)

        :returns: the list of output data
        """

        sub_d = [disc for disc in disciplines if not disc.is_scenario()]
        if recursive:
            sub_sc = [disc for disc in disciplines if disc.is_scenario()]

            # Take all sub disciplines
            flat_sub_d = list(
                f_reduce(
                    DataConversion.__set_reduce, (scen.disciplines for scen in sub_sc)
                )
            )
            # Take disciplines plus sub disciplines
            flat_sub_d = flat_sub_d + sub_d
            return DataConversion.get_all_outputs(flat_sub_d, recursive=False)

        data = f_reduce(
            DataConversion.__set_reduce,
            (disc.get_output_data_names() for disc in sub_d),
        )
        return list(data)
