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
#        :author: Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import division, unicode_literals

import unittest

from numpy import array, array_equal, hstack, zeros

from gemseo.utils.data_conversion import DataConversion, flatten_mapping


class TestDataConversion(unittest.TestCase):
    """Test the conversion between data dict and numpy arrays."""

    def test_dict_to_array(self):
        """"""
        data_dict = {"x": array([0.0, 1.0]), "y": array([2.0]), "z": array([3.0, 4.0])}
        data_names = ["z", "x"]
        zx_array = DataConversion.dict_to_array(data_dict, data_names)
        assert array_equal(zx_array, array([3.0, 4.0, 0.0, 1.0]))

        data_names = []
        empty_array = DataConversion.dict_to_array(data_dict, data_names)
        assert array_equal(empty_array, array([]))

        data_sizes = {k: v.size for k, v in data_dict.items()}
        data_names = ["z", "x"]
        dict_m = DataConversion.array_to_dict(zx_array, data_names, data_sizes)
        for k in data_names:
            assert (dict_m[k] == data_dict[k]).all()

        self.assertRaises(
            ValueError,
            DataConversion.array_to_dict,
            zeros((2, 2, 2)),
            data_names,
            data_sizes,
        )

    def test_update_dict_from_array(self):
        """Check the update of a data mapping from data array and names."""
        data_dict = {"x": array([0.0, 1.0]), "y": array([2.0]), "z": array([3, 4])}
        data_names = ["y", "z"]
        values_array = array([0.5, 1.0, 2.0])
        new_data_dict = DataConversion.update_dict_from_array(
            data_dict, data_names, values_array
        )
        expected = array([0.5])
        assert array_equal(new_data_dict["y"], expected)
        assert new_data_dict["y"].dtype == expected.dtype
        expected = array([1, 2])
        assert array_equal(new_data_dict["z"], expected)
        assert new_data_dict["z"].dtype == expected.dtype

        data_names = []
        new_data_dict = DataConversion.update_dict_from_array(
            data_dict, data_names, values_array
        )
        for k, v in data_dict.items():

            assert array_equal(new_data_dict[k], v)

        data_names = ["y"]
        values_array = 1.0
        self.assertRaises(
            TypeError,
            DataConversion.update_dict_from_array,
            data_dict,
            data_names,
            values_array,
        )

        values_array = array([0.5])
        data_dict["y"] = None
        self.assertRaises(
            ValueError,
            DataConversion.update_dict_from_array,
            data_dict,
            data_names,
            values_array,
        )

    def test_update_too_long(self):
        """"""
        data_dict = {"x": array([0.0, 1.0]), "y": array([2.0]), "z": array([3.0, 4.0])}
        data_names = ["y"]
        values_array = array([0.5, 1.5])
        with self.assertRaises(Exception):
            DataConversion.update_dict_from_array(data_dict, data_names, values_array)

    def test_update_too_short(self):
        """"""
        data_dict = {"x": array([0.0, 1.0]), "y": array([2.0]), "z": array([3.0, 4.0])}
        data_names = ["z"]
        values_array = array([0.5])
        with self.assertRaises(Exception):
            DataConversion.update_dict_from_array(data_dict, data_names, values_array)

    def test_dict_jac_to_2dmat(self):
        f_g = {"x": array([[0.0, 1.0]]), "y": array([[2.0]]), "z": array([[3.0, 4.0]])}
        jac_dict = {"f": f_g}
        data_sizes = {"f": 1, "y": 1, "x": 2, "z": 2}
        outputs = ["f"]
        inputs = ["x", "y", "z"]

        flat_jac = DataConversion.dict_jac_to_2dmat(
            jac_dict, outputs, inputs, data_sizes
        )
        assert (flat_jac == hstack([f_g[inpt] for inpt in inputs])).all()

        f_g_rec = DataConversion.jac_2dmat_to_dict(
            flat_jac, outputs, inputs, data_sizes
        )
        for k in inputs:
            assert (jac_dict["f"][k] == f_g_rec["f"][k]).all()


def test_flatten_mapping():
    """Check that a nested mapping is correctly flattened."""
    mapping = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
    expected = {"a": 1, "b_c": 2, "b_d_e": 3}
    assert flatten_mapping(mapping) == expected
