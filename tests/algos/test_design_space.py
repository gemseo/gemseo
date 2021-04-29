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
#        :author: Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
from os.path import dirname, exists, join, realpath

import numpy as np
import pytest
from numpy import array, inf, int32, ones
from numpy.linalg import norm

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt_result import OptimizationResult
from gemseo.problems.sobieski.core import SobieskiProblem

CURRENT_DIR = dirname(__file__)
TEST_INFILE = join(CURRENT_DIR, "design_space.txt")

FAIL_HDF = join(dirname(realpath(__file__)), "fail.hdf5")


@pytest.mark.usefixtures("tmp_wd")
class TestDesignSpace(unittest.TestCase):
    """Test the creation and update of the design space."""

    def test_creation(self):
        """"""
        design_space = DesignSpace()
        design_space.add_variable("x1", 3, DesignSpace.FLOAT, 0.0, 1.0)
        self.assertRaises(
            Exception, design_space.add_variable, "x1", 3, DesignSpace.FLOAT, 0.0, 1.0
        )
        self.assertRaises(
            Exception, design_space.add_variable, "x3", 0, DesignSpace.FLOAT, 0.0, 1.0
        )
        self.assertRaises(
            Exception,
            design_space.add_variable,
            "x4",
            3,
            [DesignSpace.FLOAT] * 2,
            0.0,
            1.0,
        )

        self.assertRaises(
            Exception, design_space.add_variable, "x5", -1, DesignSpace.FLOAT, 0.0, 1.0
        )
        self.assertRaises(
            Exception,
            design_space.add_variable,
            "x6",
            1,
            DesignSpace.FLOAT,
            float("nan"),
            1.0,
        )
        self.assertRaises(
            Exception, design_space.add_variable, "x7", 3, [1, "a", "float"], 0.0, 1.0
        )
        self.assertRaises(Exception, design_space.add_variable, "x8", 1, "a", 0.0, 1.0)

        self.assertRaises(
            Exception, design_space.add_variable, "x9", 3, DesignSpace.FLOAT, len, 1
        )

        self.assertRaises(
            Exception,
            design_space.add_variable,
            "x10",
            3,
            DesignSpace.FLOAT,
            [0, 1.0, 0],
            [1, 0.0, 1],
        )

        self.assertRaises(
            Exception,
            design_space.add_variable,
            "x11",
            3,
            DesignSpace.FLOAT,
            [0.0, 0.0],
            1.0,
        )

        self.assertRaises(
            ValueError,
            design_space.add_variable,
            "x12",
            1,
            DesignSpace.FLOAT,
            [[1.0]],
            [2.0],
        )

        self.assertRaises(
            ValueError,
            design_space.add_variable,
            "x13",
            1,
            DesignSpace.FLOAT,
            1.0,
            2.0,
            3.0,
        )

        self.assertRaises(
            ValueError,
            design_space.add_variable,
            "x14",
            1,
            DesignSpace.FLOAT,
            1.0,
            2.0,
            0.0,
        )

        self.assertRaises(KeyError, design_space.get_current_x_normalized)

        design_space.add_variable(
            "x15", 3, DesignSpace.FLOAT, 0.0, 1.0, value=[None, None, None]
        )

    def test_curr_x_n(self):

        design_space = DesignSpace()
        design_space.add_variable("x", 1, DesignSpace.FLOAT, 0.0, 2.0)
        design_space.add_variable("y", 1, DesignSpace.FLOAT, -2.0, 2.0)
        design_space.set_current_x({"x": array([1.0]), "y": array([0.0])})
        x_n = design_space.get_current_x_normalized()
        assert (x_n == 0.5).all()

        design_space.set_current_x(OptimizationResult(x_opt=array([1.0, 0.0])))
        x_n = design_space.get_current_x_normalized()
        assert (x_n == 0.5).all()

        self.assertRaises(
            Exception,
            design_space.set_current_x,
            {"x": array([1.0, 1.0]), "y": array([0.0])},
        )

        self.assertRaises(
            Exception,
            design_space.set_current_x,
            OptimizationResult(x_opt=array([1.0])),
        )

        self.assertRaises(Exception, design_space.set_current_x, 1.0)

    def test_1dv(self):
        ds = DesignSpace.read_from_txt(join(CURRENT_DIR, "design_space_4.txt"))
        assert ds.variables_names == ["x_shared"]

    def test_common_dtype(self):

        design_space = DesignSpace()
        design_space.add_variable("x", 1, DesignSpace.INTEGER, 0, 2)
        x_i = array([0], dtype=int32)
        design_space.set_current_x(x_i)
        dct = design_space.array_to_dict(x_i)
        x_i_conv = design_space.dict_to_array(dct)
        assert x_i_conv.dtype == x_i.dtype
        assert x_i_conv == x_i

        assert design_space.round_vect(array([1.2])) == 1
        assert design_space.round_vect(array([1.9])) == 2

        rounded = design_space.round_vect(array([[1.9], [0.1]]))
        assert (rounded == array([[2], [0]])).all()

        self.assertRaises(ValueError, design_space.round_vect, array([[[1.0]]]))

    def test_filter(self):
        """Test the filtering of a design space variables."""
        # Filtering by variable name
        design_space = DesignSpace()
        design_space.add_variable("x1", 1, "float", -1.0, 0.0, -0.5)
        design_space.add_variable("x2", 3, "float", -1.0, 0.0, -0.5)
        new_space = design_space.filter("x2", copy=True)
        assert not new_space.__contains__("x1")
        assert new_space.__contains__("x2")
        assert new_space is not design_space
        design_space.filter(["x2"])
        assert not design_space.__contains__("x1")
        assert design_space.__contains__("x2")
        # Filtering by dimensions
        design_space = DesignSpace()
        design_space.add_variable("x1", 1, "float", -1.0, 0.0, -0.5)
        design_space.add_variable("x2", 3, "float", -1.0, 0.0, -0.5)
        design_space.filter_dim("x2", [0])
        self.assertRaises(ValueError, design_space.filter_dim, "x2", [1])
        self.assertRaises(ValueError, design_space.filter, "unknown_x")

    def test_extend(self):
        """Test the extension of a design space with another."""
        design_space = DesignSpace()
        design_space.add_variable("x1", 1, "float", -1.0, 0.0, -0.5)
        other = DesignSpace()
        other.add_variable("x2", 3, "float", -1.0, 0.0, -0.5)
        design_space.extend(other)
        assert design_space.__contains__("x2")
        assert design_space.get_size("x2") == other.get_size("x2")
        assert (design_space.get_type("x2") == other.get_type("x2")).all()
        assert (design_space.get_lower_bound("x2") == other.get_lower_bound("x2")).all()
        assert (design_space.get_upper_bound("x2") == other.get_upper_bound("x2")).all()
        assert (design_space.get_current_x(["x2"]) == other.get_current_x(["x2"])).all()

    def test_active_bounds(self):

        design_space = DesignSpace()
        design_space.add_variable("x", 1, DesignSpace.FLOAT, 0.0, 2.0)
        design_space.add_variable("y", 1, DesignSpace.FLOAT, -2.0, 2.0)
        design_space.add_variable("z", 1, DesignSpace.FLOAT)
        lb_1, ub_1 = design_space.get_active_bounds(
            {"x": array([0.0]), "y": array([2.0]), "z": array([2.0])}
        )

        lb_2, ub_2 = design_space.get_active_bounds(array([1e-12, 2.0 - 1e-12, 1e-12]))

        assert lb_1 == lb_2
        assert lb_1["x"] == [True]
        assert lb_1["y"] == [False]
        assert not lb_1["z"][0]
        assert ub_1 == ub_2
        assert ub_1["y"] == [True]
        assert ub_1["x"] == [False]
        assert not ub_1["z"][0]

        self.assertRaises(Exception, design_space.get_active_bounds, "test")
        self.assertRaises(KeyError, design_space.get_active_bounds)

    def test_get_indexed_variables_names(self):
        design_space = DesignSpace()
        design_space.add_variable("x", 1)
        design_space.add_variable("z", 2)
        assert design_space.get_indexed_variables_names()[0] == "x"
        assert design_space.get_indexed_variables_names()[1] == "z!0"
        assert design_space.get_indexed_variables_names()[2] == "z!1"

    def test_bounds(self):
        design_space = DesignSpace()
        design_space.add_variable("x", 1, DesignSpace.FLOAT, None, 2.0)
        design_space.add_variable("y", 1, DesignSpace.FLOAT, 0.0, None)

        assert design_space.get_lower_bound("x") == -inf
        assert design_space.get_lower_bound("y") is not None

        assert design_space.get_upper_bound("y") == inf
        assert design_space.get_upper_bound("x") is not None

        assert design_space.get_lower_bounds(["x"]) == -inf
        assert design_space.get_lower_bounds(["y"]) == 0.0

        assert design_space.get_upper_bounds(["x"]) == 2.0
        assert design_space.get_upper_bounds(["y"]) == inf
        #         self.assertRaises(KeyError, design_space.get_lower_bounds, ["x"])
        #         self.assertRaises(KeyError, design_space.get_upper_bounds, ["y"])

        self.assertRaises(Exception, design_space.set_lower_bound, "x", ones(2))
        self.assertRaises(Exception, design_space.set_upper_bound, "x", ones(2))

        self.assertRaises(ValueError, design_space.set_upper_bound, "x", float("nan"))
        self.assertRaises(ValueError, design_space.set_lower_bound, "x", float("nan"))
        self.assertRaises(
            ValueError, design_space._check_value, array([float("nan")]), "x"
        )

    def test_normalization(self):
        design_space = DesignSpace()
        design_space.add_variable(
            "x_1", 2, DesignSpace.FLOAT, array([None, 0.0]), array([0.0, None])
        )
        design_space.add_variable("x_2", 1, DesignSpace.FLOAT, 0.0, 10.0)
        design_space.add_variable("x_3", 1, DesignSpace.INTEGER, 0.0, 10.0)
        # Test the normalization policies:
        assert not design_space.normalize["x_1"][0]
        assert not design_space.normalize["x_1"][1]
        assert design_space.normalize["x_2"]
        assert design_space.normalize["x_3"]
        # Test the normalization:
        design_space.set_current_x(array([-10.0, 10.0, 5.0, 5]))
        current_x_norm = design_space.get_current_x_normalized()
        ref_current_x_norm = array([-10.0, 10.0, 0.5, 0.5])
        self.assertAlmostEqual(norm(current_x_norm - ref_current_x_norm), 0.0)
        unnorm_curent_x = design_space.unnormalize_vect(current_x_norm)
        current_x = design_space.get_current_x()
        self.assertAlmostEqual(norm(unnorm_curent_x - current_x), 0.0)
        self.assertRaises(ValueError, design_space.normalize_vect, ones((2, 2, 2)))

        x_2d = ones((5, 4))
        x_u = design_space.unnormalize_vect(x_2d)
        assert (x_u == array([1.0, 1.0, 10.0, 10.0] * 5).reshape((5, 4))).all()

        x_n = design_space.normalize_vect(x_2d)
        assert (x_n == array([1.0, 1.0, 0.1, 0.1] * 5).reshape((5, 4))).all()

        self.assertRaises(ValueError, design_space.normalize_vect, ones((2, 2, 2)))

        self.assertRaises(ValueError, design_space.unnormalize_vect, ones((2, 2, 2)))

        design_space = DesignSpace()
        design_space.add_variable("x", 1, DesignSpace.INTEGER, 1, 1)
        assert design_space.normalize_vect(ones(1))[0] == 0.0
        assert design_space.unnormalize_vect(ones(1) * 0)[0] == 1.0
        assert design_space.unnormalize_vect(array([[0.0], [0.0]]))[0][0] == 1.0
        assert design_space.unnormalize_vect(array([[0.0], [0.0]]))[1][0] == 1.0

    def test_norm_policy(self):
        design_space = DesignSpace()
        design_space.add_variable(
            "x_1", 2, DesignSpace.FLOAT, array([None, 0.0]), array([0.0, None])
        )
        self.assertRaises(ValueError, design_space._add_norm_policy, "toto")
        design_space.variables_sizes.pop("x_1")
        self.assertRaises(ValueError, design_space._add_norm_policy, "x_1")
        design_space.variables_types.pop("x_1")
        self.assertRaises(ValueError, design_space._add_norm_policy, "x_1")

        design_space.add_variable(
            "x_c", 1, DesignSpace.FLOAT, array([0.0]), array([0.0])
        )
        assert not design_space.normalize["x_c"]

        design_space.add_variable(
            "x_e", 1, DesignSpace.FLOAT, array([0.0]), array([0.0])
        )
        design_space.variables_types["x_e"] = array(["toto"])
        self.assertRaises(ValueError, design_space._add_norm_policy, "x_e")
        design_space.variables_types.pop("x_e")
        self.assertRaises(ValueError, design_space._add_norm_policy, "x_e")

    def test_current_x(self):
        names = ["x_1", "x_2"]
        sizes = {"x_1": 1, "x_2": 2}
        l_b = {"x_1": 0.5, "x_2": (None, 2.0)}
        u_b = {"x_1": None, "x_2": (4.0, 5.0)}
        var_types = {"x_1": DesignSpace.FLOAT, "x_2": DesignSpace.INTEGER}
        x_0 = np.array([0.5, 4.0, 4.0])
        # create the design space
        design_space = DesignSpace()

        # fill the design space
        for name in names:
            design_space.add_variable(
                name, sizes[name], var_types[name], l_b=l_b[name], u_b=u_b[name]
            )

        design_space.set_current_x(x_0)
        design_space.check()
        design_space.check_membership(2 * ones(3))
        self.assertRaises(ValueError, design_space.check_membership, 2 * ones(2))
        self.assertRaises(TypeError, design_space.check_membership, [2.0] * 3)
        self.assertRaises(ValueError, design_space.check_membership, 6 * ones(3))
        self.assertRaises(
            ValueError,
            design_space.check_membership,
            {"x_1": ones(1), "x_2": 2.5 * ones(2)},
        )

        self.assertRaises(Exception, design_space.set_current_x, {"x_1": 0.0})

        self.assertRaises(Exception, design_space.set_current_x, x_0 - 1000.0)

        """
        Design Space: 3 scalar variables
        Variable   Type     Lower  Current  Upper
        x_1        float    0.5    0.5      inf
        x_2!0      integer  -inf   4        4
        x_2!1      integer  2      4        5
        """

        assert design_space.get_type("x_1") == np.array([DesignSpace.FLOAT])
        assert design_space.get_type("x_3") is None

        design_space.set_current_variable("x_1", np.array([5.0]))
        assert design_space.get_current_x_dict()["x_1"][0] == 5.0
        self.assertRaises(ValueError, design_space.set_current_variable, "x_3", 1.0)

        self.assertRaises(Exception, design_space.add_variable, "error", l_b=1.0, u_b=0)

        design_space = DesignSpace()
        design_space.add_variable("x", 1, DesignSpace.FLOAT, 0.0, 2.0)
        design_space.set_current_x({"x": None})
        assert not design_space.has_current_x()

    def get_sob_ds(self):
        names = [
            "x_shared",
            "x_1",
            "x_2",
            "x_3",
            "y_14",
            "y_32",
            "y_31",
            "y_24",
            "y_34",
            "y_23",
            "y_21",
            "y_12",
        ]
        problem = SobieskiProblem()
        def_inputs = problem.get_default_inputs_equilibrium(names)

        ref_ds = DesignSpace()
        for name in names:
            value = def_inputs[name]
            l_b, u_b = problem.get_bounds_by_name([name])
            size = value.size
            ref_ds.add_variable(name, size, "float", l_b, u_b, value)

        return ref_ds

    def test_read_write(self):
        ref_ds = self.get_sob_ds()
        f_path = "sobieski_design_space.txt"
        ref_ds.export_to_txt(f_path)
        read_ds = DesignSpace.read_from_txt(f_path)
        read_ds.get_lower_bounds()
        self.check_ds(ref_ds, read_ds, f_path)

        ds = DesignSpace.read_from_txt(TEST_INFILE)
        assert not ds.has_current_x()
        for i in range(1, 9):
            testfile = join(CURRENT_DIR, "design_space_fail_" + str(i) + ".txt")
            self.assertRaises(ValueError, DesignSpace.read_from_txt, testfile)

        for i in range(1, 4):
            testfile = join(CURRENT_DIR, "design_space_" + str(i) + ".txt")
            header = None
            if i == 2:
                header = ["name", "value", "lower_bound", "type", "upper_bound"]
            ds = DesignSpace.read_from_txt(testfile, header=header)

        ds = DesignSpace.read_from_txt(TEST_INFILE)
        ds.set_lower_bound("x_shared", None)
        ds.set_upper_bound("x_shared", None)

        out_f = "table.txt"
        ds.export_to_txt(out_f, sortby="upper_bound")
        assert exists(out_f)

    def test_dict_to_array(self):
        design_space = DesignSpace()
        design_space.add_variable("x", 1, DesignSpace.FLOAT, 0.0, 2.0)
        design_space.add_variable("y", 1, DesignSpace.FLOAT, -2.0, 2.0)
        self.assertRaises(Exception, design_space.dict_to_array, {"x": 1.0})
        self.assertRaises(KeyError, design_space.dict_to_array, {"x": array([1.0])})

        x = design_space.dict_to_array({"x": array([1.0])}, False)
        assert x == 1.0

    def check_ds(self, ref_ds, read_ds, f_path):
        """
        :param ref_ds: param read_ds:
        :param f_path:
        :param read_ds:
        """
        assert exists(f_path)
        self.assertListEqual(read_ds.variables_names, ref_ds.variables_names)

        err = read_ds.get_lower_bounds() - ref_ds.get_lower_bounds()
        self.assertAlmostEqual(norm(err), 0.0, places=14)

        err = read_ds.get_upper_bounds() - ref_ds.get_upper_bounds()
        self.assertAlmostEqual(norm(err), 0.0, places=14)

        err = read_ds.get_current_x() - ref_ds.get_current_x()
        self.assertAlmostEqual(norm(err), 0.0, places=14)

        type_read = [
            t for name in read_ds.variables_names for t in read_ds.get_type(name)
        ]

        type_ref = [
            t for name in read_ds.variables_names for t in ref_ds.get_type(name)
        ]

        self.assertListEqual(type_read, type_ref)

        for name in ref_ds.variables_names:
            assert name in read_ds.variables_names

        ref_str = str(ref_ds)
        assert ref_str == str(read_ds)
        assert len(ref_str) > 1000
        assert len(ref_str.split("\n")) > 20

    def test_hdf5_export(self):
        """Tests the export of a Design space in the HDF5 format."""
        ref_ds = self.get_sob_ds()
        f_path = "_sobieski_design_space.h5"
        ref_ds.export_hdf(f_path)
        read_ds = DesignSpace(f_path)
        self.check_ds(ref_ds, read_ds, f_path)

    def test_ctor_error_with_missing_file(self):
        self.assertRaises(Exception, DesignSpace, "dummy.h5")

    def test_fail_import(self):
        self.assertRaises(KeyError, DesignSpace().import_hdf, FAIL_HDF)

    def test_get_pretty_table(self):
        design_space = DesignSpace()
        design_space.add_variable("toto")
        str_repr = design_space.get_pretty_table().get_string()
        assert "-inf" in str(str_repr)

    def test_project_into_bounds(self):
        """Tests the projection onto the design space bounds."""
        design_space = DesignSpace()
        design_space.add_variable("x", 3, DesignSpace.FLOAT, -1.0, 2.0)
        x_c = [-2, 0.5, 3]
        x_p = design_space.project_into_bounds(x_c, normalized=False)
        self.assertAlmostEqual(norm(x_p - [-1, 0.5, 2]), 0.0)
        x_p = design_space.project_into_bounds(x_c, normalized=True)
        self.assertAlmostEqual(norm(x_p - [0, 0.5, 1]), 0.0)

    def test_contains(self):
        design_space = DesignSpace()
        design_space.add_variable("x")
        assert "x" in design_space
        assert "y" not in design_space

    def test_len(self):
        design_space = DesignSpace()
        design_space.add_variable("x")
        design_space.add_variable("y", size=2)
        assert len(design_space) == 2

    def test_getitem(self):
        design_space = DesignSpace()
        design_space.add_variable("x", value=0.5)
        design_space.add_variable("y", size=2)

        assert design_space["x"] == {
            "name": "x",
            "type": "float",
            "value": array([0.5]),
            "size": 1,
            "l_b": array([-inf]),
            "u_b": array([inf]),
        }

        assert design_space["y"]["value"] is None

        expected = "The parameter indices are comprise between 0 and 1. Got 2."
        with pytest.raises(ValueError, match=expected):
            design_space[2]

        expected = "The design space does not contain 'foo'."
        with pytest.raises(ValueError, match=expected):
            design_space["foo"]

    def test_get_variables_indexes(self):
        """Test the variables indexes getter."""
        space = DesignSpace()
        space.add_variable("x", 3)
        space.add_variable("y", 2)
        space.add_variable("z", 1)
        assert (space.get_variables_indexes(["x"]) == array([0, 1, 2])).all()
        assert (space.get_variables_indexes(["y"]) == array([3, 4])).all()
        assert (space.get_variables_indexes(["z"]) == array([5])).all()
        assert (space.get_variables_indexes(["x", "y"]) == array([0, 1, 2, 3, 4])).all()
        assert (space.get_variables_indexes(["x", "z"]) == array([0, 1, 2, 5])).all()
        assert (space.get_variables_indexes(["y", "z"]) == array([3, 4, 5])).all()
