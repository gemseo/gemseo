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
#    INITIAL AUTHORS - initial API and implementation and/or
#                       initial documentation
#        :author:  Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

from future import standard_library

from gemseo import SOFTWARE_NAME
from gemseo.algos.opt_result import OptimizationResult
from gemseo.api import configure_logger

standard_library.install_aliases()


configure_logger(SOFTWARE_NAME)


class OptResult_Tests(unittest.TestCase):
    def test_init_dict_repr(self):
        dct = {
            "x_0": [0],
            "x_opt": [1],
            "optimizer_name": "LBFGSB",
            "message": "msg",
            "f_opt": 1.1,
            OptimizationResult.HDF_CSTR_KEY + "cname": [0.0],
            "status": 1,
            "n_obj_call": 10,
            "n_grad_call": 10,
            "n_constr_call": 10,
            "is_feasible": True,
        }
        res = OptimizationResult.init_from_dict_repr(**dct)

        assert res.x_0 == dct["x_0"]
        assert res.optimizer_name == dct["optimizer_name"]
        assert res.message == dct["message"]
        assert res.f_opt == dct["f_opt"]
        assert (
            res.constraints_values["cname"]
            == dct[OptimizationResult.HDF_CSTR_KEY + "cname"]
        )
        assert res.status == dct["status"]
        assert res.n_obj_call == dct["n_obj_call"]
        assert res.n_grad_call == dct["n_grad_call"]
        assert res.n_constr_call == dct["n_constr_call"]
        assert res.is_feasible == dct["is_feasible"]

        self.assertRaises(ValueError, OptimizationResult.init_from_dict_repr, toto=4)
