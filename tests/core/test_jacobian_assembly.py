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
#        :author: Francois Gallard, Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import json
import os
import unittest

import numpy as np
import pytest
from gemseo.core.coupling_structure import MDOCouplingStructure
from gemseo.core.jacobian_assembly import JacobianAssembly
from gemseo.problems.sobieski.core.problem import SobieskiProblem
from gemseo.problems.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.sobieski.disciplines import SobieskiMission
from gemseo.problems.sobieski.process.mda_gauss_seidel import SobieskiMDAGaussSeidel

DIRNAME = os.path.dirname(__file__)


@pytest.mark.usefixtures("tmp_wd")
class TestJacobianAssembly(unittest.TestCase):
    def test_check_inputs(self):
        disc = [SobieskiAerodynamics(), SobieskiMission()]
        assembly = JacobianAssembly(MDOCouplingStructure(disc))
        in_data = SobieskiProblem().get_default_inputs()
        args = (in_data, ["y_4"], ["x_shared"], ["y_24"])
        assembly.total_derivatives(*args)

        args = (in_data, ["toto"], ["x_shared"], ["y_24"])
        self.assertRaises(ValueError, assembly.total_derivatives, *args)

        args = (in_data, ["y_4"], ["toto"], ["y_24"])
        self.assertRaises(ValueError, assembly.total_derivatives, *args)

        args = (in_data, ["y_4"], ["x_shared"], ["x_shared"])
        self.assertRaises(ValueError, assembly.total_derivatives, *args)

        args = (in_data, ["y_4"], ["x_shared"], ["y_24"])

        self.assertRaises(
            ValueError, assembly.total_derivatives, *args, matrix_type="toto"
        )

        with self.assertRaises(TypeError):
            assembly._JacobianAssembly__check_inputs(["Y5"], ["x_3"], coupl_vars=[])
        with self.assertRaises(TypeError):
            assembly._JacobianAssembly__check_inputs(["y_4"], ["X5"], coupl_vars=[])
        with self.assertRaises(TypeError):
            assembly._JacobianAssembly__check_inputs(
                ["y_4"], ["x_3"], coupl_vars=["x_3"]
            )
        with self.assertRaises(ValueError):
            assembly.total_derivatives(
                in_data, ["y_4"], ["x_3"], ["y_12"], mode="ERROR"
            )

    def test_check_errors_consistency(self):
        disciplines = [SobieskiAerodynamics(), SobieskiMission()]
        for disc in disciplines:
            disc.linearize(force_all=True)
        assembly = JacobianAssembly(MDOCouplingStructure(disciplines))

        err_var = "IDONTEXIST"
        with self.assertRaises(ValueError) as cm:
            assembly.compute_sizes(["y_4"], [err_var], ["y_24"])
        self.assertEqual(
            "Failed to determine the size of input variable " + err_var,
            str(cm.exception),
        )

        with self.assertRaises(ValueError) as cm:
            assembly._add_differentiated_inouts(["y_4"], ["x_4"], [])

        message = cm.exception.args[0]
        assert "'SobieskiMission' has the outputs" in message
        assert "but no coupling or design" in message

    def test_linear_solver(self):
        """"""
        mda = SobieskiMDAGaussSeidel()
        with self.assertRaises(AttributeError):
            mda.assembly._JacobianAssembly__check_linear_solver("bidon")

    @staticmethod
    def __compare_mda_jac_ref(comp_jac):
        """Compare a given Jacobian with reference Jacobian in file."""
        with open(os.path.join(DIRNAME, "mda_grad_sob.json")) as inf:
            refjac = json.load(inf)
            for ykey, jac_dict in refjac.items():
                if ykey not in comp_jac:
                    return False
                for xkey, jac_loc in jac_dict.items():
                    if xkey not in comp_jac[ykey]:
                        return False
                    close = np.allclose(
                        np.array(jac_loc), comp_jac[ykey][xkey], atol=1e-1
                    )
                    if not close:
                        return False
        return True

    def test_sobieski_all_modes(self):
        """Test Sobieski's coupled derivatives computed in all modes (sparse direct,
        sparse adjoint, linear operator direct, linear operator adjoint)"""
        mda = SobieskiMDAGaussSeidel("complex128")
        mda.tolerance = 1e-14
        mda.max_iter = 100
        inputs = mda.input_grammar.keys()
        indata = SobieskiProblem("complex128").get_default_inputs(names=inputs)
        # functions/variables/couplings
        functions = ["y_4", "g_1", "g_2", "g_3", "y_1", "y_2", "y_3"]
        variables = ["x_shared", "x_1", "x_2", "x_3"]
        couplings = ["y_23", "y_12", "y_14", "y_31", "y_24", "y_32", "y_34", "y_21"]
        mda.add_differentiated_inputs(variables)
        mda.add_differentiated_outputs(functions)

        # compute coupled derivatives in all possible modes
        linearization_modes = (
            JacobianAssembly.DIRECT_MODE,
            JacobianAssembly.ADJOINT_MODE,
        )
        matrix_types = (JacobianAssembly.SPARSE, JacobianAssembly.LINEAR_OPERATOR)

        # j = mda.assembly.total_derivatives(indata, functions, variables,
        #                                   couplings,
        #                                   mode=JacobianAssembly.DIRECT_MODE,
        #                                   matrix_type=JacobianAssembly.SPARSE)
        # self.__save_jac_to_json(j, "mda_grad_sob.json")

        for mode in linearization_modes:
            for matrix_type in matrix_types:
                for use_lu_fact in [True, False]:
                    if use_lu_fact and not matrix_type == JacobianAssembly.SPARSE:
                        continue
                    mda.jac = mda.assembly.total_derivatives(
                        indata,
                        functions,
                        variables,
                        couplings,
                        mode=mode,
                        matrix_type=matrix_type,
                        use_lu_fact=use_lu_fact,
                    )
                    ok = self.__compare_mda_jac_ref(mda.jac)
                    if not ok:
                        raise Exception(
                            "Linearization mode '"
                            + str(mode)
                            + " 'failed for matrix type "
                            + str(matrix_type)
                            + " and use_lu_fact ="
                            + str(use_lu_fact)
                        )
        filepath = mda.assembly.plot_dependency_jacobian(
            functions, variables, save=True, show=False, filepath="depmat"
        )
        assert os.path.exists(filepath)

        jac = mda.assembly.total_derivatives(
            indata,
            None,
            variables,
            couplings,
            mode=JacobianAssembly.ADJOINT_MODE,
            matrix_type=JacobianAssembly.SPARSE,
            use_lu_fact=False,
        )
        assert jac["y_4"]["x_shared"] is None
        assert jac["y_1"]["TOTO"] is None


#     def __save_jac_to_json(self, jac, fpath):
#         from json import dump
#         with open(fpath, "w") as outf:
#             jac_lst = {}
#             for out, jac_d in jac.items():
#                 jac_lst[out] = {}
#                 for inp, j in jac_d.items():
#                     jac_lst[out][inp] = j.tolist()
#             dump(jac_lst, outf)
