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
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

from gemseo.algos.design_space import DesignSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.formulations.mdf import MDF
from gemseo.problems.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.sobieski.disciplines import SobieskiMission
from gemseo.problems.sobieski.disciplines import SobieskiPropulsion
from gemseo.problems.sobieski.disciplines import SobieskiStructure
from gemseo.utils.xdsmizer import XDSMizer
from numpy.testing import assert_allclose

from .formulations_basetest import FormulationsBaseTest


class TestMDFFormulation(FormulationsBaseTest):
    """"""

    # Complex step mdf already tested on propane, lighter
    def build_and_run_mdf_scenario_with_constraints(
        self, formulation, algo="SLSQP", linearize=False, dtype="complex128", **options
    ):
        """

        :param formulation: param algo:
        :param linearize: Default value = False)
        :param dtype: Default value = "complex128")
        :param algo:

        """
        if not linearize:
            dtype = "complex128"
        scenario = self.build_mdo_scenario(formulation, dtype, **options)
        if linearize:
            scenario.set_differentiation_method("user")
        else:
            scenario.set_differentiation_method("complex_step", 1e-30)
        # Set the design constraints
        scenario.add_constraint(["g_1", "g_2", "g_3"], "ineq")
        xdsmjson = XDSMizer(scenario).xdsmize()
        assert len(xdsmjson) > 0
        scenario.execute(
            {
                "max_iter": 100,
                "algo": algo,
                "algo_options": {"ftol_rel": 1e-10, "ineq_tolerance": 1e-3},
            }
        )
        scenario.print_execution_metrics()
        return scenario.optimization_result.f_opt

    # Tests with SCIPY ONLY ! Other libraries are optional...

    def test_exec_mdf_cstr(self):
        """"""
        options = {
            "tolerance": 1e-10,
            "max_mda_iter": 50,
            "warm_start": True,
            "use_lu_fact": True,
        }

        obj = self.build_and_run_mdf_scenario_with_constraints(
            "MDF", "SLSQP", linearize=True, dtype="float64", **options
        )

        assert_allclose(-obj, 3964.0, atol=1.0, rtol=0)

    def test_expected_workflow(self):
        """"""
        disc1 = SobieskiStructure()
        disc2 = SobieskiPropulsion()
        disc3 = SobieskiAerodynamics()
        disc4 = SobieskiMission()
        disciplines = [disc1, disc2, disc3, disc4]
        mdf = MDF(disciplines, "y_4", DesignSpace(), inner_mda_name="MDAGaussSeidel")
        wkf = mdf.get_expected_workflow()
        self.assertEqual(
            str(wkf),
            "[MDAChain(None), {MDAGaussSeidel(None), [SobieskiStructure(None), "
            "SobieskiPropulsion(None), SobieskiAerodynamics(None), ], }, "
            "SobieskiMission(None), ]",
        )
        mdf.get_expected_dataflow()

    def test_getsuboptions(self):
        self.assertRaises(ValueError, MDF.get_sub_options_grammar)
        self.assertRaises(ValueError, MDF.get_default_sub_options_values)


def test_grammar_type():
    """Check that the grammar type is correctly used."""
    discipline = AnalyticDiscipline({"y1": "x+y2", "y2": "x+2*y1"})
    design_space = DesignSpace()
    design_space.add_variable("x")
    grammar_type = discipline.SIMPLE_GRAMMAR_TYPE
    formulation = MDF([discipline], "y1", design_space, grammar_type=grammar_type)
    assert formulation.mda.grammar_type == grammar_type
