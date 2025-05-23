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
# INITIAL AUTHORS - initial API and implementation and/or
#                   initial documentation
#        :author:  Francois Gallard, Charlie Vanaret, Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import pickle
import unittest
from os.path import exists
from pathlib import Path

import numpy as np
import pytest

from gemseo.algos.design_space import DesignSpace
from gemseo.core.coupling_structure import CouplingStructure
from gemseo.problems.mdo.scalable.data_driven.discipline import (
    DataDrivenScalableDiscipline,
)
from gemseo.problems.mdo.sobieski.core.problem import SobieskiProblem
from gemseo.problems.mdo.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.mdo.sobieski.disciplines import SobieskiMission
from gemseo.problems.mdo.sobieski.disciplines import SobieskiPropulsion
from gemseo.problems.mdo.sobieski.disciplines import SobieskiStructure
from gemseo.scenarios.mdo_scenario import MDOScenario
from gemseo.utils.pickle import to_pickle

N_SAMPLES = 10


@pytest.mark.usefixtures("tmp_wd")
class ScalableProblem(unittest.TestCase):
    """Tests of the scalable methodology on Sobieski' SSBJ."""

    (size_x, size_y) = (2, 3)
    fill_factor = 0.4
    problem = None
    original_disciplines = None
    scalable_disciplines = None
    scalable_model = "ScalableDiagonalModel"

    def setUp(self) -> None:
        """At creation of unittest, initiate a Sobieski problem class."""
        if ScalableProblem.problem is not None:
            return
        ScalableProblem.problem = SobieskiProblem()

        if ScalableProblem.original_disciplines is None:
            ScalableProblem.original_disciplines = [
                SobieskiAerodynamics(),
                SobieskiPropulsion(),
                SobieskiStructure(),
                SobieskiMission(),
            ]
            sizes = self.set_sizes()
            ScalableProblem.sizes = sizes
            ScalableProblem.scalable_disciplines = self.create_scalable_disciplines()

    def test_serialize(self) -> None:
        disc = ScalableProblem.scalable_disciplines[0]
        outf = "scalable.o"
        to_pickle(disc, outf)
        assert exists(outf)

    def _determine_size(self, name, original_sizes=False):
        """Determine variable size.

        :param str name: variable name
        :param dict original_sizes: if True, use original sizes.
            Default: True.
        """
        if original_sizes:
            inputs = self.problem.get_default_inputs()
            if name in inputs:
                return inputs[name], inputs[name].size
            out_d = {
                "y_4": np.ones(1, dtype=np.float64),
                "y_11": np.ones(1, dtype=np.float64),
            }
            if name in out_d:
                return out_d[name]

            msg = "Unknown data "
            raise ValueError(msg, name)
        if name == "y_4":
            return 0.5 * np.ones(1)
        if name.startswith("y"):
            return 0.5 * np.ones(ScalableProblem.size_y)
        if name.startswith("x"):
            return 0.5 * np.ones(ScalableProblem.size_x)
        return 0.5 * np.ones(1)

    def set_sizes(self):
        """Set the sizes of the variables (local, global, coupling) according to the
        parameterization of the scalable problem."""
        sizes = {}

        for disc in ScalableProblem.original_disciplines:
            input_names = disc.io.input_grammar
            output_names = disc.io.output_grammar
            for name in list(set(input_names) | set(output_names)):
                value = self._determine_size(name)
                sizes[name] = value.size
        return sizes

    def create_scalable_disciplines(self):
        """Initialize the scalable disciplines of Sobieski's SSBJ."""
        if ScalableProblem.scalable_disciplines is None:
            var_lb = {}
            var_ub = {}
            ScalableProblem.scalable_disciplines = []
            for discipline in ScalableProblem.original_disciplines:
                input_names = [
                    name
                    for name in discipline.io.input_grammar
                    if not name.startswith("c_")
                ]
                for input_name in input_names:
                    l_bnds, u_bnds = self.problem.get_bounds_by_name([input_name])
                    var_lb[input_name] = l_bnds
                    var_ub[input_name] = u_bnds

                sizes = ScalableProblem.sizes
                fill_factor = ScalableProblem.fill_factor
                with (Path(__file__).parent / f"{discipline.name}.pkl").open("rb") as f:
                    pickler = pickle.Unpickler(f)
                    dataset = pickler.load()
                scal_disc = DataDrivenScalableDiscipline(
                    ScalableProblem.scalable_model,
                    data=dataset,
                    sizes=sizes,
                    fill_factor=fill_factor,
                )
                ScalableProblem.scalable_disciplines.append(scal_disc)
        return ScalableProblem.scalable_disciplines

    # TESTS
    def test_plot_splines(self) -> None:
        """Test plot splines."""
        for discipline in ScalableProblem.scalable_disciplines:
            model = discipline.scalable_model
            files = model.plot_1d_interpolations(show=False, save=True)
            assert files
            for fname in files:
                assert exists(fname)

    def test_execute(self) -> None:
        """Verify that the scalable_models can be executed."""
        for disc in ScalableProblem.scalable_disciplines:
            disc.scalable_model.scalable_function()
            disc.scalable_model.scalable_derivatives()

    def test_plot_dependency(self) -> None:
        """Plot the dependency matrices."""
        for disc in ScalableProblem.scalable_disciplines:
            fname = disc.scalable_model.plot_dependency(show=False, save=True)
            assert exists(fname)

    def test_grad_funcs(self) -> None:
        """"""
        formulation = "MDF"
        dv_names = ["x_shared", "x_1", "x_2", "x_3"]

        # create design space
        design_space = DesignSpace()
        for name in dv_names:
            value = 0.5 * np.ones(ScalableProblem.sizes[name])
            design_space.add_variable(
                name,
                ScalableProblem.sizes[name],
                lower_bound=0.0,
                upper_bound=1.0,
                value=value,
            )
        # add target coupling variables for IDF
        if formulation == "IDFFormulation":
            scalable_disciplines = ScalableProblem.scalable_disciplines
            coupling_structure = CouplingStructure(scalable_disciplines)
            # add an optimization variable for each coupling variable
            for coupling in coupling_structure.strong_couplings:
                value = 0.5 * np.ones(ScalableProblem.sizes[coupling])
                design_space.add_variable(
                    coupling,
                    ScalableProblem.sizes[coupling],
                    lower_bound=0.0,
                    upper_bound=1.0,
                    value=value,
                )

        scenario = MDOScenario(
            ScalableProblem.scalable_disciplines,
            "y_4",
            design_space,
            formulation_name=formulation,
            maximize_objective=True,
        )
        scenario.set_differentiation_method("finite_differences")

        # add disciplinary constraints
        cstr_threshold = 0.5
        for cstr in ["g_1", "g_2", "g_3"]:
            scenario.add_constraint(cstr, constraint_type="ineq", value=cstr_threshold)

        opt_pb = scenario.formulation.optimization_problem

        opt_pb.preprocess_functions()
        for func in opt_pb.functions:
            func.check_grad(opt_pb.design_space.get_current_value())

    def test_grad(self) -> None:
        """Verify the analytical gradients against finite differences."""
        for disc in ScalableProblem.scalable_disciplines:
            assert disc.check_jacobian(
                derr_approx="finite_differences",
                step=1e-6,
                threshold=1e-3,
                linearization_mode="auto",
            )

    def test_group_dep(self) -> None:
        hdf_node_path = ScalableProblem.original_disciplines[3].name
        with (Path(__file__).parent / f"{hdf_node_path}.pkl").open("rb") as f:
            pickler = pickle.Unpickler(f)
            dataset = pickler.load()
        DataDrivenScalableDiscipline(
            ScalableProblem.scalable_model,
            data=dataset,
            sizes=ScalableProblem.sizes,
            fill_factor=ScalableProblem.fill_factor,
            group_dep={"y_4": []},
        )
