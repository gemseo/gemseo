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

import os
import unittest
from os.path import dirname
from os.path import exists
from os.path import join
from shutil import copy

import numpy as np
import pytest
from gemseo.algos.design_space import DesignSpace
from gemseo.caches.hdf5_cache import HDF5Cache
from gemseo.core.coupling_structure import MDOCouplingStructure
from gemseo.core.mdo_scenario import MDOScenario
from gemseo.problems.scalable.data_driven.discipline import ScalableDiscipline
from gemseo.problems.sobieski.core.problem import SobieskiProblem
from gemseo.problems.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.sobieski.disciplines import SobieskiMission
from gemseo.problems.sobieski.disciplines import SobieskiPropulsion
from gemseo.problems.sobieski.disciplines import SobieskiStructure

DIRNAME = dirname(__file__)
HDF_CACHE_PATH = join(DIRNAME, "dataset.hdf5")
COPIED_CACHE_PATH = join(DIRNAME, "dataset_discipline.hdf5")

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

    @classmethod
    def setUpClass(cls):
        """Create a copy of the cache file in order to avoid issues when using
        mutliprocessing."""
        copy(HDF_CACHE_PATH, COPIED_CACHE_PATH)

    @classmethod
    def tearDownClass(cls):
        os.remove(COPIED_CACHE_PATH)

    def setUp(self):
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

    def test_serialize(self):
        disc = ScalableProblem.scalable_disciplines[0]
        outf = "scalable.o"
        disc.serialize(outf)
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

            raise Exception("Unknown data ", name)
        else:
            if name == "y_4":
                return 0.5 * np.ones(1)
            elif name.startswith("y"):
                return 0.5 * np.ones(ScalableProblem.size_y)
            elif name.startswith("x"):
                return 0.5 * np.ones(ScalableProblem.size_x)
            else:
                return 0.5 * np.ones(1)

    def set_sizes(self):
        """Set the sizes of the variables (local, global, coupling) according to the
        parameterization of the scalable problem."""
        sizes = {}

        for disc in ScalableProblem.original_disciplines:
            input_names = disc.get_input_data_names()
            output_names = disc.get_output_data_names()
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
                    for name in discipline.get_input_data_names()
                    if not name.startswith("c_")
                ]
                for input_name in input_names:
                    l_bnds, u_bnds = self.problem.get_bounds_by_name([input_name])
                    var_lb[input_name] = l_bnds
                    var_ub[input_name] = u_bnds

                hdf_node_path = discipline.name
                sizes = ScalableProblem.sizes
                fill_factor = ScalableProblem.fill_factor
                data = HDF5Cache(HDF_CACHE_PATH, hdf_node_path).export_to_dataset()
                scal_disc = ScalableDiscipline(
                    ScalableProblem.scalable_model,
                    data=data,
                    sizes=sizes,
                    fill_factor=fill_factor,
                )
                ScalableProblem.scalable_disciplines.append(scal_disc)
        return ScalableProblem.scalable_disciplines

    # TESTS
    def test_plot_splines(self):
        """Test plot splines."""
        for discipline in ScalableProblem.scalable_disciplines:
            model = discipline.scalable_model
            files = model.plot_1d_interpolations(show=False, save=True)
            assert files
            for fname in files:
                assert exists(fname)

    def test_execute(self):
        """Verify that the scalable_models can be executed."""
        for disc in ScalableProblem.scalable_disciplines:
            disc.scalable_model.scalable_function()
            disc.scalable_model.scalable_derivatives()

    def test_plot_dependency(self):
        """Plot the dependency matrices."""
        for disc in ScalableProblem.scalable_disciplines:
            fname = disc.scalable_model.plot_dependency(show=False, save=True)
            assert exists(fname)

    def test_grad_funcs(self):
        """"""
        formulation = "MDF"
        dv_names = ["x_shared", "x_1", "x_2", "x_3"]

        # create design space
        design_space = DesignSpace()
        for name in dv_names:
            value = 0.5 * np.ones(ScalableProblem.sizes[name])
            design_space.add_variable(
                name, ScalableProblem.sizes[name], l_b=0.0, u_b=1.0, value=value
            )
        # add target coupling variables for IDF
        if formulation == "IDFFormulation":
            scalable_disciplines = ScalableProblem.scalable_disciplines
            coupling_structure = MDOCouplingStructure(scalable_disciplines)
            # add an optimization variable for each coupling variable
            for coupling in coupling_structure.strong_couplings:
                value = 0.5 * np.ones(ScalableProblem.sizes[coupling])
                design_space.add_variable(
                    coupling,
                    ScalableProblem.sizes[coupling],
                    l_b=0.0,
                    u_b=1.0,
                    value=value,
                )

        scenario = MDOScenario(
            ScalableProblem.scalable_disciplines,
            formulation=formulation,
            objective_name="y_4",
            design_space=design_space,
            maximize_objective=True,
        )
        scenario.set_differentiation_method("finite_differences")

        # add disciplinary constraints
        cstr_threshold = 0.5
        for cstr in ["g_1", "g_2", "g_3"]:
            scenario.add_constraint(cstr, "ineq", value=cstr_threshold)

        opt_pb = scenario.formulation.opt_problem

        opt_pb.preprocess_functions()
        for func in opt_pb.get_all_functions():
            func.check_grad(opt_pb.design_space.get_current_value())

    def test_grad(self):
        """Verify the analytical gradients against finite differences."""
        for disc in ScalableProblem.scalable_disciplines:
            assert disc.check_jacobian(
                derr_approx="finite_differences",
                step=1e-6,
                threshold=1e-3,
                linearization_mode="auto",
            )

    def test_get_attributes_to_serialize(self):
        attrs = self.scalable_disciplines[0].get_attributes_to_serialize()
        assert len(attrs) > 5
        for attr in attrs:
            assert isinstance(attr, str)

    def test_group_dep(self):
        hdf_node_path = ScalableProblem.original_disciplines[3].name
        ScalableDiscipline(
            ScalableProblem.scalable_model,
            data=HDF5Cache(HDF_CACHE_PATH, hdf_node_path).export_to_dataset(),
            sizes=ScalableProblem.sizes,
            fill_factor=ScalableProblem.fill_factor,
            group_dep={"y_4": []},
        )
