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
#       :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import division, unicode_literals

import pytest
from matplotlib.testing.decorators import image_comparison

from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.post.post_factory import PostFactory
from gemseo.problems.analytical.rosenbrock import Rosenbrock
from gemseo.utils.py23_compat import PY2, Path

POWER_HDF5_PATH = Path(__file__).parent / "power2_opt_pb.h5"


def test_correlations(tmp_wd, pyplot_close_all):
    """Test correlations with the Rosenbrock problem.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
        pyplot_close_all : Fixture that prevents figures aggregation
            with matplotlib pyplot.
    """
    factory = PostFactory()
    if factory.is_available("Correlations"):
        problem = Rosenbrock(20)
        OptimizersFactory().execute(problem, "L-BFGS-B")

        post = factory.execute(
            problem,
            "Correlations",
            save=True,
            n_plots_x=4,
            n_plots_y=4,
            coeff_limit=0.95,
            file_path="correlations_1",
        )
        assert len(post.output_files) == 2
        for outf in post.output_files:
            assert Path(outf).exists()


def test_correlations_import(tmp_wd, pyplot_close_all):
    """Test correlations with imported problem.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
        pyplot_close_all : Fixture that prevents figures aggregation
            with matplotlib pyplot.
    """
    factory = PostFactory()
    if factory.is_available("Correlations"):
        problem = OptimizationProblem.import_hdf(str(POWER_HDF5_PATH))
        post = factory.execute(
            problem,
            "Correlations",
            save=True,
            n_plots_x=4,
            n_plots_y=4,
            coeff_limit=0.999,
            file_path="correlations_2",
        )
        assert len(post.output_files) == 1
        for outf in post.output_files:
            assert Path(outf).exists()


def test_correlations_func_name_error():
    """Test ValueError for non-existent function."""
    factory = PostFactory()
    if factory.is_available("Correlations"):
        problem = Rosenbrock(20)
        OptimizersFactory().execute(problem, "L-BFGS-B")

        with pytest.raises(
            ValueError, match=r"The following elements are not" r" functions: .*toto.*"
        ):
            factory.execute(
                problem, "Correlations", save=False, show=False, func_names=["toto"]
            )


@pytest.mark.skipif(PY2, reason="image comparison does not work with python 2")
@pytest.mark.parametrize(
    "func_names,baseline_images",
    [(["pow2", "ineq1"], ["pow2_ineq1"]), ([], ["all_func"])],
)
@image_comparison(None, extensions=["png"])
def test_correlations_func_names(tmp_wd, baseline_images, func_names, pyplot_close_all):
    """Test func_names filter.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
        pyplot_close_all : Fixture that prevents figures aggregation
            with matplotlib pyplot.
    """
    factory = PostFactory()
    if factory.is_available("Correlations"):
        problem = OptimizationProblem.import_hdf(str(POWER_HDF5_PATH))
        post = factory.execute(
            problem,
            "Correlations",
            func_names=func_names,
            save=False,
            file_extension="png",
            n_plots_x=4,
            n_plots_y=4,
            coeff_limit=0.99,
            file_path="correlations",
        )
        post.figures
