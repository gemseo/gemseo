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
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import pytest
from gemseo.api import create_discipline
from gemseo.core.discipline import MDODiscipline
from gemseo.mda.gauss_seidel import MDAGaussSeidel
from gemseo.problems.sellar.sellar import Sellar1
from gemseo.problems.sellar.sellar import Sellar2
from gemseo.problems.sellar.sellar import SellarSystem
from gemseo.problems.sobieski.process.mda_gauss_seidel import SobieskiMDAGaussSeidel
from gemseo.utils.testing import image_comparison
from numpy import array
from numpy import isclose


@image_comparison(["sobieski"])
def test_sobieski(tmp_wd, pyplot_close_all):
    """Test the execution of Gauss-Seidel on Sobieski."""
    mda = SobieskiMDAGaussSeidel(tolerance=1e-12, max_mda_iter=30)
    mda.default_inputs["x_shared"] += 0.1
    mda.execute()
    mda.default_inputs["x_shared"] += 0.1
    mda.warm_start = True
    mda.execute()

    assert mda.residual_history[-1] < 1e-4

    mda.plot_residual_history(save=False)


def test_expected_workflow():
    """Test MDA GaussSeidel workflow should be disciplines sequence."""
    disc1 = MDODiscipline()
    disc2 = MDODiscipline()
    disc3 = MDODiscipline()
    disciplines = [disc1, disc2, disc3]

    mda = MDAGaussSeidel(disciplines)
    expected = (
        "{MDAGaussSeidel(None), [MDODiscipline(None), "
        "MDODiscipline(None), MDODiscipline(None), ], }"
    )
    assert str(mda.get_expected_workflow()) == expected


def test_self_coupled():
    for plus_y in [False, True]:
        sc_disc = SelfCoupledDisc(plus_y)
        mda = MDAGaussSeidel([sc_disc], tolerance=1e-14, max_mda_iter=40)
        _ = mda.execute()
        # assert abs(out["y"] - 2. / 3.) < 1e-6

        mda.add_differentiated_inputs(["x"])
        mda.add_differentiated_outputs(["o"])
        jac1 = mda.linearize()

        mda.set_jacobian_approximation()
        mda.cache.clear()
        jac2 = mda.linearize()
        assert abs(jac1["o"]["x"][0, 0] - jac2["o"]["x"][0, 0]) < 1e-3


@pytest.mark.parametrize("over_relax_factor", [1.0, 0.8, 1.1, 1.2, 1.5])
def test_over_relaxation(over_relax_factor):
    discs = create_discipline(
        [
            "SobieskiPropulsion",
            "SobieskiStructure",
            "SobieskiAerodynamics",
            "SobieskiMission",
        ]
    )
    tolerance = 1e-14
    mda = MDAGaussSeidel(
        discs,
        tolerance=tolerance,
        max_mda_iter=100,
        over_relax_factor=over_relax_factor,
    )
    mda.execute()
    assert mda.residual_history[-1] <= tolerance

    assert mda.local_data[mda.RESIDUALS_NORM][0] < tolerance


class SelfCoupledDisc(MDODiscipline):
    def __init__(self, plus_y=False):
        MDODiscipline.__init__(self)
        self.input_grammar.update(["y", "x"])
        self.output_grammar.update(["y", "o"])
        self.default_inputs["y"] = array([0.25])
        self.default_inputs["x"] = array([0.0])
        self.coeff = 1.0
        if not plus_y:
            self.coeff = -1.0

    def _run(self):

        self.local_data["y"] = (
            1.0 + self.coeff * 0.5 * self.local_data["y"] + self.local_data["x"]
        )
        self.local_data["o"] = self.local_data["y"] + self.local_data["x"]

    def _compute_jacobian(self, inputs=None, outputs=None):
        self.jac = {
            "y": {"y": self.coeff * array([[0.5]]), "x": array([[1.0]])},
            "o": {"y": array([[1.0]]), "x": array([[1.0]])},
        }


def test_log_convergence():
    """Check that the boolean log_convergence is correctly set."""
    disciplines = [Sellar1(), Sellar2(), SellarSystem()]
    mda = MDAGaussSeidel(disciplines)
    assert not mda.log_convergence
    mda = MDAGaussSeidel(disciplines, log_convergence=True)
    assert mda.log_convergence


def test_parallel_doe(generate_parallel_doe_data):
    """Test the execution of GaussSeidel in parallel.

    Args:
        generate_parallel_doe_data: Fixture that returns the optimum solution to
            a parallel DOE scenario for a particular `main_mda_name`.
    """
    obj = generate_parallel_doe_data("MDAGaussSeidel")
    assert isclose(array([-obj]), array([608.185]), atol=1e-3)


@pytest.mark.parametrize(
    "baseline_images,n_iterations,logscale",
    [
        (["all_iter_default_log"], None, None),
        (["all_iter_modified_log"], None, [1e-15, 10.0]),
        (["n_iter_larger_than_history"], 50, None),
        (["eight_iter_default_log"], 8, None),
        (["eight_iter_modified_log"], 8, [1e-15, 10.0]),
    ],
)
@image_comparison(None, tol=0.098)
def test_plot_residual_history(
    baseline_images, n_iterations, logscale, caplog, pyplot_close_all
):
    """Test the residual history plot.

    Args:
        baseline_images: The reference images for the test.
        n_iterations: The number of iterations to plot.
        logscale: The limits of the ``y`` axis.
        caplog: Fixture to access and control log capturing.
        pyplot_close_all: Fixture that prevents figures aggregation
            with matplotlib pyplot.
    """
    mda = SobieskiMDAGaussSeidel(tolerance=1e-12, max_mda_iter=10)
    mda.execute()
    mda.plot_residual_history(save=False, n_iterations=n_iterations, logscale=logscale)

    if n_iterations == 50:
        assert (
            "Requested 50 iterations but the residual history contains only "
            f"{len(mda.residual_history)}, plotting all the residual history."
            in caplog.text
        )
