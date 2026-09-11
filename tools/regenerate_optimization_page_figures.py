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

"""Regenerate the figures of the documentation page about optimization and DOE.

The page `docs/mdo/optimization.md` embeds figures that are committed to the
repository instead of being generated at build time, namely the optimization history
of the simple analytic problem it solves and the samplings of the DOE algorithms it
lists. This script regenerates them all.

The optimization history figure is drawn from the very problem the page describes.
The DOE figures are not: the page describes only the scalar space `x` in `[-2, 2]` and
passes `n_samples=10` in its only DOE snippet, while these figures need a space with
enough variables for a pair plot. The design space and the settings used for them,
`N_SAMPLES` included, are the ones defined below and belong to this script alone; do not
look for them on the page. They draw more points than the figures they replace.

Usage:
    python tools/regenerate_optimization_page_figures.py [DIRECTORY_PATH]

where `DIRECTORY_PATH` defaults to `docs/assets/images/doe`.
"""

from __future__ import annotations

import sys
from pathlib import Path

from numpy import cos
from numpy import exp
from numpy import ones
from numpy import sin

from gemseo import create_design_space
from gemseo import execute_post
from gemseo.core.function.array_function import ArrayFunction
from gemseo.dataset.dataset import Dataset
from gemseo.doe import OT_AXIAL_Settings
from gemseo.doe import OT_COMPOSITE_Settings
from gemseo.doe import OT_FACTORIAL_Settings
from gemseo.doe import OT_FAURE_Settings
from gemseo.doe import OT_HALTON_Settings
from gemseo.doe import OT_HASELGROVE_Settings
from gemseo.doe import OT_LHS_Settings
from gemseo.doe import OT_LHSC_Settings
from gemseo.doe import OT_MONTE_CARLO_Settings
from gemseo.doe import OT_RANDOM_Settings
from gemseo.doe import OT_SOBOL_Settings
from gemseo.doe import PYDOE_BBDESIGN_Settings
from gemseo.doe import PYDOE_FULLFACT_Settings
from gemseo.doe import PYDOE_LHS_Settings
from gemseo.doe.factory import DOELibraryFactory
from gemseo.optimization import L_BFGS_B_Settings
from gemseo.optimization import OptimizationProblem
from gemseo.optimization.factory import OptimizationLibraryFactory
from gemseo.post import OptHistoryView_Settings
from gemseo.post.dataset.pair_plot import PairPlot
from gemseo.post.dataset.pair_plot_settings import PairPlot_Settings

DEFAULT_DIRECTORY_PATH = Path("docs/assets/images/doe")
"""The directory where the page reads its figures."""

N_SAMPLES = 64
"""The number of samples of the DOE algorithms driven by a number of samples."""

SEED = 1
"""The seed of the random DOE algorithms, so that the figures do not change."""

LEVELS = [0.25, 0.5, 0.75, 1.0]
"""The levels of the DOE algorithms driven by levels."""

CENTERS = [0.5, 0.5, 0.5]
"""The centers of the DOE algorithms driven by levels."""

FILE_NAME_TO_DOE_SETTINGS = {
    "fullfact_pyDOE": PYDOE_FULLFACT_Settings(levels=[4, 4, 4]),
    "bbdesign_pyDOE": PYDOE_BBDESIGN_Settings(),
    "lhs_pyDOE": PYDOE_LHS_Settings(n_samples=N_SAMPLES, seed=SEED),
    "axial_openturns": OT_AXIAL_Settings(levels=LEVELS, centers=CENTERS),
    "composite_openturns": OT_COMPOSITE_Settings(levels=LEVELS, centers=CENTERS),
    "factorial_openturns": OT_FACTORIAL_Settings(levels=LEVELS, centers=CENTERS),
    "faure_openturns": OT_FAURE_Settings(n_samples=N_SAMPLES, seed=SEED),
    "halton_openturns": OT_HALTON_Settings(n_samples=N_SAMPLES, seed=SEED),
    "haselgrove_openturns": OT_HASELGROVE_Settings(n_samples=N_SAMPLES, seed=SEED),
    "sobol_openturns": OT_SOBOL_Settings(n_samples=N_SAMPLES, seed=SEED),
    "mc_openturns": OT_MONTE_CARLO_Settings(n_samples=N_SAMPLES, seed=SEED),
    "lhsc_openturns": OT_LHSC_Settings(n_samples=N_SAMPLES, seed=SEED),
    "lhs_openturns": OT_LHS_Settings(n_samples=N_SAMPLES, seed=SEED),
    "random_openturns": OT_RANDOM_Settings(n_samples=N_SAMPLES, seed=SEED),
}
"""The settings of the DOE algorithms, per name of the figure that draws its samples."""


def create_optimization_history_figure(directory_path: Path) -> None:
    r"""Draw the optimization history of the simple analytic problem of the page.

    The problem minimizes $\sin(x)-\exp(x)$ over $[-2,2]$ with L-BFGS-B, as the page
    does. Only the figure about the objective is kept, as it is the only one the page
    embeds.

    Args:
        directory_path: The directory where the figure is written.
    """
    design_space = create_design_space()
    design_space.add_variable(
        "x", size=1, lower_bound=-2.0, upper_bound=2.0, value=-0.5 * ones(1)
    )
    problem = OptimizationProblem(design_space)
    problem.objective = ArrayFunction(sin, name="f_1", jac=cos, expr="sin(x)") - (
        ArrayFunction(exp, name="f_2", jac=exp, expr="exp(x)")
    )
    OptimizationLibraryFactory().execute(
        problem, L_BFGS_B_Settings(normalize_design_space=True)
    )
    file_path = directory_path / "simple_opt"
    execute_post(
        problem, OptHistoryView_Settings(save=True, show=False, file_path=file_path)
    )
    for name in ["variables", "x_xstar"]:
        file_path.with_name(f"{file_path.name}_{name}.png").unlink(missing_ok=True)

    file_path.with_name(f"{file_path.name}_objective.png").replace(
        file_path.with_suffix(".png")
    )


def create_doe_figures(directory_path: Path) -> None:
    """Draw the samplings of the DOE algorithms of the page.

    Args:
        directory_path: The directory where the figures are written.
    """
    design_space = create_design_space()
    design_space.add_variable("x_1", lower_bound=0.1, upper_bound=0.4)
    design_space.add_variable("x_2", lower_bound=0.75, upper_bound=1.25)
    design_space.add_variable("x_3", lower_bound=0.75, upper_bound=1.25)
    factory = DOELibraryFactory()
    for file_name, settings in FILE_NAME_TO_DOE_SETTINGS.items():
        library = factory.create(settings.target_class_name)
        samples = library.sample_space(design_space, settings=settings)
        dataset = Dataset.from_array(samples, variable_names=["x_1", "x_2", "x_3"])
        plot = PairPlot(dataset, PairPlot_Settings(use_kde=True))
        plot.execute(save=True, show=False, file_path=directory_path / file_name)


if __name__ == "__main__":
    directory_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIRECTORY_PATH
    create_optimization_history_figure(directory_path)
    create_doe_figures(directory_path)
