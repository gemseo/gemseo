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

"""Regenerate the figures of the "What's new in GEMSEO 7" page.

The page `docs/software/whats_new_7.md` embeds figures that are committed to the
repository instead of being generated at build time:
the Pareto front of its multi-objective scenario,
the pair plot of its dependent random variables,
the Rosenblatt transformation of these variables,
the pair plot of the iris dataset by species
and a banner combining these figures, also intended for communications.
This script regenerates them all from the snippets of the page.

Usage:
    python tools/regenerate_whats_new_7_figures.py [DIRECTORY_PATH]

where `DIRECTORY_PATH` defaults to `docs/assets/images/whats_new_7`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.image import imread
from numpy import array
from openturns import CorrelationMatrix
from openturns import NormalCopula

from gemseo.discipline import AnalyticDiscipline
from gemseo.doe import PYDOE_LHS_Settings
from gemseo.optimization import MNBI_Settings
from gemseo.optimization import SLSQP_Settings
from gemseo.post import ParetoFront_Settings
from gemseo.post.dataset import PairPlot
from gemseo.post.dataset.pair_plot_settings import PairPlot_Settings
from gemseo.problem.dataset.iris import create_iris_dataset
from gemseo.scenario import EvaluationScenario
from gemseo.scenario import MDOScenario
from gemseo.space import DesignSpace
from gemseo.space import RandomSpace
from gemseo.uncertainty.distribution import OTNormalDistribution_Settings


def save_pareto_front(file_path: Path) -> None:
    """Save the Pareto front of the multi-objective scenario.

    Args:
        file_path: The path of the PNG file.
    """
    cost_model = AnalyticDiscipline({"cost": "4*x**2+4*y**2"})
    performance_model = AnalyticDiscipline({"performance": "50-(x-5)**2-(y-5)**2"})
    design_space = DesignSpace()
    design_space.add_real_variable("x", lower_bound=0.0, upper_bound=5.0, value=1.0)
    design_space.add_real_variable("y", lower_bound=0.0, upper_bound=3.0, value=1.0)
    scenario = MDOScenario([cost_model, performance_model], design_space)
    scenario.add_objective("cost")
    scenario.add_objective("performance", minimize=False)
    scenario.execute(
        MNBI_Settings(
            max_iter=1000,
            n_sub_optim=20,
            sub_optim_algo_settings=SLSQP_Settings(max_iter=100),
        )
    )
    scenario.post_process(
        ParetoFront_Settings(
            objectives_labels=("cost", "-performance"),
            fig_size=(6.4, 6.4),
            save=True,
            show=False,
            file_path=file_path,
        )
    )


def create_random_space() -> RandomSpace:
    """Create the random space of the dependent random variables.

    Returns:
        The random space.
    """
    space = RandomSpace()
    space.add_variable("x", OTNormalDistribution_Settings(mu=0.0, sigma=1.0))
    space.add_variable("y", OTNormalDistribution_Settings(mu=0.0, sigma=1.0))
    correlation = CorrelationMatrix(2)
    correlation[0, 1] = 0.8
    space.add_copula(("x", "y"), NormalCopula(correlation))
    return space


def save_copula_pair_plot(file_path: Path) -> None:
    """Save the pair plot of the dependent random variables.

    Args:
        file_path: The path of the PNG file.
    """
    scenario = EvaluationScenario(
        [AnalyticDiscipline({"z": "x+y"})], create_random_space()
    )
    scenario.add_observable("z")
    scenario.execute(PYDOE_LHS_Settings(n_samples=100))
    PairPlot(
        scenario.to_dataset(),
        PairPlot_Settings(variable_names=("x", "y"), use_ranks=True),
    ).execute(save=True, show=False, file_path=file_path)


def save_rosenblatt_transformation(file_path: Path) -> None:
    """Save the samples of the dependent random variables and their uniform images.

    Args:
        file_path: The path of the PNG file.
    """
    distribution = create_random_space().variables.distribution
    samples = distribution.compute_samples(500)
    uniform_samples = array([distribution.map_to_uniform(sample) for sample in samples])
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6.4, 3.2))
    ax1.scatter(samples[:, 0], samples[:, 1], s=4)
    ax1.set_title("Dependent normal variables")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax2.scatter(uniform_samples[:, 0], uniform_samples[:, 1], s=4)
    ax2.set_title("Image by map_to_uniform()")
    ax2.set_xlabel("$u_x$")
    ax2.set_ylabel("$u_y$")
    for ax in (ax1, ax2):
        ax.set_box_aspect(1)
        ax.grid()

    fig.tight_layout()
    fig.savefig(file_path)
    plt.close(fig)


def save_classifier_pair_plot(file_path: Path) -> None:
    """Save the pair plot of the iris dataset by species.

    Args:
        file_path: The path of the PNG file.
    """
    settings = PairPlot_Settings(
        variable_names=("sepal_length", "sepal_width", "petal_width", "specy"),
        classifier="specy",
        use_scatter=False,
        use_kde=True,
    )
    PairPlot(create_iris_dataset(), settings).execute(
        save=True, show=False, file_path=file_path
    )


def save_banner(file_path: Path, panel_paths: list[Path]) -> None:
    """Save a banner combining figures side by side.

    Args:
        file_path: The path of the PNG file.
        panel_paths: The paths of the PNG files of the figures.
    """
    fig, axes = plt.subplots(
        ncols=len(panel_paths), figsize=(4 * len(panel_paths), 4.6)
    )
    for ax, panel_path in zip(axes, panel_paths, strict=True):
        ax.imshow(imread(panel_path))
        ax.set_axis_off()

    fig.suptitle("GEMSEO 7", fontsize=28, fontweight="bold")
    fig.tight_layout()
    fig.savefig(file_path, dpi=150)
    plt.close(fig)


def main(directory_path: Path) -> None:
    """Regenerate the figures.

    Args:
        directory_path: The path of the directory to save the figures.
    """
    directory_path.mkdir(parents=True, exist_ok=True)
    pareto_front = directory_path / "pareto_front.png"
    copula = directory_path / "pair_plot_copula.png"
    rosenblatt = directory_path / "rosenblatt_transformation.png"
    classifier = directory_path / "pair_plot_classifier.png"
    save_pareto_front(pareto_front)
    save_copula_pair_plot(copula)
    save_rosenblatt_transformation(rosenblatt)
    save_classifier_pair_plot(classifier)
    save_banner(directory_path / "banner.png", [pareto_front, copula, classifier])


if __name__ == "__main__":
    main(
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).parents[1] / "docs" / "assets" / "images" / "whats_new_7"
    )
