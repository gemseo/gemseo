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

from dataclasses import fields
from unittest.mock import patch

import matplotlib.pyplot as plt
import pytest

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt.factory import OptimizationLibraryFactory
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.datasets.optimization_dataset import OptimizationDataset
from gemseo.datasets.optimization_metadata import OptimizationMetadata
from gemseo.post.base_post import BasePost
from gemseo.post.base_post_settings import BasePostSettings
from gemseo.problems.optimization.rosenbrock import Rosenbrock


@pytest.fixture(scope="module")
def problem() -> Rosenbrock:
    """The Rosenbrock problem."""
    rosenbrock = Rosenbrock()
    OptimizationLibraryFactory().execute(rosenbrock, algo_name="L-BFGS-B")
    return rosenbrock


class NewBasePost(BasePost[BasePostSettings]):
    """A new optimization post processor returning an empty figure."""

    Settings = BasePostSettings

    def _plot(self, settings: BasePostSettings) -> None:
        self._add_figure(plt.Figure(), "my_figure")


class NewBasePostWithoutOptionsGrammar(BasePost):
    """A new optimization post processor without options grammar."""


def test_fig_size(problem) -> None:
    """Check the effect of fig_size."""
    post = NewBasePost(problem)
    figure = post.execute(save=False)["my_figure"]
    assert figure.get_figwidth() == 11.0
    assert figure.get_figheight() == 11.0

    figure = post.execute(save=False, fig_size=(10, 20))["my_figure"]
    assert figure.get_figwidth() == 10
    assert figure.get_figheight() == 20


def test_settings_as_pydantic_model(problem):
    """Check that settings can be passed as a Pydantic model."""
    post = NewBasePost(problem)
    settings = BasePostSettings(save=False, fig_size=(10, 20))
    figure = post.execute(settings_model=settings)["my_figure"]
    assert figure.get_figwidth() == 10.0
    assert figure.get_figheight() == 20.0


@pytest.mark.parametrize("save", [True, False])
@pytest.mark.parametrize("show", [True, False])
def test_show_close(problem, save, show) -> None:
    """Check the use of show and close."""
    post = NewBasePost(problem)
    with (
        patch("gemseo.utils.matplotlib_figure.plt.show") as show_,
        patch("gemseo.utils.matplotlib_figure.plt.close") as close_,
        patch("gemseo.post.base_post.save_show_figure") as save_show_figure,
    ):
        post.execute(save=save, show=show)

    save_show_figure.assert_called_once()
    assert save_show_figure.call_args.args[1] is False
    assert save_show_figure.call_args.kwargs["close"] is False

    if save:
        close_.assert_called_once()
    else:
        close_.assert_not_called()

    if show:
        show_.assert_called_once()
    else:
        show_.assert_not_called()


def test_dataset_as_input_for_post(problem):
    """Test the execution of base post when a dataset is given as an input."""
    dataset = problem.to_dataset()
    post = NewBasePost(dataset)

    assert isinstance(post._dataset, OptimizationDataset)
    assert post.database is None


def test_execute_error_with_no_dataset():
    """Tests that an error is raised when executing a post without a dataset."""

    problem = OptimizationProblem(DesignSpace())
    problem.objective = MDOFunction(lambda x: x, "f")
    with pytest.raises(
        ValueError,
        match=r"The post-processor NewBasePost cannot"
        r" be created because there is no dataset "
        r"to plot.",
    ):
        NewBasePost(problem)


def test_execute_error_with_empty_dataset():
    """Tests that an error is raised when executing a post with an empty dataset."""
    dataset = OptimizationDataset()
    dataset.misc["optimization_metadata"] = OptimizationMetadata(**{
        field.name: None for field in fields(OptimizationMetadata)
    })
    post = NewBasePost(dataset)

    with pytest.raises(
        ValueError,
        match=r"The post-processor NewBasePost cannot "
        r"be solved because the optimization problem "
        r"was not solved.",
    ):
        post.execute()
