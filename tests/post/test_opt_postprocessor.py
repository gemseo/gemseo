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

import matplotlib.pyplot as plt
import pytest

from gemseo.algos.opt.factory import OptimizationLibraryFactory
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
