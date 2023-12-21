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

import re

import matplotlib.pyplot as plt
import pytest

from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.core.grammars.errors import InvalidDataError
from gemseo.post.opt_post_processor import OptPostProcessor
from gemseo.post.opt_post_processor import OptPostProcessorOptionType
from gemseo.problems.analytical.rosenbrock import Rosenbrock
from gemseo.utils.testing.helpers import concretize_classes


@pytest.fixture(scope="module")
def problem() -> Rosenbrock:
    """The Rosenbrock problem."""
    rosenbrock = Rosenbrock()
    OptimizersFactory().execute(rosenbrock, "L-BFGS-B")
    return rosenbrock


class NewOptPostProcessor(OptPostProcessor):
    """A new optimization post processor returning an empty figure."""

    def _plot(self, **options: OptPostProcessorOptionType) -> None:
        self._add_figure(plt.Figure(), "my_figure")


class NewOptPostProcessorWithoutOptionsGrammar(OptPostProcessor):
    """A new optimization post processor without options grammar."""


def test_fig_size(problem):
    """Check the effect of fig_size."""
    post = NewOptPostProcessor(problem)
    figure = post.execute(save=False)["my_figure"]
    assert figure.get_figwidth() == 6.4
    assert figure.get_figheight() == 4.8

    figure = post.execute(save=False, fig_size=(10, 20))["my_figure"]
    assert figure.get_figwidth() == 10
    assert figure.get_figheight() == 20


def test_check_options(problem):
    """Check that an error is raised when using an option that is not in the grammar."""
    with pytest.raises(
        InvalidDataError,
        match=re.escape(
            "Invalid options for post-processor NewOptPostProcessor; "
            "got bar=True, foo='True'."
        ),
    ):
        NewOptPostProcessor(problem).check_options(foo="True", bar=True)


def test_no_option_grammar(problem):
    """Check the error raised when no options grammar."""
    with pytest.raises(
        ValueError,
        match=(
            r"Options grammar for optimization post-processor does not exist, "
            r"expected: .*post "
            r"or .*NewOptPostProcessorWithoutOptionsGrammar_options\.json"
        ),
    ), concretize_classes(NewOptPostProcessorWithoutOptionsGrammar):
        NewOptPostProcessorWithoutOptionsGrammar(problem)
