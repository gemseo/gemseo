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
#                           documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""A factory of post-processors."""

from __future__ import annotations

from typing import Any

from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.core.base_factory import BaseFactory
from gemseo.post.base_post import BasePost
from gemseo.post.base_post import OptPostProcessorOptionType


class OptPostProcessorFactory(BaseFactory[BasePost]):
    """A factory of post-processors."""

    _CLASS = BasePost
    _MODULE_NAMES = ("gemseo.post",)

    def __init__(self) -> None:  # noqa:D107
        super().__init__()
        # TODO: API: is this really useful? make it private?
        self.executed_post = []

    @property
    def posts(self) -> list[str]:
        """The available post processors."""
        return self.class_names

    def get_settings(self, class_name: str) -> type[BasePost.Settings]:
        """Return the settings of a class.

        Args:
            class_name: The name of a class.

        Returns:
            The settings of the class.
        """
        return self.get_class(class_name).Settings

    def create(
        self,
        class_name: str,
        opt_problem: OptimizationProblem,
        **options: OptPostProcessorOptionType,
    ) -> BasePost:
        """Create a post-processor from its class name.

        Args:
            class_name: The name of the post-processor.
            opt_problem: The optimization problem to be post-processed.
            **options: The options of the post-processor.
        """
        return super().create(class_name, opt_problem=opt_problem, **options)

    def execute(
        self,
        opt_problem: str | OptimizationProblem,
        post_name: str,
        **options: Any,
    ) -> BasePost:
        """Post-process an optimization problem.

        Args:
            opt_problem: The optimization problem to be post-processed.
            post_name: The name of the post-processor.
            **options: The options of the post-processor.

        Returns:
            The post-processor.
        """
        if isinstance(opt_problem, str):
            opt_problem = OptimizationProblem.from_hdf(opt_problem)
        post = self.create(post_name, opt_problem)
        post.execute(**options)
        self.executed_post.append(post)
        return post

    def list_generated_plots(self) -> set[str]:
        """The generated plot files."""
        plots = []
        for post in self.executed_post:
            plots.extend(post.output_files)
        return set(plots)
