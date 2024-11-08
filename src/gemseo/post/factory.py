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

from typing import TYPE_CHECKING
from typing import Any

from gemseo.core.base_factory import BaseFactory
from gemseo.post.base_post import BasePost

if TYPE_CHECKING:
    from gemseo.algos.optimization_problem import OptimizationProblem
    from gemseo.post.base_post_settings import BasePostSettings


class PostFactory(BaseFactory[BasePost[Any]]):
    """A factory of post-processors."""

    _CLASS = BasePost
    _PACKAGE_NAMES = ("gemseo.post",)

    def execute(
        self,
        opt_problem: OptimizationProblem,
        settings_model: BasePostSettings | None = None,
        **settings: Any,
    ) -> BasePost[Any]:
        """Post-process an optimization problem.

        Args:
            opt_problem: The optimization problem to be post-processed.
            settings_model: The post-processor settings as a Pydantic model.
                If ``None``, use ``**settings``.
            **settings: The post-processor settings,
                including the algorithm name (use the keyword ``"post_name"``).
                These arguments are ignored when ``settings_model`` is not ``None``.

        Returns:
            The post-processor.
        """
        if settings_model is None:
            post_name = settings.pop("post_name")
        else:
            post_name = settings_model._TARGET_CLASS_NAME
        post = self.create(post_name, opt_problem)
        post.execute(settings_model=settings_model, **settings)
        return post
