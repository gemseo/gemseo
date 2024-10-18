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
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import gemseo.settings.doe as doe
import gemseo.settings.linear_solvers as linear_solvers
import gemseo.settings.mlearning as mlearning
import gemseo.settings.ode as ode
import gemseo.settings.opt as opt
import gemseo.settings.post as post
from gemseo.algos.doe.base_doe_library_settings import BaseDOELibrarySettings
from gemseo.algos.linear_solvers.base_linear_solver_settings import (
    BaseLinearSolverLibrarySettings,
)
from gemseo.algos.ode.base_ode_solver_library_settings import (
    BaseODESolverLibrarySettings,
)
from gemseo.algos.opt.base_optimization_library_settings import (
    BaseOptimizationLibrarySettings,
)
from gemseo.core.base_factory import BaseFactory
from gemseo.mlearning.core.algos.ml_algo_settings import BaseMLAlgoSettings
from gemseo.post.base_post_settings import BasePostSettings

if TYPE_CHECKING:
    from gemseo.algos.base_algorithm_library_settings import (
        BaseAlgorithmLibrarySettings,
    )


def get_setting_class_names(
    BaseSettings: type[BaseAlgorithmLibrarySettings],  # noqa: N803
    package_name: str,
    module_,
) -> list[str]:
    """Return the names of the settings given a type of algorithms.

    Args:
        BaseSettings: The base class specific to the type of algorithms.
        package_name: The name of the package.
        module_: The module of settings.

    Returns:
        The names of the settings.
    """

    class SettingsFactory(BaseFactory):
        _CLASS = BaseSettings
        _PACKAGE_NAMES = (package_name,)

        @property
        def class_names(self) -> list[str]:
            return [name for name in super().class_names if not name.startswith("Base")]

    for class_name in SettingsFactory().class_names:
        yield module_, class_name


@pytest.mark.parametrize(
    "class_name",
    get_setting_class_names(BaseDOELibrarySettings, "gemseo.algos.doe", doe),
)
def test_doe_settings(class_name):
    _module, class_name = class_name
    getattr(_module, class_name)


@pytest.mark.parametrize(
    "class_name",
    get_setting_class_names(BaseOptimizationLibrarySettings, "gemseo.algos.opt", opt),
)
def test_opt_settings(class_name):
    _module, class_name = class_name
    getattr(_module, class_name)


@pytest.mark.parametrize(
    "class_name",
    get_setting_class_names(BaseODESolverLibrarySettings, "gemseo.algos.ode", ode),
)
def test_ode_settings(class_name):
    _module, class_name = class_name
    getattr(_module, class_name)


@pytest.mark.parametrize(
    "class_name",
    get_setting_class_names(
        BaseLinearSolverLibrarySettings, "gemseo.algos.linear_solvers", linear_solvers
    ),
)
def test_linear_solver_settings(class_name):
    _module, class_name = class_name
    getattr(_module, class_name)


@pytest.mark.parametrize(
    "class_name",
    get_setting_class_names(BasePostSettings, "gemseo.post", post),
)
def test_post_settings(class_name):
    _module, class_name = class_name
    getattr(_module, class_name)


@pytest.mark.parametrize(
    "class_name",
    get_setting_class_names(BaseMLAlgoSettings, "gemseo.mlearning", mlearning),
)
def test_machine_learning_settings(class_name):
    _module, class_name = class_name
    getattr(_module, class_name)
