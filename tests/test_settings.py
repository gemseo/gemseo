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

from math import inf
from typing import TYPE_CHECKING
from typing import ClassVar

import pytest
from pydantic import Field
from pydantic import NonNegativeFloat
from pydantic import ValidationError

import gemseo.settings.doe as doe
import gemseo.settings.formulations as formulations
import gemseo.settings.linear_solvers as linear_solvers
import gemseo.settings.mda as mda
import gemseo.settings.mlearning as mlearning
import gemseo.settings.ode as ode
import gemseo.settings.opt as opt
import gemseo.settings.post as post
import gemseo.settings.probability_distributions as probability_distributions
from gemseo.algos.doe.base_doe_settings import BaseDOESettings
from gemseo.algos.linear_solvers.base_linear_solver_settings import (
    BaseLinearSolverSettings,
)
from gemseo.algos.ode.base_ode_solver_settings import BaseODESolverSettings
from gemseo.algos.opt.base_optimizer_settings import BaseOptimizerSettings
from gemseo.core.base_factory import BaseFactory
from gemseo.formulations.base_formulation_settings import BaseFormulationSettings
from gemseo.mda.base_mda_settings import BaseMDASettings
from gemseo.mlearning.core.algos.ml_algo_settings import BaseMLAlgoSettings
from gemseo.post.base_post_settings import BasePostSettings
from gemseo.settings.base_settings import BaseSettings
from gemseo.uncertainty.distributions.base_distribution_settings import (
    BaseDistribution_Settings,
)

if TYPE_CHECKING:
    from gemseo.algos.base_algorithm_settings import BaseAlgorithmSettings


def get_setting_classes(
    BaseSettings: type[BaseAlgorithmSettings],  # noqa: N803
    package_name: str,
    module_,
) -> list[str]:
    """Return the settings classes given a type of algorithms.

    Args:
        BaseSettings: The base class specific to the type of algorithms.
        package_name: The name of the package.
        module_: The module of settings.

    Returns:
        The settings classes.
    """

    class SettingsFactory(BaseFactory):
        _CLASS = BaseSettings
        _PACKAGE_NAMES = (package_name,)

        @property
        def classes(self) -> list[str]:
            return [
                self.get_class(name)
                for name in super().class_names
                if not name.startswith("Base")
            ]

    for cls in SettingsFactory().classes:
        # Prevent failure when testing in environments with plugins.
        if cls.__module__.startswith("gemseo."):
            yield module_, cls


class SettingsWithInvalidDefault(BaseSettings):
    name: str = Field(default=123)


@pytest.mark.parametrize(
    "module_and_cls",
    get_setting_classes(BaseDOESettings, "gemseo.algos.doe", doe),
)
def test_doe_settings(module_and_cls):
    module, cls = module_and_cls
    assert cls in module.__dict__.values()


@pytest.mark.parametrize(
    "module_and_cls",
    get_setting_classes(BaseOptimizerSettings, "gemseo.algos.opt", opt),
)
def test_opt_settings(module_and_cls):
    module, cls = module_and_cls
    assert cls in module.__dict__.values()


@pytest.mark.parametrize(
    "module_and_cls",
    get_setting_classes(BaseODESolverSettings, "gemseo.algos.ode", ode),
)
def test_ode_settings(module_and_cls):
    module, cls = module_and_cls
    assert cls in module.__dict__.values()


@pytest.mark.parametrize(
    "module_and_cls",
    get_setting_classes(
        BaseLinearSolverSettings, "gemseo.algos.linear_solvers", linear_solvers
    ),
)
def test_linear_solver_settings(module_and_cls):
    module, cls = module_and_cls
    assert cls in module.__dict__.values()


@pytest.mark.parametrize(
    "module_and_cls",
    get_setting_classes(BasePostSettings, "gemseo.post", post),
)
def test_post_settings(module_and_cls):
    module, cls = module_and_cls
    assert cls in module.__dict__.values()


@pytest.mark.parametrize(
    "module_and_cls",
    get_setting_classes(BaseMLAlgoSettings, "gemseo.mlearning", mlearning),
)
def test_machine_learning_settings(module_and_cls):
    module, cls = module_and_cls
    assert cls in module.__dict__.values()


@pytest.mark.parametrize(
    "module_and_cls",
    get_setting_classes(BaseFormulationSettings, "gemseo.formulations", formulations),
)
def test_formulation_settings(module_and_cls):
    module, cls = module_and_cls
    assert cls in module.__dict__.values()


def test_settings_inf_serialization():
    """Test the serialization of the settings with infinite values."""

    class InfinitySettings(BaseSettings):
        """A settings class with infinity values for testing serialization."""

        _TARGET_CLASS_NAME: ClassVar[str] = "InfinityTest"

        # Field with positive infinity
        pos_inf: NonNegativeFloat = Field(
            default=inf,
            description="A field with positive infinity as default value.",
        )

        # Field with negative infinity (using float)
        neg_inf: float = Field(
            default=float("-inf"),
            description="A field with negative infinity as default value.",
        )

        # Regular field for comparison
        regular_value: float = Field(
            default=123.45,
            description="A regular float value for comparison.",
        )

    settings = InfinitySettings()

    json_data = settings.model_dump_json(indent=4)

    expected_json = """
{
    "pos_inf": null,
    "neg_inf": null,
    "regular_value": 123.45
}
"""

    assert json_data == expected_json.strip()

    # Test dictionary serialization
    dict_data = settings.model_dump()

    # In dictionary form, infinity values should remain as infinity
    assert dict_data["pos_inf"] == inf
    assert dict_data["neg_inf"] == float("-inf")
    assert dict_data["regular_value"] == 123.45


@pytest.mark.parametrize(
    "module_and_cls",
    get_setting_classes(BaseMDASettings, "gemseo.mda", mda),
)
def test_mda_settings(module_and_cls):
    module, cls = module_and_cls
    assert cls in module.__dict__.values()


@pytest.mark.parametrize(
    "module_and_cls",
    get_setting_classes(
        BaseDistribution_Settings,
        "gemseo.uncertainty.distributions",
        probability_distributions,
    ),
)
def test_probability_distribution_settings(module_and_cls):
    module, cls = module_and_cls
    assert cls in module.__dict__.values()


def test_default_validation():
    """Check that default values are validated."""
    with pytest.raises(ValidationError):
        SettingsWithInvalidDefault()


@pytest.mark.parametrize(
    "cls",
    [
        cls
        for cls in BaseSettings.__subclasses__()
        if cls.__name__
        not in {"BaseGradientBasedAlgorithmSettings", "SettingsWithInvalidDefault"}
    ],
)
def test_valid_defaults_for_all_settings(cls):
    """Check that all settings classes have valid defaults."""
    cls()
