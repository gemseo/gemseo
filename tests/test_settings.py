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

import pytest
from pydantic import Field
from pydantic import NonNegativeFloat
from pydantic import ValidationError

import gemseo.doe as doe
import gemseo.formulation as formulations
import gemseo.linear as linear_solvers
import gemseo.machine_learning as mlearning
import gemseo.mda as mda
import gemseo.ode as ode
import gemseo.optimization as opt
import gemseo.post as post
import gemseo.uncertainty.distribution as probability_distributions
import gemseo.uncertainty.reliability as reliability
from gemseo.core.base_factory import BaseFactory
from gemseo.doe.core.base_doe_settings import BaseDOESettings
from gemseo.formulation.core.base_settings import BaseFormulationSettings
from gemseo.linear.core.base_linear_solver_settings import BaseLinearSolverSettings
from gemseo.machine_learning.core.model.base_ml_model_settings import (
    BaseMLModelSettings,
)
from gemseo.mda.core.base_settings import BaseMDASettings
from gemseo.ode.core.base_ode_solver_settings import BaseODESolverSettings
from gemseo.optimization.core.base_optimizer_settings import BaseOptimizerSettings
from gemseo.post.core.base_post_settings import BasePostSettings
from gemseo.uncertainty.distribution.core.base_settings import (
    BaseGenericDistributionSettings,
)
from gemseo.uncertainty.reliability.core.base_settings import (
    BaseReliabilityAlgorithmSettings,
)
from gemseo.util.pydantic import BaseSettings


def get_setting_classes(
    BaseSettings: type[BaseSettings],  # noqa: N803
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
    get_setting_classes(BaseDOESettings, "gemseo.doe", doe),
)
def test_doe_settings(module_and_cls):
    module, cls = module_and_cls
    assert getattr(module, cls.__name__) is cls


@pytest.mark.parametrize(
    "module_and_cls",
    get_setting_classes(BaseOptimizerSettings, "gemseo.optimization", opt),
)
def test_opt_settings(module_and_cls):
    module, cls = module_and_cls
    assert getattr(module, cls.__name__) is cls


@pytest.mark.parametrize(
    "module_and_cls",
    get_setting_classes(BaseODESolverSettings, "gemseo.ode", ode),
)
def test_ode_settings(module_and_cls):
    module, cls = module_and_cls
    assert getattr(module, cls.__name__) is cls


@pytest.mark.parametrize(
    "module_and_cls",
    get_setting_classes(BaseLinearSolverSettings, "gemseo.linear", linear_solvers),
)
def test_linear_solver_settings(module_and_cls):
    module, cls = module_and_cls
    assert getattr(module, cls.__name__) is cls


@pytest.mark.parametrize(
    "module_and_cls",
    get_setting_classes(BasePostSettings, "gemseo.post", post),
)
def test_post_settings(module_and_cls):
    module, cls = module_and_cls
    assert getattr(module, cls.__name__) is cls


@pytest.mark.parametrize(
    "module_and_cls",
    get_setting_classes(BaseMLModelSettings, "gemseo.machine_learning", mlearning),
)
def test_machine_learning_settings(module_and_cls):
    module, cls = module_and_cls
    assert getattr(module, cls.__name__) is cls


@pytest.mark.parametrize(
    "module_and_cls",
    get_setting_classes(BaseFormulationSettings, "gemseo.formulation", formulations),
)
def test_formulation_settings(module_and_cls):
    module, cls = module_and_cls
    assert getattr(module, cls.__name__) is cls


def test_settings_inf_serialization():
    """Test the serialization of the settings with infinite values."""

    class Infinity_Settings(BaseSettings):  # noqa: N801
        """A settings class with infinity values for testing serialization."""

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

    settings = Infinity_Settings()

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
    assert getattr(module, cls.__name__) is cls


@pytest.mark.parametrize(
    "module_and_cls",
    get_setting_classes(
        BaseGenericDistributionSettings,
        "gemseo.uncertainty.distribution",
        probability_distributions,
    ),
)
def test_probability_distribution_settings(module_and_cls):
    module, cls = module_and_cls
    assert getattr(module, cls.__name__) is cls


@pytest.mark.parametrize(
    "module_and_cls",
    get_setting_classes(
        BaseReliabilityAlgorithmSettings,
        "gemseo.uncertainty.reliability",
        reliability,
    ),
)
def test_reliability_algorithm_settings(module_and_cls):
    module, cls = module_and_cls
    assert getattr(module, cls.__name__) is cls


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
        not in {
            "BaseGradientBasedAlgorithmSettings",
            "SettingsWithInvalidDefault",
            "BaseDistributionSettings",
            "BaseJointDistributionSettings",
            "OTJointDistributionSettings",
            "SPJointDistributionSettings",
        }
    ],
)
def test_valid_defaults_for_all_settings(cls):
    """Check that all settings classes have valid defaults."""
    cls()
