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

import re
from collections import Counter
from logging import INFO
from logging import NullHandler
from logging import getLogger
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import ClassVar

import pytest
from pydantic import ValidationError

from gemseo import configuration
from gemseo.core.algorithm.base_driver_library import BaseDriverLibrary
from gemseo.core.discipline.base_discipline import BaseDiscipline
from gemseo.core.discipline.execution_statistics import ExecutionStatistics
from gemseo.core.discipline.execution_status import ExecutionStatus
from gemseo.core.function.preprocessed_function import PreprocessedFunction
from gemseo.core.parallel_execution.callable_parallel_execution import (
    CallableParallelExecution,
)
from gemseo.core.problem.evaluation import EvaluationProblem
from gemseo.discipline.analytic import AnalyticDiscipline
from gemseo.discipline.chain.parallel_chain import ParallelDisciplineChain
from gemseo.mda.gauss_seidel_newton_raphson_settings import (
    MDAGaussSeidelNewtonRaphson_Settings,
)
from gemseo.mda.jacobi_settings import MDAJacobi_Settings
from gemseo.util import global_configuration as global_configuration_module
from gemseo.util.constant import n_cpus
from gemseo.util.global_configuration import GlobalConfiguration
from gemseo.util.global_configuration import _apply
from gemseo.util.global_configuration import _fast_field_names
from gemseo.util.testing.helper import assert_exception

if TYPE_CHECKING:
    from gemseo.util.typing import StrKeyMapping

pytestmark = pytest.mark.usefixtures("restore_configuration_options")


@pytest.fixture
def apply_counter(monkeypatch) -> Counter:
    """The counter of the calls to the function applying each field.

    Args:
        monkeypatch: The fixture to patch the module.

    Returns:
        The number of calls per field name.
    """
    counter = Counter()
    monkeypatch.setattr(
        global_configuration_module,
        "_apply",
        MappingProxyType({
            name: (lambda value, name=name: counter.update([name])) for name in _apply
        }),
    )
    return counter


def assert_applied(configuration: GlobalConfiguration) -> None:
    """Check that the class attributes agree with the configuration fields.

    Args:
        configuration: The global configuration.
    """
    assert EvaluationProblem.check_bounds is configuration.check_desvars_bounds
    assert (
        BaseDiscipline.default_cache_type is not BaseDiscipline.CacheType.NONE
    ) is configuration.enable_discipline_cache
    assert ExecutionStatistics.is_enabled is configuration.enable_discipline_statistics
    assert ExecutionStatus.is_enabled is configuration.enable_discipline_status
    assert (
        PreprocessedFunction.enable_statistics
        is configuration.enable_function_statistics
    )
    assert BaseDriverLibrary.enable_progress_bar is configuration.enable_progress_bar
    assert BaseDiscipline.validate_input_data is configuration.validate_input_data
    assert BaseDiscipline.validate_output_data is configuration.validate_output_data
    assert MDAJacobi_Settings().n_processes == (
        n_cpus if configuration.enable_parallel_execution else 1
    )
    assert (
        CallableParallelExecution.enable_parallel_execution
        is configuration.enable_parallel_execution
    )


def test_default():
    """Check the GlobalConfiguration."""
    assert GlobalConfiguration.model_fields.keys() == {
        "check_desvars_bounds",
        "enable_discipline_cache",
        "enable_discipline_statistics",
        "enable_discipline_status",
        "enable_function_statistics",
        "enable_parallel_execution",
        "enable_progress_bar",
        "logging",
        "directory_manager",
        "validate_input_data",
        "validate_output_data",
    }
    settings = GlobalConfiguration()
    assert settings.check_desvars_bounds
    assert settings.enable_discipline_cache
    assert not settings.enable_discipline_statistics
    assert not settings.enable_discipline_status
    assert not settings.enable_function_statistics
    assert settings.enable_parallel_execution
    assert settings.enable_progress_bar
    assert settings.validate_input_data
    assert settings.validate_output_data
    assert_applied(settings)

    logging = settings.logging
    assert logging.date_format == "%H:%M:%S"
    assert logging.enable
    assert logging.file_path == ""
    assert logging.file_mode == "a"
    assert logging.level == INFO
    assert logging.message_format == "%(levelname)8s - %(asctime)s: %(message)s"

    dm = settings.directory_manager
    assert not dm.enable
    assert dm.execution_root_path == Path()
    assert dm.clean_up_policy == "KEEP_ALL"
    assert dm.mda_clean_up_policy == "KEEP_ALL"
    assert not dm.save_history_backup
    assert not dm.save_mda_residuals
    assert not dm.keep_failed_executions


def test_enable_fast_mode():
    """Check that the fast mode disables the options that it drives."""
    configuration.enable_fast_mode()
    for name in _fast_field_names:
        assert not getattr(configuration, name)

    assert configuration.enable_progress_bar
    assert_applied(configuration)


def test_disable_fast_mode():
    """Check that leaving the fast mode resets the options to their default values."""
    configuration.enable_fast_mode()
    configuration.disable_fast_mode()
    fields = GlobalConfiguration.model_fields
    for name in _fast_field_names:
        assert getattr(configuration, name) is fields[name].default

    assert_applied(configuration)


def test_disable_fast_mode_keeps_the_statistics_disabled():
    """Check that leaving the fast mode does not enable the statistics."""
    configuration.enable_fast_mode()
    configuration.disable_fast_mode()
    assert not configuration.enable_discipline_statistics
    assert not configuration.enable_discipline_status
    assert not configuration.enable_function_statistics
    assert configuration.check_desvars_bounds
    assert configuration.enable_discipline_cache
    assert_applied(configuration)


def test_disable_fast_mode_does_not_restore_the_previous_values():
    """Check that leaving the fast mode does not restore the values set before it."""
    configuration.validate_input_data = False
    configuration.enable_fast_mode()
    configuration.disable_fast_mode()
    assert configuration.validate_input_data
    assert_applied(configuration)


def test_option_assigned_after_the_fast_mode():
    """Check that an option assigned after the fast mode keeps its value."""
    configuration.enable_fast_mode()
    configuration.validate_input_data = True

    # Assigning another option leaves both alone.
    configuration.enable_progress_bar = False
    assert configuration.validate_input_data
    assert not configuration.check_desvars_bounds
    assert not configuration.enable_discipline_cache
    assert not configuration.validate_output_data
    assert_applied(configuration)


def test_enable_fast_mode_applies_the_options_again():
    """Check that the fast mode applies its options at every call."""
    configuration.enable_fast_mode()

    # Set a class attribute directly, as many tests and users do.
    BaseDiscipline.validate_input_data = True

    configuration.enable_fast_mode()
    assert not BaseDiscipline.validate_input_data
    assert_applied(configuration)


def test_fast_mode_applied_by_another_configuration():
    """Check that creating a configuration resets what another one applied."""
    configuration.enable_fast_mode()
    GlobalConfiguration()
    assert BaseDiscipline.validate_input_data

    # The global configuration can apply the fast mode again.
    configuration.enable_fast_mode()
    assert not BaseDiscipline.validate_input_data
    assert_applied(configuration)


def test_fast_mode_options_applied_once(apply_counter):
    """Check that the fast mode applies each option that it drives once."""
    configuration.enable_fast_mode()
    assert apply_counter == Counter(_fast_field_names)


def test_enable_fast_mode_documents_the_options():
    """Check that the docstring of enable_fast_mode lists the options that it drives."""
    docstring = GlobalConfiguration.enable_fast_mode.__doc__
    assert set(re.findall(r"- `(\w+)`", docstring)) == set(_fast_field_names)


def test_fast_rejected_by_the_constructor(snapshot):
    """Check that passing the removed fast option raises."""
    with assert_exception(ValueError, snapshot):
        GlobalConfiguration(fast=True)


def test_fast_rejected_by_the_constructor_whatever_the_case(snapshot):
    """Check that the removed fast option is rejected whatever the case of its name."""
    with assert_exception(ValueError, snapshot):
        GlobalConfiguration(FAST=True)


def test_fast_rejected_from_a_dotenv_file(tmp_wd, snapshot):
    """Check that the removed fast option raises when read from a dotenv file."""
    (tmp_wd / ".env").write_text("GEMSEO_FAST=True")
    with assert_exception(ValidationError, snapshot):
        GlobalConfiguration()


def test_fast_rejected_from_an_environment_variable(monkeypatch, snapshot):
    """Check that the removed fast option raises when set in the environment."""
    monkeypatch.setenv("GEMSEO_FAST", "True")
    with assert_exception(ValidationError, snapshot):
        GlobalConfiguration()


def test_unprefixed_fast_in_a_dotenv_file(tmp_wd):
    """Check that a dotenv file holding a key named fast is ignored."""
    (tmp_wd / ".env").write_text("FAST=True")
    assert GlobalConfiguration().validate_input_data


def test_n_processes_of_an_indirect_subclass():
    """Check that the default n_processes is applied to an indirect subclass."""

    class GrandChildSettings(MDAJacobi_Settings):
        """A settings class two levels below the parallel solver settings."""

    configuration.enable_parallel_execution = False
    assert GrandChildSettings().n_processes == 1

    configuration.enable_parallel_execution = True
    assert GrandChildSettings().n_processes == n_cpus


def test_n_processes_of_a_callable_parallel_execution():
    """Check that the default n_processes reaches CallableParallelExecution."""
    configuration.enable_parallel_execution = False
    assert CallableParallelExecution([len]).n_processes == 1

    configuration.enable_parallel_execution = True
    assert CallableParallelExecution([len]).n_processes == n_cpus

    # An explicit number of processes is left alone.
    assert CallableParallelExecution([len], n_processes=2).n_processes == 2


def test_n_processes_of_a_parallel_discipline_chain():
    """Check that the parallel execution option reaches ParallelDisciplineChain."""
    disciplines = [AnalyticDiscipline({"y": "x"}), AnalyticDiscipline({"z": "x"})]

    configuration.enable_parallel_execution = False
    chain = ParallelDisciplineChain(disciplines)
    assert chain.parallel_execution.n_processes == 1
    assert chain.parallel_lin.n_processes == 1

    configuration.enable_parallel_execution = True
    chain = ParallelDisciplineChain(disciplines)
    assert chain.parallel_execution.n_processes == len(disciplines)
    assert chain.parallel_lin.n_processes == len(disciplines)

    # An explicit number of processes is left alone.
    chain = ParallelDisciplineChain(disciplines, n_processes=1)
    assert chain.parallel_execution.n_processes == 1


def test_each_field_is_applied_once_at_creation(apply_counter):
    """Check that each field is applied once when a configuration is created."""
    GlobalConfiguration()
    assert apply_counter == Counter(_apply.keys())


def test_n_processes_of_a_nested_settings_model():
    """Check that the default n_processes reaches an embedded settings model."""
    configuration.enable_parallel_execution = False
    settings = MDAGaussSeidelNewtonRaphson_Settings(newton_raphson_settings={})
    assert settings.newton_raphson_settings.n_processes == 1

    configuration.enable_parallel_execution = True
    settings = MDAGaussSeidelNewtonRaphson_Settings(newton_raphson_settings={})
    assert settings.newton_raphson_settings.n_processes == n_cpus


def test_n_processes_of_a_subclass_with_its_own_default():
    """Check that a settings subclass declaring its own n_processes keeps it."""

    class SerialSettings(MDAJacobi_Settings):
        """A settings class pinning the number of processes."""

        _inherited_field_defaults: ClassVar[StrKeyMapping] = {"n_processes": 1}

    assert SerialSettings().n_processes == 1

    configuration.enable_parallel_execution = False
    assert SerialSettings().n_processes == 1

    configuration.enable_parallel_execution = True
    assert SerialSettings().n_processes == 1
    assert MDAJacobi_Settings().n_processes == n_cpus


def test_only_the_assigned_field_is_applied():
    """Check that assigning a field does not apply the other ones."""
    # Set a class attribute directly, as many tests and users do.
    BaseDiscipline.validate_input_data = not configuration.validate_input_data

    # Assigning another field must leave it alone.
    configuration.enable_progress_bar = not configuration.enable_progress_bar
    assert BaseDiscipline.validate_input_data is not configuration.validate_input_data
    assert BaseDriverLibrary.enable_progress_bar is configuration.enable_progress_bar

    # Assigning the field itself applies it again, even to the same value.
    configuration.validate_input_data = configuration.validate_input_data
    assert BaseDiscipline.validate_input_data is configuration.validate_input_data


def test_assigning_a_field_applying_to_nothing():
    """Check that assigning a field configuring no class applies no other field."""
    # Set a class attribute directly, as many tests and users do.
    BaseDiscipline.validate_input_data = not configuration.validate_input_data

    # directory_manager configures no class attribute, so nothing must be applied.
    configuration.directory_manager = configuration.directory_manager
    assert BaseDiscipline.validate_input_data is not configuration.validate_input_data


def test_environment_variable(monkeypatch):
    """Check the use of environment variables."""
    logger = getLogger("NewLogger")
    logger.addHandler(NullHandler())
    assert GlobalConfiguration().enable_progress_bar
    monkeypatch.setenv("GEMSEO_ENABLE_PROGRESS_BAR", "False")
    monkeypatch.setenv("GEMSEO_LOGGING_ENABLE", "False")
    configuration = GlobalConfiguration()
    assert not configuration.enable_progress_bar
    assert not configuration.logging.enable
    assert not getLogger("gemseo").handlers
    assert getLogger("").handlers
    assert getLogger("NewLogger").handlers


def test_environment_variable_env_file(monkeypatch, tmp_wd):
    """Check the use of environment variables from a .env file."""
    assert GlobalConfiguration().enable_progress_bar
    with (tmp_wd / ".env").open("w") as f:
        f.write("GEMSEO_ENABLE_PROGRESS_BAR=False\n")
        f.write("GEMSEO_LOGGING_ENABLE=True")
    configuration = GlobalConfiguration()
    assert not configuration.enable_progress_bar
    assert configuration.logging.enable


def test_extra_config_in_env_file(monkeypatch, tmp_wd):
    """Check that extra config in .env file doesn't raise validation errors."""
    with (tmp_wd / ".env").open("w") as f:
        f.write("SOME_OTHER_APP_CONFIG=value\n")

    # This should not raise a ValidationError about extra_forbidden
    configuration = GlobalConfiguration()
    assert not hasattr(configuration, "some_other_app_config")
