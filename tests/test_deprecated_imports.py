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
"""Tests for the backward compatibility of renamed and moved imports."""

from __future__ import annotations

import importlib
import importlib.util
import io
import logging
import os
import pickle
import pkgutil
import subprocess
import sys
import warnings
from enum import StrEnum
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final

import pytest

import gemseo  # noqa: F401 - ensures the deprecated-import finder is installed
from gemseo.util.testing.helper import assert_exception
from tests.marks import requires_numpy_2

if TYPE_CHECKING:
    from _pytest.mark.structures import ParameterSet


def test_moved_module_and_renamed_class():
    """A class renamed and whose module moved is reachable from the old path."""
    from gemseo.mda.gauss_seidel_newton_raphson import MDAGaussSeidelNewtonRaphson

    with pytest.warns(DeprecationWarning, match="gauss_seidel_newton_raphson"):
        from gemseo.mda.gs_newton import MDAGSNewton

    assert MDAGSNewton is MDAGaussSeidelNewtonRaphson


def test_moved_package_submodule():
    """A submodule of a renamed package redirects to the new package."""
    real = importlib.import_module("gemseo.machine_learning.regression.model.rbf")

    with pytest.warns(DeprecationWarning, match="gemseo.machine_learning"):
        old = importlib.import_module("gemseo.mlearning.regression.algos.rbf")

    assert old.RBFRegressor is real.RBFRegressor


def test_cross_package_move_without_class_rename():
    """A class moved across packages (name unchanged) is reachable from the old path."""
    from gemseo.optimization.problem import OptimizationProblem

    with pytest.warns(DeprecationWarning, match="gemseo.optimization.problem"):
        from gemseo.algos.optimization_problem import (
            OptimizationProblem as OldOptimizationProblem,
        )

    assert OldOptimizationProblem is OptimizationProblem


def test_doe_family_package_move():
    """The DOE algorithm family moved from gemseo.algos.doe to gemseo.doe."""
    from gemseo.doe.factory import DOELibraryFactory

    with pytest.warns(DeprecationWarning, match="gemseo.doe"):
        from gemseo.algos.doe.factory import DOELibraryFactory as OldFactory

    assert OldFactory is DOELibraryFactory


def test_doe_family_package_move_deep_path():
    """A deep DOE submodule whose base classes moved under core still redirects."""
    from gemseo.doe.pydoe.pydoe import PyDOELibrary

    with pytest.warns(DeprecationWarning, match="gemseo.doe"):
        from gemseo.algos.doe.pydoe.pydoe import PyDOELibrary as OldPyDOELibrary

    assert OldPyDOELibrary is PyDOELibrary


def test_package_rename_deep_path():
    """The plural-to-singular package rename covers deep module paths."""
    from gemseo.util.directory_creator import Naming

    with pytest.warns(DeprecationWarning, match="gemseo.util"):
        from gemseo.utils.directory_creator import Naming as OldNaming

    assert OldNaming is Naming


def test_renamed_class_via_old_package_path():
    """A class rename is applied when reaching it through the old package path."""
    from gemseo.post.constraint_radar import ConstraintRadar

    with pytest.warns(DeprecationWarning, match="gemseo.post.constraint_radar"):
        from gemseo.post.radar_chart import RadarChart

    assert RadarChart is ConstraintRadar


def test_dropped_reexport_of_renamed_package():
    """A name no longer re-exported redirects to the module defining it."""
    from gemseo.dataset.factory import DatasetFactory

    with pytest.warns(DeprecationWarning, match="'gemseo.dataset'"):
        from gemseo.datasets import DatasetFactory as OldDatasetFactory

    assert OldDatasetFactory is DatasetFactory


def test_dropped_reexport_of_live_package():
    """A name no longer re-exported by a package that kept its name redirects too."""
    from gemseo.uncertainty.statistic.core.base import BaseStatistics

    with pytest.warns(DeprecationWarning, match="BaseStatistics"):
        from gemseo.uncertainty import BaseStatistics as OldBaseStatistics

    assert OldBaseStatistics is BaseStatistics


def test_renamed_function():
    """A function rename is applied when reaching it through the old module path."""
    from gemseo.core.derivative.graph_traversal import set_differentiated_ios

    with pytest.warns(DeprecationWarning, match="graph_traversal"):
        from gemseo.core.derivatives.chain_rule import traverse_add_diff_io

    assert traverse_add_diff_io is set_differentiated_ios


def test_renamed_function_via_old_reexport_path():
    """A function rename is applied on the old modules re-exporting the function."""
    from gemseo.core.derivative.graph_traversal import set_mda_differentiated_ios

    with pytest.warns(DeprecationWarning, match="gemseo.mda.jacobian_assembly"):
        from gemseo.core.derivatives.jacobian_assembly import traverse_add_diff_io_mda

    assert traverse_add_diff_io_mda is set_mda_differentiated_ios


def test_renamed_attribute_of_renamed_module_warns_about_the_attribute():
    """The warning names the renamed attribute, not only the renamed module.

    The module warning alone would point at a module where the old attribute name
    either does not exist or, worse, denotes another object.
    """
    with pytest.warns(DeprecationWarning, match="'RadarChart'") as records:
        from gemseo.post.radar_chart import RadarChart  # noqa: F401

    messages = [str(record.message) for record in records]
    assert (
        "The attribute 'RadarChart' of the module 'gemseo.post.radar_chart' is "
        "deprecated; use 'gemseo.post.constraint_radar.ConstraintRadar' instead."
        in messages
    )


def test_dropped_reexport_warning_gives_the_new_location():
    """The advice of a dropped re-export points at the module defining the successor."""
    with pytest.warns(DeprecationWarning, match="'BaseMLAlgo'") as records:
        from gemseo.mlearning import BaseMLAlgo  # noqa: F401

    messages = [str(record.message) for record in records]
    assert (
        "The attribute 'BaseMLAlgo' of the module 'gemseo.mlearning' is deprecated; "
        "use 'gemseo.machine_learning.core.model.base_ml_model.BaseMLModel' instead."
        in messages
    )


def test_renamed_module_level_constant():
    """A constant renamed in a module that moved is reachable from the old path."""
    from gemseo.util.constant import infinite_int

    with pytest.warns(DeprecationWarning, match="'gemseo.util.constant.infinite_int'"):
        from gemseo.utils.constants import C_LONG_MAX

    assert infinite_int == C_LONG_MAX


def test_moved_module_level_constant():
    """A constant moved to another module is reachable from the old path."""
    from gemseo.util.constant import epsilon

    with pytest.warns(DeprecationWarning, match="'gemseo.util.constant.epsilon'"):
        from gemseo.utils.derivatives.error_estimators import EPSILON as OLD_EPSILON

    assert epsilon == OLD_EPSILON


@pytest.mark.parametrize(
    "old_module_name", ["gemseo.utils.logging", "gemseo.utils.logging_tools"]
)
def test_removed_gemseo_logger_constant_resolves_to_the_gemseo_logger(
    old_module_name,
):
    """The removed GEMSEO_LOGGER constant is the gemseo logger from its old paths."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old_module = importlib.import_module(old_module_name)

    with pytest.warns(DeprecationWarning, match="'gemseo.logger'"):
        gemseo_logger = old_module.GEMSEO_LOGGER

    assert gemseo_logger is logging.getLogger("gemseo")


@pytest.mark.parametrize(
    ("old_name", "new_module_name", "new_name"),
    [
        ("BaseScenario", "gemseo.scenario.evaluation", "EvaluationScenario"),
        (
            "BaseParallelMDASettings",
            "gemseo.mda.core.base_parallel_solver_settings",
            "BaseMDAParallelSolverSettings",
        ),
    ],
)
def test_renamed_top_level_reexport(old_name, new_module_name, new_name):
    """A class re-exported by the gemseo 6.3.2 package is reachable from it."""
    new_object = getattr(importlib.import_module(new_module_name), new_name)

    with pytest.warns(DeprecationWarning, match=f"'{new_module_name}.{new_name}'"):
        old_object = getattr(gemseo, old_name)

    assert old_object is new_object


_top_level_module_warning: Final[str] = (
    "The attribute 'base_parallel_mda_settings' of the module 'gemseo' is deprecated; "
    "use 'gemseo.mda.core.base_parallel_solver_settings' instead."
)
"""The warning of the module re-exported by the gemseo 6.3.2 package."""


def test_renamed_top_level_reexport_of_a_module():
    """A module re-exported by the gemseo 6.3.2 package is importable from it."""
    from gemseo.mda.core import base_parallel_solver_settings

    with pytest.warns(DeprecationWarning, match=_top_level_module_warning):
        from gemseo import base_parallel_mda_settings

    assert base_parallel_mda_settings is base_parallel_solver_settings


def test_renamed_top_level_reexport_of_a_module_as_an_attribute():
    """A module re-exported by the gemseo 6.3.2 package is an attribute of it."""
    from gemseo.mda.core import base_parallel_solver_settings

    with pytest.warns(DeprecationWarning, match=_top_level_module_warning):
        module = gemseo.base_parallel_mda_settings

    assert module is base_parallel_solver_settings


def test_renamed_attribute_to_a_module_not_imported_yet(monkeypatch):
    """An attribute renamed to a module imports the latter when not imported yet.

    Args:
        monkeypatch: Fixture to forget the module and the table of the renames.
    """
    from gemseo import _deprecation

    live_module = ModuleType("gemseo.fake_live_module")
    new_name = "gemseo.mda.core.base_parallel_solver_settings"
    monkeypatch.setattr(
        _deprecation, "attribute_renames", {live_module.__name__: {"old": new_name}}
    )
    monkeypatch.delitem(sys.modules, new_name)
    monkeypatch.delattr(
        importlib.import_module("gemseo.mda.core"), "base_parallel_solver_settings"
    )
    _deprecation._install_attribute_aliases(live_module)

    with pytest.warns(DeprecationWarning, match=f"use '{new_name}' instead"):
        module = live_module.old

    assert module is sys.modules[new_name]


@pytest.mark.parametrize(
    "new_name",
    [
        # Neither an attribute nor a submodule of a package.
        "gemseo.mda.core.does_not_exist",
        # Not an attribute of a module, which has no submodule.
        "gemseo.util.string.does_not_exist",
    ],
)
def test_renamed_attribute_to_a_missing_name_raises(monkeypatch, new_name, snapshot):
    """An attribute renamed to a missing name raises an `AttributeError`.

    Args:
        monkeypatch: Fixture to patch the table of the renamed attributes.
        new_name: The missing new name.
        snapshot: Fixture to compare the error message with a snapshot.
    """
    from gemseo import _deprecation

    live_module = ModuleType("gemseo.fake_live_module")
    monkeypatch.setattr(
        _deprecation, "attribute_renames", {live_module.__name__: {"old": new_name}}
    )
    _deprecation._install_attribute_aliases(live_module)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with assert_exception(AttributeError, snapshot):
            live_module.old  # noqa: B018


def test_star_import_from_deprecated_module():
    """A star import from an old path binds the names of the new one."""
    from gemseo.util import string

    namespace: dict[str, object] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        exec("from gemseo.utils.string_tools import *", namespace)  # noqa: S102

    expected = {name for name in vars(string) if not name.startswith("_")}
    assert expected
    assert expected <= set(namespace)


def test_star_import_from_deprecated_module_skips_unmigrated_names(monkeypatch):
    """A star import from an old path skips the names that raise.

    A name removed with no replacement, or whose migration cannot be automated, is
    left out of the `__all__` of the stand-in even when the new module defines it, so
    that the star import does not fail partway.

    Args:
        monkeypatch: Fixture to forget the stand-in and patch the table of the names
            that raise.
    """
    from gemseo import _deprecation

    module_name = "gemseo.utils.string_tools"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        importlib.import_module(module_name)
    # The stand-in is rebuilt from the patched tables, and restored afterwards.
    monkeypatch.delitem(sys.modules, module_name)
    monkeypatch.setattr(
        _deprecation,
        "unmigrated_attributes",
        {module_name: frozenset({"partial", "deepcopy"})},
    )
    namespace: dict[str, object] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        exec(f"from {module_name} import *", namespace)  # noqa: S102

    assert "escape" in namespace
    assert "partial" not in namespace
    assert "deepcopy" not in namespace


def test_star_import_from_deprecated_module_binds_the_renamed_names():
    """A star import from an old path binds the old names renamed in the old module.

    The new module does not define them, as `ParameterSpace` renamed to `RandomSpace`.
    """
    from gemseo.space.random import RandomSpace

    namespace: dict[str, object] = {}
    with pytest.warns(DeprecationWarning, match="'ParameterSpace'"):
        exec("from gemseo.algos.parameter_space import *", namespace)  # noqa: S102

    assert namespace["ParameterSpace"] is RandomSpace


def test_dir_of_deprecated_module():
    """`dir` on an old path exposes the names of the new one."""
    from gemseo.util import string

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = importlib.import_module("gemseo.utils.string_tools")

    assert "MultiLineString" in dir(old)
    assert set(dir(string)) <= set(dir(old))


def test_star_import_from_dissolved_package(monkeypatch):
    """A star import from the dissolved settings package binds nothing.

    The old package defined no name, only its submodules were renamed.
    """
    monkeypatch.delitem(sys.modules, "gemseo.settings", raising=False)
    namespace: dict[str, object] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        exec("from gemseo.settings import *", namespace)  # noqa: S102

    # `__warningregistry__` is added by the warnings machinery, not by the import.
    assert set(namespace) - {"__builtins__", "__warningregistry__"} == set()


def test_missing_dependency_is_not_reported_as_a_missing_attribute(monkeypatch):
    """A dependency missing behind an alias is not masked by an `AttributeError`."""
    old = importlib.import_module("gemseo.mda.gs_newton")

    class _Boom:
        __name__ = "gemseo.mda.gauss_seidel_newton_raphson"

        def __getattr__(self, name: str) -> object:
            msg = "No module named 'some_optional_dependency'"
            raise ModuleNotFoundError(msg, name="some_optional_dependency")

    monkeypatch.setitem(old.__dict__, "_deprecation_target", _Boom())
    with pytest.raises(ModuleNotFoundError, match="some_optional_dependency"):
        old.MDAGSNewton  # noqa: B018


@pytest.mark.parametrize(
    "new_name",
    [
        # A module of GEMSEO.
        "gemseo.mda.gauss_seidel_newton_raphson",
        # A module of an installed plugin, whose own dependency is missing.
        "gemseo_fake_plugin.module",
    ],
)
def test_missing_dependency_of_a_redirected_module_is_reraised(monkeypatch, new_name):
    """A dependency missing behind a stand-in is re-raised unchanged.

    It is not mistaken for a module moved to a plugin that is not installed.

    Args:
        monkeypatch: Fixture to patch the import of the new module.
        new_name: The name of the new module.
    """
    from gemseo import _deprecation

    error = ModuleNotFoundError(
        "No module named 'some_optional_dependency'", name="some_optional_dependency"
    )

    def import_module(name: str) -> ModuleType:
        raise error

    monkeypatch.setattr(_deprecation, "import_module", import_module)
    loader = _deprecation._DeprecatedModuleLoader("gemseo.fake_old_module", new_name)

    with pytest.raises(ModuleNotFoundError) as exc_info:
        loader.create_module(None)

    assert exc_info.value is error


def test_user_warning_filters_are_not_overridden():
    """`install` leaves the warning filters alone when the user configured them."""
    helper = "from gemseo.scenarios.mdo_scenario import MDOScenario\n"
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-W", "error::DeprecationWarning", "-c", helper],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0, result.stdout
    assert "DeprecationWarning" in result.stderr


@requires_numpy_2
def test_class_deprecation_is_visible_under_the_default_filters():
    """The deprecation of a class is shown although it is raised by library code.

    The default filters silence a `DeprecationWarning` that is not raised from
    `__main__`; `install` registers a filter so that this one is shown.
    """
    pickle_path = Path(__file__).parent / "space" / "design_space_6_3_3.pkl"
    # Load through the gemseo helper, as a user does: the warning is then raised
    # from library code, which the default filters silence.
    helper = (
        f"from gemseo.util.pickle import from_pickle\nfrom_pickle(r'{pickle_path}')\n"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", helper],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "The class 'gemseo.space.variable.Variable' is deprecated" in result.stderr


def test_data_type_deprecation_is_visible_under_the_default_filters():
    """The deprecation of a data type value is shown although raised by library code.

    The default filters silence a `DeprecationWarning` that is not raised from
    `__main__`; `install` registers a filter so that this one is shown.

    This can only be checked in a child process: `pytest` runs every test inside
    `warnings.catch_warnings()` with `simplefilter("always")`, which replaces the
    filter under test. The child is run without any `-W` flag on purpose, since
    `install` registers the filters only when the user configured none.
    """
    csv_path = Path(__file__).parent / "space" / "design_space_legacy_type.csv"
    # Read through the gemseo API, as a user does: the warning is then raised
    # from library code, which the default filters silence.
    helper = (
        f"from gemseo.space import DesignSpace\nDesignSpace.from_csv(r'{csv_path}')\n"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", helper],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "The variable data type 'float' is deprecated" in result.stderr


def test_renamed_submodule_is_not_an_attribute_rename():
    """A renamed submodule and a renamed function are told apart by their section."""
    with pytest.warns(
        DeprecationWarning, match="'gemseo.machine_learning.data_formatter'"
    ):
        importlib.import_module("gemseo.mlearning.data_formatters")

    from gemseo.core.derivative.graph_traversal import set_differentiated_ios

    with pytest.warns(DeprecationWarning, match="'traverse_add_diff_io'"):
        from gemseo.core.derivatives.chain_rule import traverse_add_diff_io

    assert traverse_add_diff_io is set_differentiated_ios


def test_renamed_attribute_of_live_module():
    """A class rename is applied on a module that kept its own name."""
    from gemseo.post.dataset.radviz import RadViz

    with pytest.warns(DeprecationWarning, match="'Radar'"):
        from gemseo.post.dataset.radviz import Radar

    assert Radar is RadViz


def test_renamed_attribute_of_live_package():
    """A class rename is applied on a package that kept its own name."""
    from gemseo.post import ConstraintRadar_Settings

    with pytest.warns(DeprecationWarning, match="'RadarChart_Settings'"):
        from gemseo.post import RadarChart_Settings

    assert RadarChart_Settings is ConstraintRadar_Settings


def test_renamed_attribute_of_live_module_imported_after_install(monkeypatch):
    """A live module imported after `install` is aliased by the finder.

    [install][gemseo._deprecation.install] only aliases the live modules already
    imported; the others go through the finder, which wraps their real loader.
    """
    module_name = "gemseo.post.dataset.radviz"
    monkeypatch.delitem(sys.modules, module_name)

    module = importlib.import_module(module_name)

    with pytest.warns(DeprecationWarning, match="'Radar'"):
        assert module.Radar is module.RadViz


def test_alias_loader_delegates_the_loader_protocol(monkeypatch):
    """The loader wrapping the real one delegates the rest of the loader protocol."""
    module_name = "gemseo.post.dataset.radviz"
    monkeypatch.delitem(sys.modules, module_name)

    spec = importlib.util.find_spec(module_name)

    assert spec.loader.get_filename(module_name).endswith("radviz.py")


def test_live_module_without_real_spec_is_left_to_the_import_machinery(monkeypatch):
    """A live alias entry with no real spec falls back to the normal import.

    This happens for a stale entry, whose module no longer exists, and for a namespace
    package, whose spec carries no loader to wrap.
    """
    from gemseo import _deprecation

    module_name = "gemseo.post.dataset.radviz"
    monkeypatch.delitem(sys.modules, module_name)
    monkeypatch.setattr(_deprecation, "_find_spec", lambda fullname: None)

    module = importlib.import_module(module_name)

    with pytest.raises(AttributeError):
        module.Radar  # noqa: B018


def test_install_ignores_the_live_modules_not_imported_yet(monkeypatch):
    """`install` leaves the live modules that are not imported yet to the finder.

    Its post-insert sweep only iterates the modules already in `sys.modules`, so a
    module that is not imported yet is never touched by it (and never accidentally
    imported by it either). The user warning filters are left alone at the same
    time, as both depend on the state that `install` finds rather than on the alias
    tables.
    """
    from gemseo import _deprecation

    module_name = "gemseo.post.dataset.radviz"
    monkeypatch.delitem(sys.modules, module_name)
    monkeypatch.setattr(_deprecation, "_installed", False)
    monkeypatch.setattr(sys, "warnoptions", ["error::DeprecationWarning"])
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    filters = list(warnings.filters)

    _deprecation.install()

    assert warnings.filters == filters
    assert module_name not in sys.modules


def test_lazy_reexport_of_live_package_still_works():
    """Aliasing the attributes of a package does not break its lazy re-export."""
    import gemseo.post
    from gemseo.post.som import SOM

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert gemseo.post.SOM is SOM


def test_unknown_attribute_of_live_package_raises(snapshot):
    """An unknown attribute of a package with aliases raises AttributeError."""
    import gemseo.post

    with assert_exception(AttributeError, snapshot):
        gemseo.post.does_not_exist  # noqa: B018


def test_unknown_module_raises():
    """An unknown submodule of a live package still raises ModuleNotFoundError."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("gemseo.mda.does_not_exist")


def test_unknown_attribute_raises():
    """An unknown attribute of a redirected module raises ImportError."""
    with pytest.raises(ImportError):
        from gemseo.mda.gs_newton import DoesNotExist  # noqa: F401


def test_new_name_does_not_warn():
    """Importing a current (new) name emits no DeprecationWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        importlib.import_module("gemseo.mda.gauss_seidel_newton_raphson")


def test_install_is_idempotent():
    """Calling `install` again once already installed is a no-op."""
    from gemseo._deprecation import install

    meta_path_length_before = len(sys.meta_path)
    install()
    assert len(sys.meta_path) == meta_path_length_before


def test_warning_shown_under_default_filters(tmp_path):
    """The deprecation is shown even when triggered from library code (not __main__).

    The default warning filters silence `DeprecationWarning` outside `__main__`;
    [install][gemseo._deprecation.install] adds a filter that overrides this.
    """
    helper = tmp_path / "helper_deprecated_import.py"
    helper.write_text("from gemseo.scenarios.mdo_scenario import MDOScenario\n")
    env = dict(os.environ, PYTHONPATH=str(tmp_path))
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", "import helper_deprecated_import"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert "DeprecationWarning" in result.stderr
    assert (
        "The module 'gemseo.scenarios' is deprecated; use 'gemseo.scenario' instead."
        in result.stderr
    )
    assert (
        "The module 'gemseo.scenarios.mdo_scenario' is deprecated; "
        "use 'gemseo.scenario.mdo' instead." in result.stderr
    )


def test_dissolved_settings_package(monkeypatch):
    """The dissolved gemseo.settings package imports and warns, but has no names."""
    monkeypatch.delitem(sys.modules, "gemseo.settings", raising=False)
    with pytest.warns(DeprecationWarning, match="'gemseo.settings' is deprecated"):
        settings = importlib.import_module("gemseo.settings")

    assert "SLSQP_Settings" not in dir(settings)
    assert not hasattr(settings, "SLSQP_Settings")
    assert not hasattr(settings, "__all__")


def test_dissolved_settings_from_package_import_submodule(monkeypatch):
    """A submodule of the dissolved package is importable from the package."""
    monkeypatch.delitem(sys.modules, "gemseo.settings", raising=False)
    monkeypatch.delitem(sys.modules, "gemseo.settings.opt", raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from gemseo.settings import opt

    assert opt.SLSQP_Settings.__name__ == "SLSQP_Settings"


def test_dissolved_settings_name_is_not_resolved(monkeypatch):
    """The dissolved package does not resolve the names of its new locations."""
    monkeypatch.delitem(sys.modules, "gemseo.settings", raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(ImportError):
            from gemseo.settings import SLSQP_Settings  # noqa: F401


def test_dissolved_settings_submodule_redirect(monkeypatch):
    """An old settings aggregator module redirects to its domain package."""
    from gemseo.optimization import SLSQP_Settings

    monkeypatch.delitem(sys.modules, "gemseo.settings.opt", raising=False)
    with pytest.warns(DeprecationWarning, match="gemseo.optimization"):
        from gemseo.settings.opt import SLSQP_Settings as OldSLSQP_Settings

    assert OldSLSQP_Settings is SLSQP_Settings


def test_dissolved_settings_chain_flattened(monkeypatch):
    """A 6.x.y plural settings module resolves directly to the final location."""
    from gemseo.formulation import MDF_Settings

    monkeypatch.delitem(sys.modules, "gemseo.settings.formulations", raising=False)
    with pytest.warns(DeprecationWarning, match="'gemseo.formulation'"):
        old = importlib.import_module("gemseo.settings.formulations")

    assert old.MDF_Settings is MDF_Settings


def test_dissolved_package_unknown_attribute(snapshot):
    """Any attribute of a dissolved package raises AttributeError."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        settings = importlib.import_module("gemseo.settings")

    with assert_exception(AttributeError, snapshot):
        settings.does_not_exist  # noqa: B018


def _needs_a_missing_package(old_module: str) -> bool:
    """Return whether an old module is redirected to a package that is not installed.

    Such a package is a GEMSEO plugin, e.g. `gemseo_excel`, or a third-party one
    whose names the codemod renames, e.g. `strenum`.

    Args:
        old_module: The old fully-qualified module name.

    Returns:
        Whether the module is, or moved to, a package outside GEMSEO missing from the
        environment.
    """
    from gemseo._deprecation.aliases import module_renames

    new_module = module_renames.get(old_module, old_module)
    package = new_module.partition(".")[0]
    return package != "gemseo" and importlib.util.find_spec(package) is None


def test_every_rename_entry_is_reachable():
    """Every rename-table entry redirects to an importable target.

    The names whose migration cannot be automated, the `TODO <text>` entries of
    `attributes:`, are no renames: importing one raises instead of resolving.
    """
    from gemseo._deprecation.aliases import attribute_renames
    from gemseo._deprecation.aliases import module_renames

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for old_module in module_renames:
            if not _needs_a_missing_package(old_module):
                importlib.import_module(old_module)
        for old_module, renames in attribute_renames.items():
            if _needs_a_missing_package(old_module):
                continue
            module = importlib.import_module(old_module)
            for old_name in renames:
                getattr(module, old_name)


def test_manual_migration_raises(snapshot):
    """A name whose migration cannot be automated raises instead of being aliased.

    An `ImportError` is raised rather than an `AttributeError`, so that a
    `from ... import ...` of the name reports this message instead of the generic
    one that the import machinery builds from an `AttributeError`. The message ends
    with the instruction of the `TODO <text>` entry of the name.

    Args:
        snapshot: Fixture to compare the error message with a snapshot.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        module = importlib.import_module(
            "gemseo.disciplines.scenario_adapters.mdo_objective_scenario_adapter"
        )
        with assert_exception(ImportError, snapshot):
            module.MDOObjectiveScenarioAdapter  # noqa: B018


def test_every_manual_migration_entry_raises():
    """Every entry of the manual-migration table raises instead of being aliased.

    The message of the error ends with the instruction of the entry.
    """
    from gemseo._deprecation.aliases import manual_migrations

    assert manual_migrations
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for old_module, migrations in manual_migrations.items():
            module = importlib.import_module(old_module)
            for old_name, migration in migrations.items():
                with pytest.raises(ImportError) as exc_info:
                    getattr(module, old_name)
                assert str(exc_info.value) == (
                    f"The attribute {old_name!r} of the module {old_module!r} was "
                    f"removed; {migration}."
                )


def test_removed_module_raises(snapshot):
    """A module removed with no replacement raises, instead of being redirected.

    Without it, the rename of its package would redirect the import to a module that
    does not exist either, with a message naming the latter.

    Args:
        snapshot: Fixture to compare the error message with a snapshot.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with assert_exception(ModuleNotFoundError, snapshot):
            importlib.import_module("gemseo.utils.enumeration")


@pytest.mark.parametrize(
    ("package_name", "module_name"),
    [
        # A renamed package, reached through its stand-in.
        ("gemseo.utils", "enumeration"),
        # A package that kept its name.
        ("gemseo.core", "_discipline_class_injector"),
    ],
)
def test_removed_module_raises_when_imported_from_its_package(
    package_name, module_name, snapshot
):
    """A module removed with no replacement raises when imported from its package.

    The import machinery swallows the error of `from package import module` for a
    module that does not exist, when the error names the module; the message must
    not be lost.

    Args:
        package_name: The name of the package of the removed module.
        module_name: The name of the removed module in its package.
        snapshot: Fixture to compare the error message with a snapshot.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with assert_exception(ModuleNotFoundError, snapshot):
            exec(f"from {package_name} import {module_name}", {})  # noqa: S102


def test_removed_module_raises_when_its_spec_is_found(snapshot):
    """Finding the spec of a module removed with no replacement raises.

    Args:
        snapshot: Fixture to compare the error message with a snapshot.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with assert_exception(ModuleNotFoundError, snapshot):
            importlib.util.find_spec("gemseo.utils.enumeration")


_removed_attribute_cases: Final[tuple[ParameterSet, ...]] = (
    # A renamed module, reached through its stand-in.
    pytest.param(
        "gemseo.utils.study_analyses.study_analysis_cli",
        "STUDY_ANALYSIS_TYPES",
        id="gemseo.utils.study_analyses.study_analysis_cli-STUDY_ANALYSIS_TYPES",
    ),
    # A module that kept its name.
    pytest.param(
        "gemseo.core.discipline.io",
        "_GRAMMAR_FACTORY",
        id="gemseo.core.discipline.io-_GRAMMAR_FACTORY",
    ),
)
"""The modules, reached through a stand-in or live, and one of their removed names."""


@pytest.mark.parametrize(("module_name", "name"), _removed_attribute_cases)
def test_removed_attribute_raises(module_name, name, snapshot):
    """An attribute removed with no replacement raises an error saying so.

    It is an `AttributeError`, whether the module was renamed or kept its name.

    Args:
        module_name: The old name of the module defining the attribute.
        name: The name of the removed attribute.
        snapshot: Fixture to compare the error message with a snapshot.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        module = importlib.import_module(module_name)
        with assert_exception(AttributeError, snapshot):
            getattr(module, name)


@pytest.mark.parametrize(("module_name", "name"), _removed_attribute_cases)
def test_removed_attribute_is_probeable(module_name, name):
    """`hasattr` and `getattr` with a default work on a removed attribute.

    Args:
        module_name: The old name of the module defining the attribute.
        name: The name of the removed attribute.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        module = importlib.import_module(module_name)
    sentinel = object()
    assert not hasattr(module, name)
    assert getattr(module, name, sentinel) is sentinel


@pytest.mark.parametrize(("module_name", "name"), _removed_attribute_cases)
def test_removed_attribute_raises_when_imported_from_its_module(
    module_name, name, snapshot
):
    """A removed attribute imported by a `from` statement raises an error saying so.

    It is an `ImportError`, as the `from` statement would replace an `AttributeError`
    by a generic one losing the message.

    Args:
        module_name: The old name of the module defining the attribute.
        name: The name of the removed attribute.
        snapshot: Fixture to compare the error message with a snapshot.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with assert_exception(ImportError, snapshot):
            exec(f"from {module_name} import {name}", {})  # noqa: S102


@pytest.fixture
def dissolved_package_with_a_removed_attribute(monkeypatch) -> ModuleType:
    """Return the dissolved `gemseo.settings` package, with a removed attribute `Gone`.

    Args:
        monkeypatch: Fixture to patch the table of the removed attributes.

    Returns:
        The stand-in of the dissolved package.
    """
    from gemseo import _deprecation

    monkeypatch.setattr(
        _deprecation, "removed_attributes", {"gemseo.settings": frozenset({"Gone"})}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return importlib.import_module("gemseo.settings")


def test_removed_attribute_of_dissolved_package_is_not_an_attribute(
    dissolved_package_with_a_removed_attribute,
):
    """`hasattr` returns `False` for a removed attribute of a dissolved package.

    Args:
        dissolved_package_with_a_removed_attribute: The stand-in of a dissolved
            package with a removed attribute `Gone`.
    """
    assert not hasattr(dissolved_package_with_a_removed_attribute, "Gone")


def test_removed_attribute_of_dissolved_package_gets_the_default(
    dissolved_package_with_a_removed_attribute,
):
    """`getattr` returns the default for a removed attribute of a dissolved package.

    Args:
        dissolved_package_with_a_removed_attribute: The stand-in of a dissolved
            package with a removed attribute `Gone`.
    """
    sentinel = object()
    assert getattr(dissolved_package_with_a_removed_attribute, "Gone", sentinel) is (
        sentinel
    )


def test_removed_attribute_of_dissolved_package_raises(
    dissolved_package_with_a_removed_attribute, snapshot
):
    """A removed attribute of a dissolved package raises an error saying so.

    Args:
        dissolved_package_with_a_removed_attribute: The stand-in of a dissolved
            package with a removed attribute `Gone`.
        snapshot: Fixture to compare the error message with a snapshot.
    """
    with assert_exception(AttributeError, snapshot):
        dissolved_package_with_a_removed_attribute.Gone  # noqa: B018


def test_removed_attribute_of_dissolved_package_raises_when_imported_from_it(
    dissolved_package_with_a_removed_attribute, snapshot
):
    """A removed attribute imported from a dissolved package raises an error saying so.

    Args:
        dissolved_package_with_a_removed_attribute: The stand-in of a dissolved
            package with a removed attribute `Gone`.
        snapshot: Fixture to compare the error message with a snapshot.
    """
    module_name = dissolved_package_with_a_removed_attribute.__name__
    with assert_exception(ImportError, snapshot):
        exec(f"from {module_name} import Gone", {})  # noqa: S102


def test_module_moved_to_a_missing_plugin_raises(snapshot):
    """A module moved to a plugin that is not installed raises an error naming it.

    Args:
        snapshot: Fixture to compare the error message with a snapshot.
    """
    if importlib.util.find_spec("gemseo_excel") is not None:
        pytest.skip("The gemseo-excel plugin is installed.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with assert_exception(ModuleNotFoundError, snapshot):
            importlib.import_module("gemseo.disciplines.wrappers.xls_discipline")


def test_every_removal_entry_raises():
    """Every module and attribute removed with no replacement raises when imported."""
    from gemseo._deprecation.aliases import removed_attributes
    from gemseo._deprecation.aliases import removed_modules

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for old_module in removed_modules:
            with pytest.raises(ModuleNotFoundError) as exc_info:
                importlib.import_module(old_module)
            assert str(exc_info.value) == (
                f"The module {old_module!r} was removed, with no replacement."
            )
        for old_module, names in removed_attributes.items():
            if _needs_a_missing_package(old_module):
                continue
            module = importlib.import_module(old_module)
            for name in names:
                with pytest.raises(AttributeError) as exc_info:
                    getattr(module, name)
                assert str(exc_info.value) == (
                    f"The attribute {name!r} of the module {old_module!r} was "
                    "removed, with no replacement."
                )


def test_manual_migration_raises_for_dissolved_package(monkeypatch, snapshot):
    """A manual migration is refused on a dissolved package too.

    Args:
        monkeypatch: Fixture to patch the manual-migration table.
        snapshot: Fixture to compare the error message with a snapshot.
    """
    from gemseo import _deprecation
    from gemseo._deprecation.aliases import manual_migrations

    # The table is frozen, and read through the name bound in `_deprecation`, so it is
    # replaced there instead of being mutated.
    monkeypatch.setattr(
        _deprecation,
        "manual_migrations",
        {
            **manual_migrations,
            "gemseo.settings": {"Animation": "use gemseo.post.Animation instead"},
        },
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        settings = importlib.import_module("gemseo.settings")

    with assert_exception(ImportError, snapshot):
        settings.Animation  # noqa: B018


@pytest.mark.parametrize(
    ("module_name", "class_name", "expected_module"),
    [
        # Upstream 6.x.y names.
        ("gemseo.algos.database", "Database", "gemseo.core.problem.database"),
        (
            "gemseo.algos.evaluation_problem",
            "EvaluationProblem",
            "gemseo.core.problem.evaluation",
        ),
        # Class rename applies via attribute_renames.
        (
            "gemseo.algos.base_algo_factory",
            "BaseAlgoFactory",
            "gemseo.core.algorithm.base_algorithm_factory",
        ),
        # Re-exported through optimization.termination_criteria.
        (
            "gemseo.algos.stop_criteria",
            "MaxIterReachedException",
            "gemseo.core.problem.termination_criterion",
        ),
        (
            "gemseo.mlearning.core.algos.ml_algo",
            "BaseMLAlgo",
            "gemseo.machine_learning.core.model.base_ml_model",
        ),
        # Upstream 6.x.y name
        # (problems.mdo.sobieski.core -> problem.mdo.sobieski.standalone).
        (
            "gemseo.problems.mdo.sobieski.core.problem",
            "SobieskiProblem",
            "gemseo.problem.mdo.sobieski.standalone.problem",
        ),
        # Dissolved gemseo.settings aggregator (issue 1719).
        (
            "gemseo.settings.opt",
            "SLSQP_Settings",
            "gemseo.optimization.scipy_local.settings.slsqp",
        ),
        (
            "gemseo.settings.probability_distributions",
            "SPNormalDistribution_Settings",
            "gemseo.uncertainty.distribution.scipy.normal_settings",
        ),
    ],
)
def test_pickle_find_class(module_name, class_name, expected_module):
    """Old pickled paths resolve to the relocated classes via find_class.

    `pickle.Unpickler.find_class` resolves the module and attribute names it is
    given via `__import__` and `getattr`, both of which the deprecated-import
    finder intercepts, so unpickling objects pickled under an old module path
    keeps working.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        unpickler = pickle.Unpickler(io.BytesIO(b""))
        cls = unpickler.find_class(module_name, class_name)
    assert cls.__module__ == expected_module


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("gemseo.core.problem.evaluation", "EvaluationProblem"),
        ("gemseo.optimization.problem", "OptimizationProblem"),
        ("gemseo.core.algorithm.base_driver_library", "BaseDriverLibrary"),
    ],
)
def test_renamed_member_of_a_nested_enumeration(module_name, class_name):
    """The old name of a member of an enumeration nested in a class resolves.

    Args:
        module_name: The name of the module defining the class.
        class_name: The name of the class holding the enumeration.
    """
    cls = getattr(importlib.import_module(module_name), class_name)

    with pytest.warns(
        DeprecationWarning,
        match=(
            "The attribute 'USER_GRAD' of the class 'DifferentiationMethod' is "
            "deprecated; use 'USER' instead."
        ),
    ):
        old_member = cls.DifferentiationMethod.USER_GRAD

    assert old_member is cls.DifferentiationMethod.USER
    # The alias is not a member of the enumeration.
    assert "USER_GRAD" not in cls.DifferentiationMethod.__members__


def test_renamed_member_of_a_nested_enumeration_by_name():
    """The old name of a member of a nested enumeration resolves by name lookup.

    It is not listed among the members all the same.
    """
    from gemseo.optimization.problem import OptimizationProblem

    differentiation_method = OptimizationProblem.DifferentiationMethod
    with pytest.warns(
        DeprecationWarning,
        match=(
            "The attribute 'USER_GRAD' of the class 'DifferentiationMethod' is "
            "deprecated; use 'USER' instead."
        ),
    ):
        old_member = differentiation_method["USER_GRAD"]

    assert old_member is differentiation_method.USER
    assert "USER_GRAD" not in differentiation_method.__members__
    assert "USER_GRAD" not in [member.name for member in differentiation_method]


_upper_cased_member_cases: Final[tuple[ParameterSet, ...]] = (
    pytest.param(
        "gemseo.doe.scipy.settings.base_scipy_doe_settings",
        "Hypersphere",
        "volume",
        "VOLUME",
        id="Hypersphere.volume",
    ),
    pytest.param(
        "gemseo.doe.scipy.settings.base_scipy_doe_settings",
        "Strength",
        "one",
        "ONE",
        id="Strength.one",
    ),
    pytest.param(
        "gemseo.dataset", "DatasetClassName", "IODataset", "IO_DATASET", id="IODataset"
    ),
    pytest.param(
        "gemseo.post.dataset.pair_plot_settings",
        "ColormapName",
        "cool",
        "COOL",
        id="ColormapName.cool",
    ),
    pytest.param(
        "gemseo.uncertainty.statistic.sp_parametric",
        "SPParametricStatistics.DistributionName",
        "norm",
        "NORM",
        id="SPParametricStatistics.DistributionName.norm",
    ),
    pytest.param(
        "gemseo.uncertainty.statistic.ot_parametric",
        "OTParametricStatistics.FittingCriterion",
        "ChiSquared",
        "CHI_SQUARED",
        id="OTParametricStatistics.FittingCriterion.ChiSquared",
    ),
)
"""The enumerations whose members were renamed to the upper-case convention.

Each case is the module defining the enumeration, the path of the enumeration in the
module, an old member name and the new one.
"""


def _get_enumeration(module_name: str, path: str) -> type[StrEnum]:
    """Return an enumeration from its module.

    Args:
        module_name: The name of the module defining the enumeration.
        path: The dotted path of the enumeration in the module.

    Returns:
        The enumeration.
    """
    enumeration = importlib.import_module(module_name)
    for name in path.split("."):
        enumeration = getattr(enumeration, name)
    return enumeration


@pytest.mark.parametrize(
    ("module_name", "path", "old_name", "new_name"), _upper_cased_member_cases
)
def test_upper_cased_member_resolves(module_name, path, old_name, new_name):
    """The old name of a member renamed to the upper-case convention resolves.

    Args:
        module_name: The name of the module defining the enumeration.
        path: The dotted path of the enumeration in the module.
        old_name: The old name of the member.
        new_name: The new name of the member.
    """
    enumeration = _get_enumeration(module_name, path)

    with pytest.warns(DeprecationWarning, match=f"use '{new_name}' instead"):
        old_member = getattr(enumeration, old_name)

    assert old_member is enumeration[new_name]


@pytest.mark.parametrize(
    ("module_name", "path", "old_name", "new_name"),
    [
        *_upper_cased_member_cases,
        # The old name is not aliased as an attribute, which would shadow the method
        # `str.center` of the members.
        pytest.param(
            "gemseo.doe.pydoe.settings.pydoe_lhs",
            "Criterion",
            "center",
            "CENTER",
            id="Criterion.center",
        ),
    ],
)
def test_upper_cased_member_resolves_by_name(module_name, path, old_name, new_name):
    """The old name of a member renamed to the upper-case convention resolves by name.

    Args:
        module_name: The name of the module defining the enumeration.
        path: The dotted path of the enumeration in the module.
        old_name: The old name of the member.
        new_name: The new name of the member.
    """
    enumeration = _get_enumeration(module_name, path)

    with pytest.warns(DeprecationWarning, match=f"use '{new_name}' instead"):
        old_member = enumeration[old_name]

    assert old_member is enumeration[new_name]


def test_unknown_member_name_of_an_aliased_enumeration_raises(snapshot):
    """An unknown name of an enumeration with renamed members raises a `KeyError`.

    Args:
        snapshot: Fixture to compare the error message with a snapshot.
    """
    from gemseo.space.variable import DataType

    with assert_exception(KeyError, snapshot):
        DataType["DOES_NOT_EXIST"]


def test_nested_renames_of_a_shared_class_are_merged():
    """The renames of a nested class held by several classes are all applied.

    Each class holding it lists its own renames of it.
    """
    from gemseo._deprecation import _alias_class_and_nested_attributes

    class Shared(StrEnum):
        NEW_A = "a"
        NEW_B = "b"

    class First:
        Enumeration = Shared

    class Second:
        Enumeration = Shared

    _alias_class_and_nested_attributes(
        First, {"Enumeration.OLD_A": "Enumeration.NEW_A"}
    )
    _alias_class_and_nested_attributes(
        Second, {"Enumeration.OLD_B": "Enumeration.NEW_B"}
    )

    with pytest.warns(DeprecationWarning, match="'OLD_A'"):
        member_a = Shared.OLD_A
    with pytest.warns(DeprecationWarning, match="'OLD_B'"):
        member_b = Shared["OLD_B"]

    assert member_a is Shared.NEW_A
    assert member_b is Shared.NEW_B


def test_renames_added_later_apply_to_the_subclass_bodies():
    """A rename added to a class already aliased applies to its subclass bodies."""
    from gemseo._deprecation import _alias_class_attributes

    class Base:
        new_a = "a"
        new_b = "b"

    _alias_class_attributes(Base, {"OLD_A": "new_a"})
    _alias_class_attributes(Base, {"OLD_B": "new_b"})

    with pytest.warns(DeprecationWarning, match="'OLD_B'"):

        class Sub(Base):
            OLD_B = "other"

    assert Sub.new_b == "other"


def test_nested_rename_whose_path_leads_to_no_class_is_ignored():
    """A dotted rename is ignored when its path does not lead to a class."""
    from gemseo._deprecation import _alias_class_and_nested_attributes

    class Holder:
        not_a_class = 1

    _alias_class_and_nested_attributes(
        Holder, {"not_a_class.OLD": "not_a_class.NEW", "missing.OLD": "missing.NEW"}
    )

    assert "OLD" not in vars(Holder)


def test_renamed_hook_of_a_formulation_subclass_body():
    """A formulation implementing the old name of the input space hook still works.

    The implementation is remapped to the new name, which then no longer is abstract.
    """
    from gemseo.formulation.core.base import BaseFormulation

    with pytest.warns(
        DeprecationWarning,
        match=(
            "The attribute '_update_design_space' of the class 'BaseFormulation' is "
            "deprecated; use '_update_input_space' instead."
        ),
    ):

        class Formulation(BaseFormulation):
            def _update_design_space(self) -> None:
                """Update the input space."""

    assert Formulation._update_input_space is vars(Formulation)["_update_design_space"]
    assert "_update_input_space" not in Formulation.__abstractmethods__


def _create_module_renaming_an_attribute(
    is_live: bool, new_name: str, monkeypatch: pytest.MonkeyPatch
) -> ModuleType:
    """Create a module whose attribute `Old` is renamed, and register it.

    Args:
        is_live: Whether the module kept its name, or else is the stand-in of a
            renamed module.
        new_name: The fully-qualified new name of `Old`.
        monkeypatch: Fixture to patch the table of the renamed attributes and to
            register the module.

    Returns:
        The module.
    """
    from gemseo import _deprecation

    if is_live:
        module = ModuleType("gemseo.fake_live_module")
    else:
        module = _deprecation._DeprecatedModule("gemseo.fake_old_module")
        module.__dict__["_deprecation_target"] = ModuleType("gemseo.fake_new_module")
    monkeypatch.setattr(
        _deprecation, "attribute_renames", {module.__name__: {"Old": new_name}}
    )
    if is_live:
        _deprecation._install_attribute_aliases(module)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


_is_live_module: Final[pytest.MarkDecorator] = pytest.mark.parametrize(
    "is_live", [True, False], ids=["live", "stand-in"]
)
"""Parametrize a test with a module that kept its name and with a stand-in."""


@_is_live_module
def test_attribute_renamed_to_a_plugin(is_live, monkeypatch):
    """An attribute moved to a plugin resolves on a live module and on a stand-in.

    Args:
        is_live: Whether the module kept its name, or else is a stand-in.
        monkeypatch: Fixture to patch the table of the renamed attributes.
    """
    new_module = ModuleType("gemseo_fake_plugin.module")
    new_module.New = object()
    monkeypatch.setitem(sys.modules, new_module.__name__, new_module)
    new_name = "gemseo_fake_plugin.module.New"
    module = _create_module_renaming_an_attribute(is_live, new_name, monkeypatch)

    with pytest.warns(DeprecationWarning, match=f"use '{new_name}' instead"):
        new_object = module.Old

    assert new_object is new_module.New


_missing_plugin_name: Final[str] = "gemseo_missing_plugin.module.New"
"""The new name of an attribute moved to a plugin that is not installed."""


@_is_live_module
def test_attribute_renamed_to_a_missing_plugin_raises(is_live, monkeypatch, snapshot):
    """An attribute moved to a plugin that is not installed raises an error naming it.

    Args:
        is_live: Whether the module kept its name, or else is a stand-in.
        monkeypatch: Fixture to patch the table of the renamed attributes.
        snapshot: Fixture to compare the error message with a snapshot.
    """
    module = _create_module_renaming_an_attribute(
        is_live, _missing_plugin_name, monkeypatch
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with assert_exception(AttributeError, snapshot):
            module.Old  # noqa: B018


@_is_live_module
def test_attribute_renamed_to_a_missing_plugin_is_probeable(is_live, monkeypatch):
    """`hasattr` returns `False` for an attribute moved to a missing plugin.

    Args:
        is_live: Whether the module kept its name, or else is a stand-in.
        monkeypatch: Fixture to patch the table of the renamed attributes.
    """
    module = _create_module_renaming_an_attribute(
        is_live, _missing_plugin_name, monkeypatch
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        has_attribute = hasattr(module, "Old")

    assert not has_attribute


@_is_live_module
def test_attribute_renamed_to_a_missing_plugin_raises_when_imported_from_its_module(
    is_live, monkeypatch, snapshot
):
    """An attribute moved to a missing plugin raises when imported by a `from`.

    Args:
        is_live: Whether the module kept its name, or else is a stand-in.
        monkeypatch: Fixture to patch the table of the renamed attributes.
        snapshot: Fixture to compare the error message with a snapshot.
    """
    module = _create_module_renaming_an_attribute(
        is_live, _missing_plugin_name, monkeypatch
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with assert_exception(ImportError, snapshot):
            exec(f"from {module.__name__} import Old", {})  # noqa: S102


@_is_live_module
def test_missing_dependency_of_a_plugin_is_reraised(is_live, monkeypatch):
    """A dependency missing behind an attribute moved to a plugin is re-raised.

    It is not mistaken for the plugin not being installed.

    Args:
        is_live: Whether the module kept its name, or else is a stand-in.
        monkeypatch: Fixture to patch the table of the renamed attributes and the
            import of the new module.
    """
    from gemseo import _deprecation

    error = ModuleNotFoundError(
        "No module named 'some_optional_dependency'", name="some_optional_dependency"
    )

    def import_module(name: str) -> ModuleType:
        raise error

    module = _create_module_renaming_an_attribute(
        is_live, "gemseo_fake_plugin.module.New", monkeypatch
    )
    monkeypatch.setattr(_deprecation, "import_module", import_module)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(ModuleNotFoundError) as exc_info:
            module.Old  # noqa: B018

    assert exc_info.value is error


def test_renamed_class_attribute_warns_and_resolves():
    """The old name of a renamed class attribute warns and resolves to the new one."""
    from gemseo.dataset.dataset import Dataset

    with pytest.warns(DeprecationWarning, match="'DEFAULT_GROUP'"):
        old_value = Dataset.DEFAULT_GROUP

    assert old_value == Dataset.default_group


def test_renamed_aggregation_function_enum_warns_and_resolves():
    """The enumeration of the aggregation functions resolves under its old name."""
    from gemseo.discipline.constraint_aggregation import ConstraintAggregation

    with pytest.warns(DeprecationWarning, match="'EvaluationFunction'"):
        old_value = ConstraintAggregation.EvaluationFunction

    assert old_value is ConstraintAggregation.AggregationFunction


@pytest.mark.parametrize(
    ("old_name", "new_name"),
    [
        ("_EVALUATION_FUNCTION_MAP", "_aggregation_function_map"),
        ("_JACOBIAN_EVALUATION_FUNCTION_MAP", "_jacobian_aggregation_function_map"),
    ],
)
def test_renamed_aggregation_function_map_warns_and_resolves(old_name, new_name):
    """The maps of aggregation functions resolve under their old, master, names.

    They went through a develop-only intermediate name
    (`_evaluation_function_map`, `_jacobian_evaluation_function_map`),
    which `bump-version.yml` must skip: the old name it registers is the one
    of the last release, not the develop-only intermediate one.
    """
    from gemseo.discipline.constraint_aggregation import ConstraintAggregation

    with pytest.warns(DeprecationWarning, match=old_name):
        old_value = getattr(ConstraintAggregation, old_name)

    assert old_value is getattr(ConstraintAggregation, new_name)


def test_renamed_class_attribute_warns_and_resolves_through_an_instance():
    """The old name of a renamed class attribute also resolves through an instance."""
    from gemseo.dataset.dataset import Dataset

    dataset = Dataset()
    with pytest.warns(DeprecationWarning, match="'DEFAULT_GROUP'"):
        old_value = dataset.DEFAULT_GROUP

    assert old_value == dataset.default_group


def test_renamed_class_attribute_of_a_renamed_class_warns_and_resolves():
    """A class attribute renamed together with the class itself still resolves.

    The `classes:` section keys its block by the class's old name (`BaseMLAlgo`);
    the table is rekeyed to the class's current name (`BaseMLModel`) so this works.
    """
    from gemseo.machine_learning.core.model.base_ml_model import BaseMLModel

    with pytest.warns(DeprecationWarning, match="'SHORT_ALGO_NAME'"):
        old_value = BaseMLModel.SHORT_ALGO_NAME

    assert old_value == BaseMLModel.short_name


def test_renamed_class_attribute_of_the_standalone_sobieski_structure_resolves():
    """The standalone SobieskiStructure's renamed class attribute keeps resolving.

    A different class of the same name, `gemseo.problem.mdo.sobieski.discipline.
    SobieskiStructure` (the public discipline wrapper), shares no attribute with this
    one; the alias must still be installed here since this class does declare
    `stress_limit`.
    """
    from gemseo.problem.mdo.sobieski.standalone.structure import SobieskiStructure

    with pytest.warns(DeprecationWarning, match="'STRESS_LIMIT'"):
        old_value = SobieskiStructure.STRESS_LIMIT

    assert old_value == SobieskiStructure.stress_limit == 1.09


def test_renamed_enumeration_member_warns_and_resolves():
    """The old name of a renamed enumeration member warns and resolves.

    An enumeration is a class like any other for the `classes:` section, but its
    members live in the class namespace, so the data descriptor must not disturb
    them.
    """
    from gemseo.space.variable import DataType

    with pytest.warns(DeprecationWarning, match="'FLOAT'"):
        old_value = DataType.FLOAT

    assert old_value is DataType.REAL
    assert tuple(member.name for member in DataType) == (
        "CATALOG",
        "CATEGORICAL",
        "DISCRETE",
        "INTEGER",
        "REAL",
    )


def test_renamed_enumeration_member_of_a_namesake_class_is_left_alone():
    """A namesake enumeration keeps the member the table renames elsewhere.

    `SobieskiBase.DataType` shares its name with the data type of a variable but
    enumerates NumPy dtypes; it declares no `REAL` and its `FLOAT` is still live, so
    the guards of the installer must leave it untouched.
    """
    from gemseo.problem.mdo.sobieski.standalone.util import SobieskiBase

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert SobieskiBase.DataType.FLOAT == "float64"


def test_renamed_protected_class_attribute_warns_and_resolves():
    """A renamed protected (`_`-prefixed) class attribute also warns and resolves.

    The public renames above are exercised through `Dataset`, whose renamed
    attributes are plain class constants; `Serializable._ATTR_NOT_TO_SERIALIZE`
    plays the same role here for a protected name, its new name
    `_attr_not_to_serialize` being a plain class attribute rather than one set only
    in `__init__`, so it resolves on the class itself.
    """
    from gemseo.core.serializable import Serializable

    with pytest.warns(DeprecationWarning, match="'_ATTR_NOT_TO_SERIALIZE'"):
        old_value = Serializable._ATTR_NOT_TO_SERIALIZE

    assert old_value is Serializable._attr_not_to_serialize


def test_renamed_class_attribute_skips_an_unrelated_class_of_the_same_name():
    """A `classes:` entry is not applied to a different class sharing its name.

    `gemseo.problem.mdo.sobieski.discipline.SobieskiStructure` (the public
    discipline wrapper) merely shares its name with `gemseo.problem.mdo.sobieski.
    standalone.structure.SobieskiStructure`, the class the `STRESS_LIMIT` rename
    entry is written for; the wrapper never had `STRESS_LIMIT` and must not be given
    the alias.
    """
    from gemseo.problem.mdo.sobieski.discipline import SobieskiStructure

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert not hasattr(SobieskiStructure, "STRESS_LIMIT")


def test_setting_a_renamed_class_attribute_on_an_instance_writes_the_new_name():
    """Setting the old name on an instance writes the new attribute instead."""
    from gemseo.dataset.dataset import Dataset

    dataset = Dataset()
    with pytest.warns(DeprecationWarning, match="'DEFAULT_GROUP'"):
        dataset.DEFAULT_GROUP = "custom_group"

    assert dataset.default_group == "custom_group"


def test_subclass_body_assigning_a_renamed_class_attribute_warns_and_remaps():
    """A subclass body that still assigns the old name is remapped to the new one."""
    from gemseo.machine_learning.regression.core.base_regressor import BaseRegressor

    with pytest.warns(DeprecationWarning, match="'DEFAULT_TRANSFORMER'"):

        class NewRegressor(BaseRegressor):
            DEFAULT_TRANSFORMER: ClassVar = {"scale": 1}

    assert NewRegressor.default_transformer == {"scale": 1}


def test_subclass_body_assigning_both_names_keeps_the_new_value():
    """A subclass assigning both the old and the new name keeps the new value."""
    from gemseo.machine_learning.regression.core.base_regressor import BaseRegressor

    class NewRegressor(BaseRegressor):
        DEFAULT_TRANSFORMER: ClassVar = {"old": True}
        default_transformer: ClassVar = {"new": True}

    assert NewRegressor.default_transformer == {"new": True}


def test_subclass_body_assigning_a_renamed_protected_class_attribute_warns_and_remaps():
    """A subclass body assigning an old protected class attribute is remapped.

    Regression test: before protected (`_`-prefixed) class attributes were covered
    by the `classes:` table, nothing remapped a subclass body still assigning the
    old name (e.g. `_ATTR_NOT_TO_SERIALIZE`) to the new one
    (`_attr_not_to_serialize`), so the assignment was silently dropped and the base
    class's default value was used instead, e.g. silently breaking the pickling of
    a discipline overriding which attributes not to serialize.
    """
    from gemseo.core.discipline import Discipline

    new_value = Discipline._attr_not_to_serialize.union(["_lock"])

    with pytest.warns(DeprecationWarning, match="'_ATTR_NOT_TO_SERIALIZE'"):

        class NewDiscipline(Discipline):
            _ATTR_NOT_TO_SERIALIZE: ClassVar = new_value

    assert NewDiscipline._attr_not_to_serialize == new_value


def test_subclass_body_assigning_renamed_protected_factory_attribute_warns_and_remaps():
    """A factory subclass body assigning old protected names is remapped.

    Same regression as above, on the other documented plugin extension point: a
    [BaseFactory][gemseo.core.base_factory.BaseFactory] subclass is meant to
    override `_class` and `_package_names` (see its docstring), so a subclass whose
    body still assigns the old names, `_CLASS` and `_PACKAGE_NAMES`, must have both
    remapped.
    """
    from gemseo.core.base_factory import BaseFactory

    with pytest.warns(DeprecationWarning, match="'_CLASS'") as records:

        class NewFactory(BaseFactory):
            _CLASS: ClassVar = object
            _PACKAGE_NAMES: ClassVar = ("gemseo",)

    messages = [str(record.message) for record in records]
    assert (
        "The attribute '_CLASS' of the class 'BaseFactory' is deprecated; "
        "use '_class' instead." in messages
    )
    assert (
        "The attribute '_PACKAGE_NAMES' of the class 'BaseFactory' is deprecated; "
        "use '_package_names' instead." in messages
    )
    assert NewFactory._class is object
    assert NewFactory._package_names == ("gemseo",)


def test_unknown_attribute_of_an_aliased_class_raises(snapshot):
    """An attribute unrelated to any rename still raises `AttributeError`."""
    from gemseo.dataset.dataset import Dataset

    with assert_exception(AttributeError, snapshot):
        Dataset.does_not_exist_at_all  # noqa: B018


def test_existing_init_subclass_hook_still_runs_after_injection():
    """A base class's own `__init_subclass__` still runs after alias injection."""
    from gemseo._deprecation import _alias_class_attributes

    created_subclasses = []

    class Base:
        def __init_subclass__(cls, **kwargs) -> None:
            super().__init_subclass__(**kwargs)
            created_subclasses.append(cls)

    _alias_class_attributes(Base, {"OLD": "new"})

    class Sub(Base):
        pass

    assert created_subclasses == [Sub]


def test_every_class_attribute_rename_is_reachable():
    """Every class-attribute-rename entry aliases a live, importable class.

    A class is found by walking every `gemseo` module (skipping the ones an optional
    dependency prevents from importing) and looking for a class defined there whose
    name is a `class_attribute_renames` key. Several classes may share that name, and
    `_alias_class_attributes` skips a homonym that does not declare the new name, so
    for each of its old names, the alias must be installed (a `_RenamedClassAttribute`
    descriptor) on at least one of the classes sharing the name, or, for a stale table
    entry, the old name must still be live on at least one of them.
    """
    from gemseo._deprecation import _RenamedClassAttribute
    from gemseo._deprecation.aliases import class_attribute_renames

    def check_rename_is_reachable(classes: list[type], old_name: str) -> bool:
        """Check whether a rename's old name resolves on at least one class.

        Args:
            classes: The classes sharing the name the rename entry is keyed by.
            old_name: The old attribute name to check.

        Returns:
            Whether at least one class either carries the `_RenamedClassAttribute`
            descriptor for `old_name`, or still exposes `old_name` live (a stale
            entry correctly left alone).
        """
        # A dotted old name is that of an attribute of a nested class.
        *path, old_name = old_name.split(".")
        for cls in classes:
            owner = cls
            for name in path:
                owner = getattr(owner, name)
            if isinstance(owner.__dict__.get(old_name), _RenamedClassAttribute):
                return True
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                if hasattr(owner, old_name):
                    # A stale entry correctly left alone: the old name is still live.
                    return True
        return False

    root = Path(gemseo.__file__).parent
    found: dict[str, list[type]] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for module_info in pkgutil.walk_packages([str(root)], prefix="gemseo."):
            try:
                module = importlib.import_module(module_info.name)
            except ImportError:
                continue
            for obj in list(vars(module).values()):
                if (
                    isinstance(obj, type)
                    and getattr(obj, "__module__", None) == module.__name__
                    and obj.__name__ in class_attribute_renames
                ):
                    found.setdefault(obj.__name__, []).append(obj)

    missing = sorted(set(class_attribute_renames) - set(found))
    assert not missing, f"registered class(es) not found live: {missing}"

    broken = []
    for name, classes in found.items():
        for old_name, new_name in class_attribute_renames[name].items():
            if check_rename_is_reachable(classes, old_name):
                continue
            broken.append(f"{name}.{old_name} -> {new_name}")
    assert not broken, f"alias not installed for: {broken}"
