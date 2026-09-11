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
import os
import pickle
import pkgutil
import subprocess
import sys
import warnings
from pathlib import Path
from typing import ClassVar

import pytest

import gemseo  # noqa: F401 - ensures the deprecated-import finder is installed
from gemseo.util.testing.helper import assert_exception


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


def test_dir_of_deprecated_module():
    """`dir` on an old path exposes the names of the new one."""
    from gemseo.util import string

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = importlib.import_module("gemseo.utils.string_tools")

    assert "MultiLineString" in dir(old)
    assert set(dir(string)) <= set(dir(old))


def test_star_import_from_dissolved_package(monkeypatch):
    """A star import from the dissolved settings package binds its former names."""
    from gemseo.optimization import SLSQP_Settings

    monkeypatch.delitem(sys.modules, "gemseo.settings", raising=False)
    namespace: dict[str, object] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        exec("from gemseo.settings import *", namespace)  # noqa: S102

    assert namespace["SLSQP_Settings"] is SLSQP_Settings


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
    """The dissolved gemseo.settings package resolves attributes across its targets."""
    from gemseo.formulation import MDF_Settings
    from gemseo.optimization import SLSQP_Settings

    monkeypatch.delitem(sys.modules, "gemseo.settings", raising=False)
    with pytest.warns(DeprecationWarning, match="'gemseo.settings' is deprecated"):
        settings = importlib.import_module("gemseo.settings")

    assert settings.SLSQP_Settings is SLSQP_Settings
    assert settings.MDF_Settings is MDF_Settings


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
    """An unknown attribute of a dissolved package raises AttributeError."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        settings = importlib.import_module("gemseo.settings")

    with assert_exception(AttributeError, snapshot):
        settings.does_not_exist  # noqa: B018


def test_every_rename_entry_is_reachable():
    """Every rename-table entry redirects to an importable target.

    The names whose migration cannot be automated are excluded:
    importing one raises instead of resolving.
    """
    from gemseo._deprecation.aliases import attribute_renames
    from gemseo._deprecation.aliases import manual_migrations
    from gemseo._deprecation.aliases import module_renames

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for old_module in module_renames:
            importlib.import_module(old_module)
        for old_module, renames in attribute_renames.items():
            module = importlib.import_module(old_module)
            manual_names = manual_migrations.get(old_module, {})
            for old_name in renames:
                if old_name not in manual_names:
                    getattr(module, old_name)


def test_manual_migration_raises(snapshot):
    """A name whose migration cannot be automated raises instead of being aliased.

    An `ImportError` is raised rather than an `AttributeError`, so that a
    `from ... import ...` of the name reports this message instead of the generic
    one that the import machinery builds from an `AttributeError`.

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
    """Every entry of the manual-migration table raises instead of being aliased."""
    from gemseo._deprecation.aliases import manual_migrations

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for old_module, migrations in manual_migrations.items():
            module = importlib.import_module(old_module)
            for old_name in migrations:
                with pytest.raises(ImportError):
                    getattr(module, old_name)


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
            "gemseo.settings": {"Animation": "gemseo.post.Animation"},
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


def test_renamed_class_attribute_warns_and_resolves():
    """The old name of a renamed class attribute warns and resolves to the new one."""
    from gemseo.dataset.dataset import Dataset

    with pytest.warns(DeprecationWarning, match="'DEFAULT_GROUP'"):
        old_value = Dataset.DEFAULT_GROUP

    assert old_value == Dataset.default_group


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


def test_renamed_protected_class_attribute_warns_and_resolves():
    """A renamed protected (`_`-prefixed) class attribute also warns and resolves.

    The public renames above are exercised through `Dataset`, whose renamed
    attributes are plain class constants; `TerminationCriterion._MESSAGE` plays the
    same role here for a protected name, its new name `_message` being a plain
    class attribute rather than one set only in `__init__`, so it resolves on the
    class itself.
    """
    from gemseo.core.problem.termination_criterion import TerminationCriterion

    with pytest.warns(DeprecationWarning, match="'_MESSAGE'"):
        old_value = TerminationCriterion._MESSAGE

    assert old_value is TerminationCriterion._message


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
        for cls in classes:
            if isinstance(cls.__dict__.get(old_name), _RenamedClassAttribute):
                return True
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                if hasattr(cls, old_name):
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
