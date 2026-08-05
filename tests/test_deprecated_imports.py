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
import subprocess
import sys
import warnings

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
    from gemseo.util.constant import INFINITE_INT

    with pytest.warns(DeprecationWarning, match="'gemseo.util.constant.INFINITE_INT'"):
        from gemseo.utils.constants import C_LONG_MAX

    assert C_LONG_MAX == INFINITE_INT


def test_moved_module_level_constant():
    """A constant moved to another module is reachable from the old path."""
    from gemseo.util.constant import EPSILON

    with pytest.warns(DeprecationWarning, match="'gemseo.util.constant.EPSILON'"):
        from gemseo.utils.derivatives.error_estimators import EPSILON as OLD_EPSILON

    assert OLD_EPSILON == EPSILON


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

    The user warning filters are left alone at the same time, as both depend on the
    state that `install` finds rather than on the alias tables.
    """
    from gemseo import _deprecation

    module_name = "gemseo.post.dataset.radviz"
    monkeypatch.delitem(sys.modules, module_name)
    monkeypatch.setattr(_deprecation, "_installed", False)
    monkeypatch.setattr(_deprecation, "LIVE_ALIASED_MODULES", frozenset({module_name}))
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
    """Every rename-table entry redirects to an importable target."""
    from gemseo._deprecation.aliases import ATTRIBUTE_RENAMES
    from gemseo._deprecation.aliases import MODULE_RENAMES

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for old_module in MODULE_RENAMES:
            importlib.import_module(old_module)
        for old_module, renames in ATTRIBUTE_RENAMES.items():
            module = importlib.import_module(old_module)
            for old_name in renames:
                getattr(module, old_name)


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
        # Class rename applies via ATTRIBUTE_RENAMES.
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
