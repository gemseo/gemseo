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
#    INITIAL AUTHORS - initial API and implementation and/or
#                      initial documentation
#        :author:  Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import logging
from importlib.metadata import EntryPoint
from logging import Logger
from logging import getLogger
from logging import root
from pathlib import Path

import pytest

from gemseo.core.base_factory import BaseFactory
from gemseo.util.logging import LoggingConfiguration
from gemseo.util.logging import LoggingContext
from gemseo.util.logging import OneLineLogging
from gemseo.util.logging import _get_logger_names
from gemseo.util.testing.helper import assert_exception


def test_default() -> None:
    """Check the default configuration of the LoggingContext."""
    context = LoggingContext()
    assert context.logger == logging.getLogger("gemseo")
    assert context.level == logging.WARNING
    assert context.handler is None
    assert context.close


def test_custom() -> None:
    """Check the default configuration of the LoggingContext."""
    context = LoggingContext(
        logging.getLogger("foo"), level=logging.ERROR, close=False, handler="bar"
    )
    assert context.level == logging.ERROR
    assert context.logger.name == "foo"
    assert context.handler == "bar"
    assert not context.close


@pytest.mark.parametrize("close", [False, True])
def test_handler(tmp_wd, close) -> None:
    """Check the use of a handler."""
    file_path = Path("log.txt")
    handler = logging.FileHandler(file_path)
    handler.set_name("handler_name")
    logger = logging.getLogger()
    with LoggingContext(logger, handler=handler, close=close):
        logger.info("foo")
        logger.warning("bar")

    log = file_path.read_text()

    assert "foo" not in log
    assert "bar" in log
    assert (handler.name in logging._handlers) is not close


def test_selective_logging(caplog) -> None:
    """Check logging with LoggingContext.

    The LoggingContext changes the level of the logger that is passed to it to WARNING:
    all the messages logged by this logger with INFO level will be silent.
    """
    caplog.set_level(logging.INFO)
    logger = logging.getLogger()
    logger.info("1. This should appear.")
    with LoggingContext(logging.root):
        logger.warning("2. This should appear.")
        logger.info("3. This should not appear.")

    logger.info("4. This should appear.")
    with LoggingContext(logging.root, level=logging.ERROR):
        logger.warning("5. This should not appear.")
        logger.error("6. This should appear.")

    with LoggingContext(logging.root, level=None):
        logger.info("7. This should appear.")

    assert "1. This should appear." in caplog.text
    assert "2. This should appear." in caplog.text
    assert "3. This should not appear." not in caplog.text
    assert "4. This should appear." in caplog.text
    assert "5. This should not appear." not in caplog.text
    assert "6. This should appear." in caplog.text
    assert "7. This should appear." in caplog.text


@pytest.mark.parametrize(("propagate", "name"), [(False, "foo"), (True, "root")])
def test_propagate(propagate, name) -> None:
    """Check that the while-loop breaks when logger.propagate is False."""
    logger = logging.getLogger("foo")
    logger.propagate = propagate
    logger.handlers = []
    context = OneLineLogging(logger)
    with context:
        pass

    assert context._OneLineLogging__logger.name == name


def test_while_false() -> None:
    """Check that the while-loop stops when logger.parent is None."""
    context = OneLineLogging(logging.root)
    with context:
        pass


@pytest.fixture(autouse=True)
def restore_loggers():
    """Restore the level, handlers and propagation of the loggers.

    Configuring a logger leaks into every subsequent test, which then may capture
    nothing from it, so every test of this module is undone afterwards.
    The loggers created by a test are reset.
    """
    loggers = (root, *_get_loggers())
    states = [
        (logger.level, logger.handlers[:], logger.propagate) for logger in loggers
    ]
    yield
    for logger, (level, handlers, propagate) in zip(loggers, states, strict=True):
        logger.level = level
        logger.handlers = handlers
        logger.propagate = propagate

    for logger in set(_get_loggers()).difference(loggers):
        logger.level = logging.NOTSET
        logger.handlers = []
        logger.propagate = True


def _get_loggers() -> tuple[Logger, ...]:
    """Return the loggers of the logger dictionary, except the placeholders.

    Returns:
        The loggers.
    """
    return tuple(
        logger
        for logger in root.manager.loggerDict.copy().values()
        if isinstance(logger, Logger)
    )


@pytest.fixture
def declare_plugin(monkeypatch, tmp_path):
    """Return a function declaring the only GEMSEO plugins."""

    def declare(*values: str, search_path_modules: tuple[str, ...] = ()) -> None:
        """Declare the plugins.

        Args:
            *values: The values of the entry points.
            search_path_modules: The names of the modules
                in the directory of the environment variable `GEMSEO_PATH`.
        """
        entry_points = tuple(
            EntryPoint(name=value, value=value, group=BaseFactory.plugin_entry_point)
            for value in values
        )
        monkeypatch.setattr(
            "gemseo.core.base_factory.entry_points",
            lambda group: (
                entry_points if group == BaseFactory.plugin_entry_point else ()
            ),
        )
        if search_path_modules:
            for name in search_path_modules:
                (tmp_path / f"{name}.py").touch()
            monkeypatch.setenv(BaseFactory._env_var_with_search_paths, str(tmp_path))
        else:
            monkeypatch.delenv(BaseFactory._env_var_with_search_paths, raising=False)
        _get_logger_names.cache_clear()

    yield declare
    _get_logger_names.cache_clear()


@pytest.mark.parametrize(
    ("values", "search_path_modules", "logger_name"),
    [
        pytest.param(("gemseo_test",), (), "gemseo_test", id="gemseo_package"),
        pytest.param(
            ("test_lib.gemseo_plugin",),
            (),
            "test_lib.gemseo_plugin",
            id="other_package_module",
        ),
        pytest.param(
            (),
            ("test_search_path_plugin",),
            "test_search_path_plugin",
            id="search_path",
        ),
    ],
)
def test_configured_loggers(declare_plugin, values, search_path_modules, logger_name):
    """Check that LoggingConfiguration configures the loggers of the plugins.

    The plugins are declared by entry points or found in the search paths,
    and not imported.
    Only the logger of the plugin module gets the level.
    """
    declare_plugin(*values, search_path_modules=search_path_modules)
    module_logger = getLogger(f"{logger_name}.module")
    LoggingConfiguration(level=logging.WARNING)
    assert getLogger(logger_name).level == logging.WARNING
    assert module_logger.level == logging.NOTSET


def test_search_path_changed(declare_plugin, monkeypatch, tmp_path):
    """Check that a new search path is taken into account by a new configuration."""
    declare_plugin()
    LoggingConfiguration(level=logging.WARNING)
    assert getLogger("test_search_path_plugin").level == logging.NOTSET
    (tmp_path / "test_search_path_plugin.py").touch()
    monkeypatch.setenv(BaseFactory._env_var_with_search_paths, str(tmp_path))
    LoggingConfiguration(level=logging.WARNING)
    assert getLogger("test_search_path_plugin").level == logging.WARNING


def test_records_emitted_once(declare_plugin, capsys):
    """Check that the records of the modules of a plugin are emitted once.

    The plugin declares nested modules by entry points
    and the configuration is applied twice.
    """
    declare_plugin("gemseo_test", "gemseo_test.x")
    LoggingConfiguration(level=logging.WARNING)
    LoggingConfiguration(level=logging.WARNING)
    module_names = ("gemseo_test.x", "gemseo_test.x.y", "gemseo_test.z")
    for module_name in module_names:
        getLogger(module_name).warning("record of %s", module_name)

    error = capsys.readouterr().err
    for module_name in module_names:
        assert error.count(f"record of {module_name}\n") == 1


@pytest.mark.parametrize(
    ("values", "logger_name"),
    [
        pytest.param((), "gemseo_test_imported", id="undeclared_gemseo_package"),
        pytest.param(("test_lib.gemseo_plugin",), "test_lib", id="host_library"),
    ],
)
def test_other_loggers_untouched(declare_plugin, values, logger_name):
    """Check that the loggers of the modules that are not plugins are left alone."""
    declare_plugin(*values)
    logger = getLogger(logger_name)
    handler = logging.NullHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.ERROR)
    LoggingConfiguration(configure_root_logger=True)
    assert logger.handlers == [handler]
    assert logger.level == logging.ERROR
    assert logger.propagate


def test_disabled_plugin_loggers_without_handlers(declare_plugin):
    """Check that disabling the logs removes the handlers of the plugin loggers."""
    declare_plugin("gemseo_test")
    LoggingConfiguration()
    LoggingConfiguration(enable=False)
    assert not getLogger("gemseo_test").handlers


def test_configure_root_logger_default():
    """Root logger untouched when `configure_root_logger` is False (the default)."""
    initial_level = root.level
    initial_handlers = root.handlers[:]

    LoggingConfiguration(level=logging.WARNING)

    assert root.level == initial_level
    assert root.handlers == initial_handlers


def test_configure_root_logger_enabled():
    """Root logger configured when `configure_root_logger` is True."""
    LoggingConfiguration(level=logging.WARNING, configure_root_logger=True)

    assert root.level == logging.WARNING
    assert any(isinstance(h, logging.StreamHandler) for h in root.handlers)

    # The GEMSEO logger is still configured alongside the root logger.
    gemseo_logger = getLogger("gemseo")
    assert gemseo_logger.level == logging.WARNING
    assert any(isinstance(h, logging.StreamHandler) for h in gemseo_logger.handlers)

    LoggingConfiguration(enable=False, configure_root_logger=True)
    assert root.handlers == []


def test_configure_root_logger_disables_gemseo_propagation():
    """GEMSEO loggers stop propagating when the root logger is configured.

    Otherwise, the same record would be emitted by the GEMSEO handler and
    the root handler.
    """
    gemseo_logger = getLogger("gemseo")
    LoggingConfiguration(configure_root_logger=True)
    assert not gemseo_logger.propagate

    LoggingConfiguration(configure_root_logger=False)
    assert gemseo_logger.propagate


def test_cannot_disable_root_logger(snapshot):
    """The root logger configuration cannot be disabled once it is enabled."""
    configuration = LoggingConfiguration(configure_root_logger=True)

    with assert_exception(ValueError, snapshot):
        configuration.configure_root_logger = False

    # The loggers are not reconfigured.
    assert not getLogger("gemseo").propagate
