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
from logging import getLogger
from pathlib import Path

import pytest

from gemseo.utils.logging import LoggingConfiguration
from gemseo.utils.logging import LoggingContext
from gemseo.utils.logging import OneLineLogging


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


def test_configured_loggers():
    """Check that LoggingConfiguration configures the loggers for GEMSEO and plugins."""
    gemseo_logger = getLogger("gemseo")
    gemseo_logger.level = logging.NOTSET
    gemseo_plugin_logger = getLogger("gemseo_plugin")
    gemseo_module_logger = getLogger("gemseo.module")
    for logger in (gemseo_logger, gemseo_plugin_logger, gemseo_module_logger):
        assert logger.level == logging.NOTSET

    LoggingConfiguration(level=logging.WARNING)
    assert gemseo_module_logger.level == logging.NOTSET
    for logger in (gemseo_logger, gemseo_plugin_logger):
        assert logger.level == logging.WARNING


@pytest.fixture
def root_logger():
    """Yield the root logger with its initial level and handlers restored."""
    logger = logging.getLogger()
    initial_level = logger.level
    initial_handlers = logger.handlers[:]
    yield logger
    logger.level = initial_level
    logger.handlers = initial_handlers


def test_configure_root_logger_default(root_logger):
    """Root logger untouched when `configure_root_logger` is False (the default)."""
    initial_level = root_logger.level
    initial_handlers = root_logger.handlers[:]

    LoggingConfiguration(level=logging.WARNING)

    assert root_logger.level == initial_level
    assert root_logger.handlers == initial_handlers


def test_configure_root_logger_enabled(root_logger):
    """Root logger configured when `configure_root_logger` is True."""
    LoggingConfiguration(level=logging.WARNING, configure_root_logger=True)

    assert root_logger.level == logging.WARNING
    assert any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers)

    # The GEMSEO logger is still configured alongside the root logger.
    gemseo_logger = getLogger("gemseo")
    assert gemseo_logger.level == logging.WARNING
    assert any(isinstance(h, logging.StreamHandler) for h in gemseo_logger.handlers)

    LoggingConfiguration(enable=False, configure_root_logger=True)
    assert root_logger.handlers == []


def test_configure_root_logger_disables_gemseo_propagation(root_logger):
    """GEMSEO loggers stop propagating when the root logger is configured.

    Otherwise, the same record would be emitted by the GEMSEO handler and
    the root handler.
    """
    gemseo_logger = getLogger("gemseo")
    LoggingConfiguration(configure_root_logger=True)
    assert not gemseo_logger.propagate

    LoggingConfiguration(configure_root_logger=False)
    assert gemseo_logger.propagate
