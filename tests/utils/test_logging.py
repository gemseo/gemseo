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
from pathlib import Path

import pytest

from gemseo.utils.logging_tools import LoggingContext
from gemseo.utils.logging_tools import OneLineLogging


def test_default():
    """Check the default configuration of the LoggingContext."""
    context = LoggingContext(logging.root)
    assert context.logger == logging.root
    assert context.level == logging.WARNING
    assert context.handler is None
    assert context.close


def test_custom():
    """Check the default configuration of the LoggingContext."""
    context = LoggingContext(
        logging.getLogger("foo"), level=logging.ERROR, close=False, handler="bar"
    )
    assert context.level == logging.ERROR
    assert context.logger.name == "foo"
    assert context.handler == "bar"
    assert not context.close


@pytest.mark.parametrize("close", [False, True])
def test_handler(tmp_wd, close):
    """Check the use of a handler."""
    file_path = Path("log.txt")
    handler = logging.FileHandler(file_path)
    handler.set_name("handler_name")
    logger = logging.getLogger()
    with LoggingContext(logger, handler=handler, close=close):
        logger.info("foo")
        logger.warning("bar")

    with file_path.open("r") as f:
        log = f.read()

    assert "foo" not in log
    assert "bar" in log
    assert (handler.name in logging._handlers) is not close


def test_selective_logging(caplog):
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
def test_propagate(propagate, name):
    """Check that the while-loop breaks when logger.propagate is False."""
    logger = logging.getLogger("foo")
    logger.propagate = propagate
    logger.handlers = []
    context = OneLineLogging(logger)
    with context:
        pass

    assert context._OneLineLogging__logger.name == name


def test_while_false():
    """Check that the while-loop stops when logger.parent is None."""
    context = OneLineLogging(logging.root)
    with context:
        pass
