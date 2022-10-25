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

from gemseo.utils.logging_tools import LoggingContext


def test_default():
    """Check the default configuration of the LoggingContext."""
    context = LoggingContext()
    assert context.logger == logging.root
    assert context.level == logging.WARNING
    assert context.handler is None
    assert context.close


def test_custom():
    """Check the default configuration of the LoggingContext."""
    context = LoggingContext(
        level=logging.ERROR, logger=logging.getLogger("foo"), close=False, handler="bar"
    )
    assert context.level == logging.ERROR
    assert context.logger.name == "foo"
    assert context.handler == "bar"
    assert not context.close


def test_selective_logging(caplog):
    """Check logging with LoggingContext.

    The LoggingContext changes the level of the logger that is passed to it to WARNING:
    all the messages logged by this logger with INFO level will be silent.
    """
    caplog.set_level(logging.INFO)
    logger = logging.getLogger()
    logger.info("1. This should appear.")
    with LoggingContext():
        logger.warning("2. This should appear.")
        logger.info("3. This should not appear.")

    logger.info("4. This should appear.")
    with LoggingContext(level=logging.ERROR):
        logger.warning("5. This should not appear.")
        logger.error("6. This should appear.")

    assert "1. This should appear." in caplog.text
    assert "2. This should appear." in caplog.text
    assert "3. This should not appear." not in caplog.text
    assert "4. This should appear." in caplog.text
    assert "5. This should not appear." not in caplog.text
    assert "6. This should appear." in caplog.text
