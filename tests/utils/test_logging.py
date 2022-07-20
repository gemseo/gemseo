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
import logging

from gemseo.utils.logging_tools import LoggingContext


def log_with_selective_logging() -> None:
    """Log with selective logging context manager."""
    logger = logging.getLogger()
    logger.info("1. This should appear.")
    with LoggingContext(logger, logging.WARNING):
        logger.warning("2. This should appear.")
        logger.info("3. This should not appear.")

    logger.info("4. This should appear.")


def test_selective_logging(caplog):
    """Check the selective logging context manager."""
    caplog.set_level(logging.INFO)
    log_with_selective_logging()
    for i in [1, 2, 4]:
        assert f"{i}. This should appear." in caplog.text

    assert "3. This should not appear." not in caplog.text
