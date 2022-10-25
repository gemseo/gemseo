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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Antoine Dechaume
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Logging tools
=============
"""
from __future__ import annotations

import logging
import types


class MultiLineHandlerMixin:
    """Stateless mixin class to override logging handlers behavior."""

    @staticmethod
    def __get_raw_record_message(record):
        """Return the raw message of a log record."""
        return record.msg

    def emit(self, record):
        """Emit one logging message per input record line."""
        # compute the message without the logging prefixes (timestamp, level, ...)
        message = record.getMessage()
        # replace getMessage so the message is not computed again when emitting
        # each line
        record.getMessage = types.MethodType(self.__get_raw_record_message, record)
        # backup old raw message
        old_msg = record.msg
        for line in message.split("\n"):
            record.msg = line
            super().emit(record)
        # restore genuine getMessage and raw message for other handlers
        record.getMessage = types.MethodType(logging.LogRecord.getMessage, record)
        record.msg = old_msg


class MultiLineStreamHandler(MultiLineHandlerMixin, logging.StreamHandler):
    """StreamHandler to split multiline logging messages."""


class MultiLineFileHandler(MultiLineHandlerMixin, logging.FileHandler):
    """FileHandler to split multiline logging messages."""


class LoggingContext:
    """Context manager for selective logging.

    Change the level of the logger in a ``with`` block.

    Examples:
        >>> logger = logger.getLogger()
        >>> logger.setLevel(logging.INFO)
        >>> logger.info("This should appear.")
        >>> with LoggingContext():
        >>>    logger.warning("This should appear.")
        >>>    logger.info("This should not appear.")
        >>>
        >>> logger.info("This should appear.")

    Source: `Logging Cookbook
    <https://docs.python.org/3/howto/
    logging-cookbook.html#using-a-context-manager-for-selective-logging>`_
    """

    def __init__(
        self,
        logger: logging.Logger = logging.root,
        level: int = logging.WARNING,
        handler: logging.StreamHandler | None = None,
        close: bool = True,
    ) -> None:
        """
        Args:
            logger: The logger.
            level: The level of the logger to be used on block entry.
            handler: An additional handler to be used on block entry.
            close: Whether to close the handler on block exit.
        """
        self.logger = logger
        self.level = level
        self.handler = handler
        self.close = close

    def __enter__(self) -> None:
        if self.level is not None:
            self.old_level = self.logger.level
            self.logger.setLevel(self.level)
        if self.handler:
            self.logger.addHandler(self.handler)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.level is not None:
            self.logger.setLevel(self.old_level)
        if self.handler:
            self.logger.removeHandler(self.handler)
        if self.handler and self.close:
            self.handler.close()
        # implicit return of None => don't swallow exceptions
