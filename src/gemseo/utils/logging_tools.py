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
"""Logging tools."""

from __future__ import annotations

import types
from dataclasses import dataclass
from logging import WARNING
from logging import FileHandler
from logging import Formatter
from logging import Logger
from logging import LogRecord
from logging import StreamHandler
from logging import getLogger
from sys import stderr
from sys import stdout
from types import TracebackType
from typing import Final

DEFAULT_DATE_FORMAT: Final[str] = "%H:%M:%S"
"""The format of the date of the logged message."""

DEFAULT_MESSAGE_FORMAT: Final[str] = "%(levelname)8s - %(asctime)s: %(message)s"
"""The format of the logged message."""


GEMSEO_LOGGER: Final[Logger] = getLogger("gemseo")
"""The GEMSEO's logger."""


@dataclass
class LoggingSettings:
    """The settings of a logger."""

    date_format: str = DEFAULT_DATE_FORMAT
    """The format of the date of the logged message."""

    message_format: str = DEFAULT_MESSAGE_FORMAT
    """The format of the logged message."""

    logger: Logger = GEMSEO_LOGGER
    """The logger."""


LOGGING_SETTINGS = LoggingSettings()
"""The logging settings.

The parameters are changed by :func:`.configure_logger`.
"""


class MultiLineHandlerMixin:
    """Stateless mixin class to override logging handlers behavior."""

    @staticmethod
    def __get_raw_record_message(record):
        """Return the raw message of a log record."""
        return record.msg

    def emit(self, record) -> None:
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
        record.getMessage = types.MethodType(LogRecord.getMessage, record)
        record.msg = old_msg


class MultiLineStreamHandler(MultiLineHandlerMixin, StreamHandler):
    """StreamHandler to split multiline logging messages."""


class MultiLineFileHandler(MultiLineHandlerMixin, FileHandler):
    """FileHandler to split multiline logging messages."""


class LoggingContext:
    """Context manager for selective logging.

    Change the level of the logger in a ``with`` block.

    Examples:
        >>> import logging
        >>> logger = logging.getLogger()
        >>> logger.setLevel(logging.INFO)
        >>> logger.info("This should appear.")
        >>> with LoggingContext(logger):
        >>>    logger.warning("This should appear.")
        >>>    logger.info("This should not appear.")
        >>>
        >>> logger.info("This should appear.")

    Source: `Logging Cookbook
    <https://docs.python.org/3/howto/
    logging-cookbook.html#using-a-context-manager-for-selective-logging>`_
    """

    logger: Logger
    """The logger."""

    level: int | None
    """The level of the logger to be used on block entry.

    If ``None``, do not change the level of the logger.
    """

    handler: StreamHandler
    """An additional handler to be used on block entry."""

    close: bool
    """Whether to close the handler on block exit."""

    def __init__(
        self,
        logger: Logger,
        level: int | None = WARNING,
        handler: StreamHandler | None = None,
        close: bool = True,
    ) -> None:
        """
        Args:
            logger: The logger.
            level: The level of the logger to be used on block entry.
                If ``None``, do not change the level of the logger.
            handler: An additional handler to be used on block entry.
            close: Whether to close the handler on block exit.
        """  # noqa:D205 D212 D415
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

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.level is not None:
            self.logger.setLevel(self.old_level)
        if self.handler:
            self.logger.removeHandler(self.handler)
        if self.handler and self.close:
            self.handler.close()
        # implicit return of None => don't swallow exceptions


class OneLineLogging:
    """A context manager to make the StreamHandlers use only one line.

    Each record replaces the previous one.
    """

    __handler: StreamHandler | None
    """The handler used by the context manager if any."""

    __terminators: tuple[str]
    """One line terminator per stream handler."""

    __formatters: tuple[Formatter]
    """One formatter per stream handler."""

    __logger: Logger
    """The logger used by the context manager when the handler is not ``None``."""

    def __init__(self, logger: Logger) -> None:
        """
        Args:
            logger: The logger.
        """  # noqa: D205 D212 D415
        self.__handler = None
        self.__formatter = ()
        self.__terminator = ()
        while logger:
            self.__logger = logger
            stop = False
            for handler in logger.handlers:
                if isinstance(handler, StreamHandler) and handler.stream in {
                    stdout,
                    stderr,
                }:
                    self.__handler = handler
                    self.__terminator = handler.terminator
                    self.__formatter = handler.formatter
                    stop = True

            if stop:
                break

            if not logger.propagate:
                break

            logger = logger.parent

    def __enter__(self) -> None:
        if self.__handler is not None:
            self.__handler.terminator = ""
            self.__handler.formatter = Formatter(
                fmt="\r" + LOGGING_SETTINGS.message_format,
                datefmt=LOGGING_SETTINGS.date_format,
            )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.__handler is not None:
            self.__handler.terminator = self.__terminator
            self.__handler.formatter = self.__formatter
            self.__handler.stream.write("\n")
