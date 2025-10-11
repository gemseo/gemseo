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
"""Logging."""

from __future__ import annotations

import types
from dataclasses import dataclass
from logging import WARNING
from logging import FileHandler
from logging import Formatter
from logging import LogRecord
from logging import StreamHandler
from logging import getLogger
from logging import root
from pathlib import Path
from sys import stderr
from sys import stdout
from typing import TYPE_CHECKING
from typing import Final
from typing import TypeVar

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator

from gemseo.utils.constants import _LOGGING_DATE_FORMAT
from gemseo.utils.constants import _LOGGING_FILE_MODE
from gemseo.utils.constants import _LOGGING_FILE_PATH
from gemseo.utils.constants import _LOGGING_LEVEL
from gemseo.utils.constants import _LOGGING_MESSAGE_FORMAT

if TYPE_CHECKING:
    from logging import Logger
    from types import TracebackType

    from _typeshed import SupportsWrite
    from typing_extensions import Self

# TODO: API: remove this variable.
DEFAULT_DATE_FORMAT: Final[str] = _LOGGING_DATE_FORMAT
"""The format of the date of the logged message."""

# TODO: API: remove this variable.
DEFAULT_MESSAGE_FORMAT: Final[str] = _LOGGING_MESSAGE_FORMAT
"""The format of the logged message."""

# TODO: API: remove this variable.
GEMSEO_LOGGER: Final[Logger] = getLogger("gemseo")
"""The GEMSEO's logger."""

_StreamT = TypeVar("_StreamT", bound="SupportsWrite[str]")


class LoggingConfiguration(BaseModel, validate_assignment=True):
    """The configuration for |g| loggers."""

    date_format: str = Field(
        default=_LOGGING_DATE_FORMAT,
        description="The logging date format.",
    )

    enable: bool = Field(default=True, description="Whether to enable GEMSEO logging.")

    file_path: str | Path = Field(
        default=_LOGGING_FILE_PATH,
        description="The path to the log file, if outputs must be written in a file.",
    )

    file_mode: str = Field(
        default=_LOGGING_FILE_MODE,
        description="""The logging output file mode,
either 'w' (overwrite) or 'a' (append).""",
    )

    level: str | int = Field(
        default=_LOGGING_LEVEL,
        description="""The numerical value or name of the logging level,
as defined in :py:mod:`logging`.
Values can either be
``logging.NOTSET`` (``"NOTSET"``),
``logging.DEBUG`` (``"DEBUG"``),
``logging.INFO`` (``"INFO"``),
``logging.WARNING`` (``"WARNING"`` or ``"WARN"``),
``logging.ERROR`` (``"ERROR"``), or
``logging.CRITICAL`` (``"FATAL"`` or ``"CRITICAL"``).""",
    )

    message_format: str = Field(
        default=_LOGGING_MESSAGE_FORMAT, description="The logging message format."
    )

    @model_validator(mode="after")
    def __validate(self) -> Self:
        """Create the logger."""
        # Configure the loggers for GEMSEO and its plugins.
        # Do not configure the loggers for their modules.
        if self.enable:
            for name in root.manager.loggerDict:
                if _is_gemseo_logger(name):
                    _configure_logger(
                        name,
                        self.level,
                        self.message_format,
                        self.date_format,
                        self.file_path,
                        self.file_mode,
                    )
        else:
            for name in root.manager.loggerDict:
                if _is_gemseo_logger(name):
                    logger = getLogger(name)
                    for handler in logger.handlers[:]:
                        logger.removeHandler(handler)

        return self


def _is_gemseo_logger(name: str) -> bool:
    """Check whether a name is the name of the GEMSEO logger or one of its plugins.

    Args:
        name: The name.

    Returns:
        Whether the name is the name of the GEMSEO logger or one of its plugins.
    """
    return name.startswith("gemseo") and "." not in name


def _configure_logger(
    name: str,
    level: int | str,
    message_format: str,
    date_format: str,
    file_path: str | Path,
    file_mode: str,
    stream: _StreamT | None = None,
) -> Logger:
    """Configure a logger.

    Args:
        name: The name of the logger.
        level: The logging level.
        message_format: The logging message format.
        date_format: The logging date format.
        file_path: The path to the log file, if outputs must be written in a file.
        file_mode: The logging file mode.
        stream: The stream to use for logging, if any.

    Returns:
        The configured logger.
    """
    logger = getLogger(name)
    logger.setLevel(level)
    formatter = Formatter(fmt=message_format, datefmt=date_format)

    # Remove all existing handlers.
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    stream_handler = MultiLineStreamHandler(stream)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if file_path:
        file_handler = MultiLineFileHandler(
            file_path, mode=file_mode, delay=True, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# TODO: API: remove and use gemseo.configuration.logger instead.
@dataclass
class LoggingSettings:
    """The settings of a logger."""

    date_format: str = DEFAULT_DATE_FORMAT
    """The format of the date of the logged message."""

    message_format: str = DEFAULT_MESSAGE_FORMAT
    """The format of the logged message."""

    logger: Logger = GEMSEO_LOGGER
    """The logger."""


# TODO: API: remove and use gemseo.configuration.logger instead.
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
        logger: Logger | None = None,
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
        self.logger = getLogger("gemseo") if logger is None else logger
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
        from gemseo.utils.global_configuration import _configuration

        if self.__handler is not None:
            self.__handler.terminator = ""
            self.__handler.formatter = Formatter(
                fmt="\r" + _configuration.logging.message_format,
                datefmt=_configuration.logging.date_format,
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
