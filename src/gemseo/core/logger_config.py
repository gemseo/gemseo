# -*- coding: utf-8 -*-
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
#                        documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Python logger configuration made easier
***************************************
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import sys

from future import standard_library
from future.utils import with_metaclass

from gemseo.utils.singleton import SingleInstancePerAttributeId

standard_library.install_aliases()


class LoggerConfig(with_metaclass(SingleInstancePerAttributeId, object)):
    """Class to easily set the self.logger configuration."""

    def __init__(self, logger):
        """
        Initialize handlers
        """
        self.logger = logger
        logger_streams = [
            hdlr.stream for hdlr in logger.handlers if hasattr(hdlr, "stream")
        ]
        if sys.stdout not in logger_streams:
            self.stdout_hdlr = logging.StreamHandler(sys.stdout)
            self.logger.addHandler(self.stdout_hdlr)
        else:
            hdlr_indx = logger_streams.index(sys.stdout)
            self.stdout_hdlr = logger.handlers[hdlr_indx]

        if sys.stderr not in logger_streams:
            self.stderr_hdlr = logging.StreamHandler(sys.stderr)
            self.logger.addHandler(self.stderr_hdlr)
        else:
            hdlr_indx = logger_streams.index(sys.stderr)
            self.stderr_hdlr = logger.handlers[hdlr_indx]
        self.file_handler = None
        self.formatter = None
        self.logger.propagate = False

    def activate_debug(self):
        """Activate the debug prints"""
        self.logger.setLevel(logging.DEBUG)

    def deactivate_debug(self):
        """De Activate the debug prints, show info or more critical"""
        self.logger.setLevel(logging.INFO)

    def hide_non_warnings(self):
        """Activate the warning and critical prints"""
        self.logger.setLevel(logging.WARNING)

    def show_info(self):
        """Shows info or more critical"""
        self.logger.setLevel(logging.INFO)

    def set_level(self, level):
        """Set logging level

        :param level: logger level (DEBUG, INFO...)
        """
        self.logger.setLevel(level)

    def add_logging_file(self, filename="mdo_scenario.log", mode="a", delay=True):
        """Adds a logging file

        :param filename: the output file (Default value = "mdo_scenario.log")
        :param mode: write mode of the file (Default value = "a")
        :param delay: if True , waits for the first emit to write the log file
            (Default value = True)
        """
        self.file_handler = logging.FileHandler(
            filename, mode=mode, delay=delay, encoding="utf-8"
        )
        self.logger.addHandler(self.file_handler)
        self.file_handler.setFormatter(self.formatter)

    def deactivate_file_logging(self):
        """Deactivate file logging"""
        self.logger.removeHandler(self.file_handler)

    def set_logger_config(
        self,
        level=None,
        date_format=None,
        message_format=None,
        filename=None,
        filemode="a",
    ):
        """Sets the self.logger configuration

        :param level: self.logger print level, default INFO, can be :
            self.logger.DEBUG, self.logger.INFO,
            self.logger.WARNING, logging.CRITICAL
        :param date_format: date format, if None, use a default one
        :param message_format: message format, if None, use a default one
        :param filename: the file path if outputs must be written in a file
            (Default value = None)
        :param filemode: Default value = 'a')
        """
        if level is None:
            level = logging.INFO
        if date_format is None:
            date_format = "%d/%m/%Y %H:%M:%S"
            date_format = "%H:%M:%S"
        if message_format is None:
            message_format = "%(levelname)8s - %(asctime)s : %(message)s"
        self.formatter = logging.Formatter(message_format, date_format)
        if filename is not None:
            self.add_logging_file(filename, mode=filemode, delay=True)

        self.logger.setLevel(level)
        self.stdout_hdlr.addFilter(MaxLevelFilter(logging.INFO))
        self.stdout_hdlr.setLevel(level)
        self.stderr_hdlr.setLevel(logging.WARNING)

        self.stdout_hdlr.setFormatter(self.formatter)
        self.stderr_hdlr.setFormatter(self.formatter)

        # Capture warnings module outputs, used in matplotlib for instance
        logging.captureWarnings(True)
        self.logger.gemseo_config = self


class MaxLevelFilter(logging.Filter):
    """Filters (lets through) all messages with level <= LEVEL"""

    def __init__(self, level):
        """
        Constructor

        :param level: max message level
        """
        self.level = level
        super(MaxLevelFilter, self).__init__()

    def filter(self, record):
        """Filters a log message if the level of the message
        is lower than self.level

        :param record: the log record
        :returns: True if the message should be displayed

        """
        return record.levelno <= self.level
