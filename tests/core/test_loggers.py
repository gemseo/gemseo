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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Damien Guenot
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import sys
import unittest

from future import standard_library

from gemseo import SOFTWARE_NAME
from gemseo.api import configure_logger
from gemseo.core.logger_config import LoggerConfig

standard_library.install_aliases()


LOGGER = configure_logger(SOFTWARE_NAME)


class Test_Logger(unittest.TestCase):
    """ """

    def test_init(self):
        """ """
        logger = logging.getLogger("TOTO")
        conf1 = LoggerConfig(logger)
        self.assertIsNone(conf1.file_handler)
        self.assertIsNone(conf1.formatter)

        conf2 = LoggerConfig(logger)
        assert conf1.stdout_hdlr == conf2.stdout_hdlr
        assert conf1.stderr_hdlr == conf2.stderr_hdlr
        stdout_hdlr = logging.StreamHandler(sys.stdout)

        logger = logging.getLogger("TOTO2")
        logger.addHandler(stdout_hdlr)
        LoggerConfig(logger)

        logger = logging.getLogger("TOTO3")
        logger.addHandler(logging.StreamHandler(sys.stderr))
        LoggerConfig(logger)

    def test_activate_debug(self):
        """ """
        log = LoggerConfig(LOGGER)
        log.activate_debug()
        self.assertEqual(LOGGER.level, logging.DEBUG)

    def test_deactivate_debug(self):
        """ """
        log = LoggerConfig(LOGGER)
        log.activate_debug()
        log.deactivate_debug()
        self.assertEqual(LOGGER.level, logging.INFO)

    def test_hide_non_warnings(self):
        """ """
        log = LoggerConfig(LOGGER)
        log.hide_non_warnings()
        self.assertEqual(LOGGER.level, logging.WARNING)

    def test_show_info(self):
        """ """
        log = LoggerConfig(LOGGER)
        log.activate_debug()
        log.show_info()
        self.assertEqual(LOGGER.level, logging.INFO)

    def test_set_logger_config(self):
        """ """
        log = LoggerConfig(LOGGER)
        log.activate_debug()
        log.set_logger_config(filename="test.log")
        date_format = "%H:%M:%S"
        message_format = "%(levelname)8s - %(asctime)s : %(message)s"
        self.assertEqual(
            type(logging.Formatter(message_format, date_format)), type(log.formatter)
        )
        log.set_level(logging.DEBUG)
        log.set_level(logging.INFO)
        log.deactivate_file_logging()
