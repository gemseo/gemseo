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

# The MIT License (MIT)
#
# Copyright (c) 2014 Jon Wolverton github.com/wolverdude
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import unittest

from future import standard_library

from gemseo.api import configure_logger
from gemseo.third_party.genson_generator import Schema

if sys.version_info.major == 2:
    from gemseo.third_party.fastjsonschema import compile
    from gemseo.third_party.fastjsonschema.exceptions import JsonSchemaException
else:
    from fastjsonschema import compile
    from fastjsonschema.exceptions import JsonSchemaException
from gemseo import SOFTWARE_NAME

standard_library.install_aliases()
configure_logger(SOFTWARE_NAME)


class SchemaTestCase(unittest.TestCase):
    """ """

    def assertGenSchema(self, instance, options, expected):
        """

        :param instance: param options:
        :param expected:
        :param options:

        """
        actual = Schema(**options).add_object(instance).to_dict()
        self.assertSchema(actual, expected)
        self.assertObjectValid(instance, actual)
        return actual

    def assertSchema(self, actual, expected):
        """

        :param actual: param expected:
        :param expected:

        """
        self.assertValidSchema(actual)
        self.assertEqual(actual, expected)

    def assertValidSchema(self, schema):
        """

        :param schema:

        """
        compile(schema)

    def assertObjectValid(self, data, schema):
        """

        :param data: param schema:
        :param schema:

        """
        compile(schema)(data)

    def assertObjectInvalid(self, data, schema):
        """

        :param data: param schema:
        :param schema:

        """
        with self.assertRaises(JsonSchemaException):
            compile(schema)(data)
