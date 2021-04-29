# -*- coding: utf-8 -*-
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

from __future__ import absolute_import, division, unicode_literals

import unittest

from gemseo.third_party.genson_generator import Schema, SchemaError

from . import base


class TestType(base.SchemaTestCase):
    """"""

    def test_no_schema(self):
        """"""
        schema = {}
        s = Schema()
        s.add_schema(schema)
        self.assert_schema(s.to_dict(), schema)

    def test_single_type(self):
        """"""
        schema = {"type": "string"}
        s = Schema()
        s.add_schema(schema)
        self.assert_schema(s.to_dict(), schema)

    def test_single_type_unicode(self):
        """"""
        schema = {"type": "string"}
        s = Schema()
        s.add_schema(schema)
        self.assert_schema(s.to_dict(), schema)

    def test_multi_type(self):
        """"""
        schema = {"type": ["boolean", "null", "number", "string"]}
        s = Schema()
        s.add_schema(schema)
        self.assert_schema(s.to_dict(), schema)

    def test_incompatible(self):
        schema = {"a": ["boolean"]}
        s = Schema()
        s.add_schema(schema)
        schema = {"a": ["integer"]}
        self.assertRaises(SchemaError, s.add_schema, schema)

        err = SchemaError("a")
        assert len(repr(err)) > 10


class TestPreserveKeys(base.SchemaTestCase):
    """"""

    def test_preserves_existing_keys(self):
        """"""
        schema = {"type": "number", "value": 5}
        s = Schema()
        s.add_schema(schema)
        self.assert_schema(s.to_dict(), schema)


if __name__ == "__main__":
    unittest.main()
