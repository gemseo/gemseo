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
#        :author: Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
from builtins import super
from os.path import dirname

from future import standard_library
from future.utils import with_metaclass

from gemseo import SOFTWARE_NAME
from gemseo.api import configure_logger
from gemseo.utils.singleton import (
    SingleInstancePerAttributeEq,
    SingleInstancePerAttributeId,
    SingleInstancePerFileAttribute,
)

standard_library.install_aliases()


configure_logger(SOFTWARE_NAME)


class TestSingleton(unittest.TestCase):
    """Test the signletons"""

    def test_sing_attr(self):
        class SingleEq(with_metaclass(SingleInstancePerAttributeEq, object)):
            def __init__(self, arg):
                super(SingleEq, self).__init__()
                self.arg = arg

        a = SingleEq(1)
        b = SingleEq(1)
        c = SingleEq(2)
        assert a == b
        assert a != c

        class SingleEq2(with_metaclass(SingleInstancePerAttributeEq, object)):
            def __init__(self, arg=None):
                super(SingleEq2, self).__init__()

        d = SingleEq2()
        e = SingleEq2(None)
        assert d == e

    def test_sing_id(self):
        class SingleId(with_metaclass(SingleInstancePerAttributeId, object)):
            def __init__(self, arg):
                super(SingleId, self).__init__()
                self.arg = arg

        a = SingleId(self)
        b = SingleId(self)
        c = SingleId(a)
        assert a == b
        assert a != c

        class SingleIdFail(with_metaclass(SingleInstancePerAttributeId, object)):
            def __init__(self):
                super(SingleIdFail, self).__init__()

        self.assertRaises(ValueError, SingleIdFail)

    def test_sing_File(self):

        file_loc = __file__

        class SingleFile(with_metaclass(SingleInstancePerFileAttribute, object)):
            def __init__(self, arg):
                super(SingleFile, self).__init__()
                self.arg = arg

        a = SingleFile(file_loc)
        b = SingleFile(file_loc)
        c = SingleFile(dirname(file_loc))
        assert a == b
        assert a != c
        self.assertRaises(TypeError, SingleFile, self)

        class SingleFileFail(with_metaclass(SingleInstancePerFileAttribute, object)):
            def __init__(self):
                super(SingleFileFail, self).__init__()

        self.assertRaises(ValueError, SingleFileFail)

    def test_id_collision_inst(self):
        class SingleId1(with_metaclass(SingleInstancePerAttributeId, object)):
            def __init__(self, arg):
                super(SingleId1, self).__init__()
                self.arg = arg

        class SingleId2(with_metaclass(SingleInstancePerAttributeId, object)):
            def __init__(self, arg):
                super(SingleId2, self).__init__()
                self.arg = arg

        toto = type("TOTO")()
        s1 = SingleId1(toto)
        s2 = SingleId2(toto)

        assert not isinstance(s1, type(s2))

    def test_id_collision_file(self):
        class SingleFId1(with_metaclass(SingleInstancePerFileAttribute, object)):
            def __init__(self, arg):
                super(SingleFId1, self).__init__()
                self.arg = arg

        class SingleFId2(with_metaclass(SingleInstancePerFileAttribute, object)):
            def __init__(self, arg):
                super(SingleFId2, self).__init__()
                self.arg = arg

        toto = type("TOTO2")()
        s1 = SingleFId1(toto)
        s2 = SingleFId2(toto)

        assert not isinstance(s1, type(s2))
