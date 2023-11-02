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
from __future__ import annotations

from os.path import dirname

import pytest

from gemseo.utils.singleton import SingleInstancePerAttributeId
from gemseo.utils.singleton import SingleInstancePerFileAttribute


def test_sing_id():
    class SingleId(metaclass=SingleInstancePerAttributeId):
        def __init__(self, arg):
            super().__init__()
            self.arg = arg

    a = SingleId(0)
    b = SingleId(0)
    c = SingleId(a)
    assert a is b
    assert a is not c

    class SingleIdFail(metaclass=SingleInstancePerAttributeId):
        def __init__(self):
            super().__init__()

    with pytest.raises(ValueError):
        SingleIdFail()


def test_sing_file():
    file_loc = __file__

    class SingleFile(metaclass=SingleInstancePerFileAttribute):
        def __init__(self, arg):
            super().__init__()
            self.arg = arg

    a = SingleFile(file_loc)
    b = SingleFile(file_loc)
    c = SingleFile(dirname(file_loc))
    assert a is b
    assert a != c

    with pytest.raises(ValueError):
        SingleFile()

    class SingleFileFail(metaclass=SingleInstancePerFileAttribute):
        def __init__(self):
            super().__init__()

    with pytest.raises(ValueError):
        SingleFileFail()


def test_id_collision_inst():
    class SingleId1(metaclass=SingleInstancePerAttributeId):
        def __init__(self, arg):
            super().__init__()
            self.arg = arg

    class SingleId2(metaclass=SingleInstancePerAttributeId):
        def __init__(self, arg):
            super().__init__()
            self.arg = arg

    toto = ""
    s1 = SingleId1(toto)
    s2 = SingleId2(toto)

    assert not isinstance(s1, type(s2))


def test_id_collision_file():
    class SingleFId1(metaclass=SingleInstancePerFileAttribute):
        def __init__(self, arg):
            super().__init__()
            self.arg = arg

    class SingleFId2(metaclass=SingleInstancePerFileAttribute):
        def __init__(self, arg):
            super().__init__()
            self.arg = arg

    toto = ""
    s1 = SingleFId1(toto)
    s2 = SingleFId2(toto)

    assert not isinstance(s1, type(s2))
