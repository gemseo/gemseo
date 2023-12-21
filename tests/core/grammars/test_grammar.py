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
from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from gemseo.core.grammars.errors import InvalidDataError
from gemseo.core.grammars.factory import GrammarFactory

FACTORY = GrammarFactory()


@pytest.mark.parametrize("class_name", tuple(FACTORY.class_names))
def test_serialize(tmp_wd, class_name):
    """Check that a grammar can be properly serialized."""
    # TODO: implement the serialization of PydanticGrammar
    if class_name == "PydanticGrammar":
        return

    original_g = FACTORY.create(class_name, "g")
    original_g.update_from_types({"x": int, "y": bool})
    original_g.add_namespace("x", "n")
    original_g.required_names.remove("y")

    with Path("foo.pkl").open("wb") as outfobj:
        pickler = pickle.Pickler(outfobj, protocol=2)
        pickler.dump(original_g)

    with Path("foo.pkl").open("rb") as outfobj:
        pickler = pickle.Unpickler(outfobj)
        g = pickler.load()

    assert g.name == original_g.name
    assert g.required_names == original_g.required_names
    assert g.to_namespaced == original_g.to_namespaced
    assert g.from_namespaced == original_g.from_namespaced

    g.validate({"n:x": 1, "y": False})

    if class_name != "SimplerGrammar":
        for data in [{"y": False}, {"n:x": 1.5}, {"n:x": 1, "y": 3}]:
            with pytest.raises(InvalidDataError):
                g.validate(data)
