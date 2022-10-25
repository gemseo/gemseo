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
#                         documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import unittest

from gemseo.core.data_processor import ComplexDataProcessor
from gemseo.core.data_processor import FloatDataProcessor
from gemseo.core.data_processor import NameMapping
from gemseo.core.discipline import MDODiscipline
from gemseo.problems.sobieski.disciplines import SobieskiMission
from numpy import array
from numpy import complex128
from numpy import float64
from numpy import ndarray
from scipy import linalg


class TestDataProcessor(unittest.TestCase):
    """"""

    def test_float_data_processor(self):
        """"""
        dp = FloatDataProcessor()
        in_data = {"a": array([1.1]), "b": array([3.1, 4.1])}
        pre_data = dp.pre_process_data(in_data)
        assert len(pre_data) == len(in_data)
        for k, v in pre_data.items():
            assert k in in_data
            if k == "a":
                assert isinstance(v, float)
            else:
                assert isinstance(v, list)

        post_data = dp.post_process_data(pre_data)
        assert len(post_data) == len(in_data)
        for k, v in post_data.items():
            assert k in in_data
            assert isinstance(v, ndarray)

    def test_complex_data_processor(self):
        """"""
        dp = ComplexDataProcessor()
        in_data = {"a": array([1.1 + 2j]), "b": array([3.1, 4.1 + 3j])}
        pre_data = dp.pre_process_data(in_data)
        assert len(pre_data) == len(in_data)
        for k, v in pre_data.items():
            assert k in in_data
            assert linalg.norm(v - in_data[k].real) == 0.0
            assert linalg.norm(v.imag) == 0
            assert v.dtype == float64

        post_data = dp.post_process_data(pre_data)
        assert len(post_data) == len(in_data)
        for k, v in post_data.items():
            assert k in in_data
            assert isinstance(v, ndarray)
            assert v.dtype == complex128

        sm = SobieskiMission("float64")
        sm.data_processor = dp
        sm.execute(
            {"x_shared": array(sm.default_inputs["x_shared"], dtype="complex128")}
        )

        assert sm.local_data["y_4"].dtype == complex128

    def test_name_mapping(self):
        disc = LocalDisc()
        disc.data_processor = NameMapping({"A": "a", "B": "b", "O": "o"})
        out = disc.execute({"A": array([1]), "B": array([2])})
        assert out["O"] == array([3.0])


class LocalDisc(MDODiscipline):
    def __init__(self):
        super().__init__()
        self.input_grammar.update(["A", "B"])
        self.output_grammar.update(["O"])

    def _run(self):
        self.local_data["o"] = self.local_data["a"] + self.local_data["b"]
