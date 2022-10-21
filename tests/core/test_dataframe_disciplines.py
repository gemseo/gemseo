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
# Antoine DECHAUME
from __future__ import annotations

import numpy as np
import numpy.testing
import pytest
from gemseo.caches.cache_factory import CacheFactory
from gemseo.core.chain import MDOChain
from gemseo.core.chain import MDOParallelChain
from gemseo.core.discipline import MDODiscipline
from gemseo.core.discipline_data import DisciplineData
from gemseo.core.mdofunctions.function_generator import MDOFunctionGenerator
from gemseo.mda.gauss_seidel import MDAGaussSeidel
from gemseo.mda.jacobi import MDAJacobi
from gemseo.mda.newton import MDAQuasiNewton
from gemseo.utils.testing import compare_dict_of_arrays
from numpy import array
from numpy import ndarray
from pandas import DataFrame

from .test_discipline_data import to_df_key


class DFChooser(MDODiscipline):
    """Base class allowing to choose whether to use a data frame and share it for both
    the inputs and the outputs."""

    def __init__(
        self,
        with_df: bool,
        grammar_type: str,
        df_shares_io: bool,
    ):
        super().__init__(grammar_type=grammar_type)
        self.with_df = with_df
        if df_shares_io:
            self.output_name = to_df_key("x", "b")
        else:
            self.output_name = to_df_key("y", "b")


class A(DFChooser):
    """Discipline with 1 input and 1 output.

    It may use a DataFrame that shares the inputs and outputs. It computes a 1D affine
    function.
    """

    def __init__(
        self,
        with_df: bool,
        grammar_type: str,
        df_shares_io: bool = False,
    ):
        super().__init__(with_df, grammar_type, df_shares_io)

        if self.with_df:
            self.default_inputs = {"x": DataFrame(data={"a": array([0.0])})}
            if self.grammar_type == MDODiscipline.JSON_GRAMMAR_TYPE:
                self.input_grammar.update_from_data({to_df_key("x", "a"): array([0.0])})
                self.output_grammar.update_from_data({self.output_name: array([0.0])})
            else:
                self.input_grammar.update({to_df_key("x", "a"): ndarray})
                self.output_grammar.update({self.output_name: ndarray})
        else:
            self.default_inputs = {"x": array([0.0])}
            if self.grammar_type == MDODiscipline.JSON_GRAMMAR_TYPE:
                self.input_grammar.update_from_data({"x": array([0.0])})
                self.output_grammar.update_from_data({"y": array([0.0])})
            else:
                self.input_grammar.update({"x": ndarray})
                self.output_grammar.update({"y": ndarray})

    def _run(self):
        d = self.local_data
        if self.with_df:
            d[self.output_name] = 1 - 0.2 * d[to_df_key("x", "a")]
        else:
            d["y"] = 1 - 0.2 * d["x"]


class B(DFChooser):
    """Discipline with 1 inputs and 1 outputs.

    It may use a DataFrame that shares the inputs and outputs. It computes a 1D affine
    function.
    """

    def __init__(
        self,
        with_df: bool,
        grammar_type: str,
        df_shares_io: bool = False,
    ):
        super().__init__(with_df, grammar_type, df_shares_io)
        if self.with_df:
            if df_shares_io:
                df_name = "x"
            else:
                df_name = "y"

            self.default_inputs = {df_name: DataFrame(data={"b": array([0.0])})}
            if self.grammar_type == MDODiscipline.JSON_GRAMMAR_TYPE:
                self.input_grammar.update_from_data({self.output_name: array([0.0])})
                self.output_grammar.update_from_data(
                    {to_df_key("x", "a"): array([0.0])}
                )
            else:
                self.input_grammar.update({self.output_name: ndarray})
                self.output_grammar.update({to_df_key("x", "a"): ndarray})
        else:
            self.default_inputs = {"y": array([0.0])}
            if self.grammar_type == MDODiscipline.JSON_GRAMMAR_TYPE:
                self.input_grammar.update_from_data({"y": array([0.0])})
                self.output_grammar.update_from_data({"x": array([0.0])})
            else:
                self.input_grammar.update({"y": ndarray})
                self.output_grammar.update({"x": ndarray})

    def _run(self):
        d = self.local_data
        if self.with_df:
            d[to_df_key("x", "a")] = 1 - 0.3 * d[self.output_name]
        else:
            d["x"] = 1 - 0.3 * d["y"]


def get_executed_disc(
    disc_class: type,
    with_df: bool,
    grammar_type: str,
    df_shares_io: bool = False,
) -> MDODiscipline:
    """Create, execute and return a discipline.

    Args:
        disc_class: The class of the discipline.
        with_df: Whether to use data frames.
        grammar_type: The type of grammar.
        df_shares_io: Whether the same dataframe is used for both inputs and outputs.

    Returns:
        The discipline.
    """
    disc = disc_class(
        [
            A(with_df, grammar_type, df_shares_io),
            B(with_df, grammar_type, df_shares_io),
        ],
        grammar_type=grammar_type,
    )
    disc.execute()
    return disc


@pytest.mark.parametrize("df_shares_io", [False, True])
@pytest.mark.parametrize(
    "grammar_type", [MDODiscipline.SIMPLE_GRAMMAR_TYPE, MDODiscipline.JSON_GRAMMAR_TYPE]
)
@pytest.mark.parametrize(
    "disc_class",
    [MDAGaussSeidel, MDAJacobi, MDAQuasiNewton, MDOChain, MDOParallelChain],
)
def test_disciplines_comparison(grammar_type, disc_class, df_shares_io):
    """Compare results of data frames against NumPy arrays with disciplines."""
    with_df = False
    disc = get_executed_disc(disc_class, with_df, grammar_type)

    with_df = True
    disc_with_df = get_executed_disc(disc_class, with_df, grammar_type, df_shares_io)

    if df_shares_io:
        output_name = to_df_key("x", "b")
    else:
        output_name = to_df_key("y", "b")

    assert len(disc_with_df.local_data) == len(disc.local_data)
    assert disc_with_df.local_data[to_df_key("x", "a")] == disc.local_data["x"]
    assert disc_with_df.local_data[output_name] == disc.local_data["y"]


def test_mdo_function_comparison():
    """Compare results of data frames against NumPy arrays with MDOFunctions."""
    grammar_type = MDODiscipline.SIMPLE_GRAMMAR_TYPE

    with_df = False
    fct_gen = MDOFunctionGenerator(A(with_df, grammar_type))
    fct = fct_gen.get_function(["x"], ["y"])

    with_df = True
    fct_gen = MDOFunctionGenerator(A(with_df, grammar_type))
    fct_with_df = fct_gen.get_function([to_df_key("x", "a")], [to_df_key("y", "b")])

    x = np.array([1.0])
    assert fct(x) == fct_with_df(x)


class A2(A):
    """Discipline with 2 inputs and 2 outputs."""

    def __init__(self, with_df):
        super().__init__(with_df, MDODiscipline.SIMPLE_GRAMMAR_TYPE)
        if self.with_df:
            self.default_inputs["x"]["c"] = array([0.0])
            self.input_grammar.update({to_df_key("x", "c"): ndarray})
            self.output_grammar.update({to_df_key("y", "d"): ndarray})
        else:
            self.default_inputs["c"] = array([0.0])
            self.input_grammar.update({"c": ndarray})
            self.output_grammar.update({"d": ndarray})

    def _run(self):
        super()._run()
        d = self.local_data
        if self.with_df:
            d[to_df_key("y", "d")] = d[to_df_key("x", "c")] ** 2
        else:
            d["d"] = d["c"] ** 2


def test_mdo_function_array_dispatch():
    """Compare results of data frames against NumPy arrays with MDOFunction array
    dispatch."""
    with_df = False
    fct_gen = MDOFunctionGenerator(A2(with_df))
    fct = fct_gen.get_function(["x", "c"], ["y", "d"])

    with_df = True
    fct_gen = MDOFunctionGenerator(A2(with_df))
    fct_with_df = fct_gen.get_function(
        [to_df_key("x", "a"), to_df_key("x", "c")],
        [to_df_key("y", "b"), to_df_key("y", "d")],
    )

    x = np.array([1.0, 2.0])
    np.testing.assert_array_equal(fct(x), fct_with_df(x))


def test_discipline_outputs():
    """Compare discipline outputs of data frames against NumPy arrays."""
    res = A2(False).execute({"x": np.array([1.0]), "c": np.array([2.0])})
    res_with_df = A2(True).execute(
        {to_df_key("x", "a"): np.array([1.0]), to_df_key("x", "c"): np.array([2.0])}
    )

    assert res_with_df[to_df_key("x", "a")] == res["x"]
    assert res_with_df[to_df_key("x", "c")] == res["c"]
    assert res_with_df[to_df_key("y", "b")] == res["y"]
    assert res_with_df[to_df_key("y", "d")] == res["d"]


def assert_disc_data_equal(dd1, dd2):
    assert len(dd1) == len(dd2)
    for key, value in dd1.items():
        assert key in dd2
        numpy.testing.assert_array_equal(value, dd2[key])


@pytest.mark.parametrize(
    "cache_name,cache_options",
    [
        ("SimpleCache", dict()),
        ("MemoryFullCache", dict()),
        ("MemoryFullCache", dict(is_memory_shared=False)),
        ("HDF5Cache", dict(hdf_file_path="dummy.h5", hdf_node_path="DummyCache")),
    ],
)
def test_cache(cache_name, cache_options, tmp_wd):
    """Verify cache usages."""
    cache = CacheFactory().create(cache_name, **cache_options)

    input_data = DisciplineData({"i": np.arange(3)})
    data_out = DisciplineData({"o": np.arange(4)})

    cache[input_data] = (data_out, {})

    _, out, jac = cache[input_data]

    compare_dict_of_arrays(out, data_out)

    assert not jac


def test_serialization(tmp_wd):
    """Verify serialization."""
    with_df = True
    disc = A(with_df, MDODiscipline.JSON_GRAMMAR_TYPE)
    disc.execute()
    pickle_file_name = "a.pickle"
    disc.serialize(pickle_file_name)

    new_disc = MDODiscipline.deserialize(pickle_file_name)

    compare_dict_of_arrays(disc.local_data, new_disc.local_data)
