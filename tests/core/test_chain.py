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

import os
import pickle
import re
import unittest
from itertools import permutations
from typing import TYPE_CHECKING

import pytest
from numpy import allclose
from numpy import array
from numpy import ones

from gemseo.core._process_flow.execution_sequences.parallel import ParallelExecSequence
from gemseo.core.chains.additive_chain import MDOAdditiveChain
from gemseo.core.chains.chain import MDOChain
from gemseo.core.chains.parallel_chain import MDOParallelChain
from gemseo.core.chains.warm_started_chain import MDOWarmStartedChain
from gemseo.core.discipline import Discipline
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.disciplines.splitter import Splitter
from gemseo.problems.mdo.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.mdo.sobieski.disciplines import SobieskiMission
from gemseo.problems.mdo.sobieski.disciplines import SobieskiPropulsion
from gemseo.problems.mdo.sobieski.disciplines import SobieskiStructure
from gemseo.problems.mdo.sobieski.process.mdo_chain import SobieskiChain
from gemseo.utils.discipline import DummyDiscipline

if TYPE_CHECKING:
    from gemseo.typing import StrKeyMapping

DIRNAME = os.path.dirname(__file__)


class Testmdochain(unittest.TestCase):
    """"""

    def get_disciplines_list(self, perm, dtype="complex128"):
        disciplines = [
            SobieskiStructure(dtype),
            SobieskiAerodynamics(dtype),
            SobieskiPropulsion(dtype),
            SobieskiMission(dtype),
        ]

        return [disciplines[p] for p in perm]

    def test_linearize_sobieski_chain_combinatorial(self) -> None:
        """"""
        for perm in permutations(range(4)):
            disciplines = self.get_disciplines_list(perm)
            chain = MDOChain(disciplines)
            ok = chain.check_jacobian(
                derr_approx="complex_step", step=1e-30, threshold=1e-6
            )
            assert ok

    def test_add_differentiated_inputs(self) -> None:
        disciplines = [
            SobieskiStructure(),
            SobieskiAerodynamics(),
            SobieskiPropulsion(),
            SobieskiMission(),
        ]
        chain = MDOChain(disciplines)
        chain.add_differentiated_inputs()
        chain.add_differentiated_outputs()

    def test_parallel_chain_combinatorial_thread(self) -> None:
        for perm in permutations(range(4)):
            for use_deep_copy in [True, False]:
                disciplines = self.get_disciplines_list(perm)
                chain = MDOParallelChain(
                    disciplines, use_threading=True, use_deep_copy=use_deep_copy
                )
                chain.linearize(compute_all_jacobians=True)
                ok = chain.check_jacobian(
                    chain.io.input_grammar.defaults,
                    derr_approx="complex_step",
                    step=1e-30,
                    threshold=1e-6,
                )
                assert ok

    @pytest.mark.skip_under_windows
    def test_parallel_chain_combinatorial_mprocess(self) -> None:
        # Keep the two first only as MP is slow there
        perms = list(permutations(range(4)))[:2]
        for perm in perms:
            disciplines = self.get_disciplines_list(perm)
            chain = MDOParallelChain(disciplines, use_threading=False)
            ok = chain.check_jacobian(
                chain.io.input_grammar.defaults,
                derr_approx="complex_step",
                step=1e-30,
                threshold=1e-6,
            )
            assert ok

    def test_workflow_dataflow(self) -> None:
        disciplines = self.get_disciplines_list(range(4))
        chain = MDOParallelChain(disciplines)
        assert isinstance(
            chain.get_process_flow().get_execution_flow(), ParallelExecSequence
        )
        assert chain.get_process_flow().get_data_flow() == []

    def test_common_in_out(self) -> None:
        # Check that the linearization works with a discipline
        # that has inputs and outputs of the same name
        a = AnalyticDiscipline({"x": "x"}, name="a")
        o = AnalyticDiscipline({"o": "x+y"}, name="o")
        chain = MDOChain([a, o])
        assert chain.check_jacobian(
            {"x": ones(1), "y": ones(1)},
            step=1e-6,
            threshold=1e-5,
            derr_approx="finite_differences",
        )

    def test_double_mission_chain(self) -> None:
        # Create a chain that adds two missions
        disciplines = [SobieskiMission(), SobieskiMission()]
        outputs_to_sum = ["y_4"]
        chain = MDOAdditiveChain(disciplines, outputs_to_sum)

        # Check the output value
        chain.execute()
        mission = SobieskiMission()
        mission.execute()
        assert allclose(chain.io.data["y_4"], mission.io.data["y_4"] * 2.0)

        # Check the output Jacobian
        chain.check_jacobian(threshold=1e-5)


def test_mdo_chain_serialization(tmp_wd) -> None:
    """Test that an MDOChain can be serialized, loaded and executed.

    The focus of this test is to guarantee that the loaded MDOChain instance can be
    executed, if an AttributeError is raised, it means that the attribute is missing in
    MDOChain._ATTR_TO_SERIALIZE.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    chain = SobieskiChain()
    with open("chain.pkl", "wb") as file:
        pickle.dump(chain, file)

    with open("chain.pkl", "rb") as file:
        chain = pickle.load(file)
    chain.check_jacobian(threshold=1e-5)
    chain.execute()


@pytest.mark.parametrize(
    ("variable_names", "expected"), [(["y_21"], True), ([], False)]
)
def test_warm_started_mdo_chain(variable_names, expected) -> None:
    """Test that the variables are warm-started properly."""
    disciplines = [
        SobieskiStructure(),
        SobieskiAerodynamics(),
    ]
    chain = MDOWarmStartedChain(
        disciplines=disciplines, variable_names_to_warm_start=variable_names
    )
    out = chain.execute()
    y_12 = out["y_12"]
    chain.cache.clear()
    out = chain.execute()
    assert (y_12 != out["y_12"]).any() == expected


def test_warm_started_mdo_chain_jac() -> None:
    """Test that the Jacobian of an MDOWarmStartedChain raises an exception."""
    chain = MDOWarmStartedChain([SobieskiMission()], variable_names_to_warm_start=[])
    with pytest.raises(
        NotImplementedError,
        match=re.escape("MDOWarmStartedChain cannot be linearized."),
    ):
        chain.check_jacobian()


@pytest.mark.parametrize("variable_names", [("y_4", "i_dont_exist"), ("i_dont_exist",)])
def test_warm_started_mdo_chain_variables(variable_names) -> None:
    """Test an exception if a variable that is not in the chain is warm started."""
    with pytest.raises(
        ValueError,
        match="The following variable names are not "
        r"outputs of the chain: \{'i_dont_exist'\}\."
        r" Available outputs are: \['y_4'\]\.",
    ):
        MDOWarmStartedChain(
            [SobieskiMission()], variable_names_to_warm_start=variable_names
        )


@pytest.fixture
def two_virtual_disciplines() -> list[Discipline]:
    """Create two dummy disciplines that have no _run method and can only be executed in
    virtual mode.

    Returns:
        The two disciplines.
    """
    disc_1 = DummyDiscipline("d1")
    disc_1.io.input_grammar.update_from_names(["x"])
    disc_1.io.output_grammar.update_from_names(["y"])
    disc_1.io.input_grammar.defaults = {"x": array([1.0])}
    disc_1.default_output_data = {"y": array([2.0])}
    disc_1.virtual_execution = True

    disc_2 = DummyDiscipline("d2")
    disc_2.io.input_grammar.update_from_names(["y"])
    disc_2.io.output_grammar.update_from_names(["z"])
    disc_2.io.input_grammar.defaults = {"y": array([3.0])}
    disc_2.default_output_data = {"z": array([4.0])}
    disc_2.virtual_execution = True

    return [disc_1, disc_2]


def test_virtual_exe_chain(two_virtual_disciplines) -> None:
    """Test a chain with disciplines in virtual execution mode."""
    chain = MDOChain(two_virtual_disciplines)
    chain.execute()
    assert chain.io.data["z"] == 4.0
    assert chain.io.data["y"] == 2.0


def test_jacobian_of_chain_including_splitter() -> None:
    """Test the jacobian of an MDOChain including a splitter."""
    splitter_disc = Splitter(
        input_name="x", output_names_to_input_indices={"x_1": [0], "x_2": [1]}
    )
    analytic_disc = AnalyticDiscipline({"y": "x_1+x_2"})
    chain = MDOChain([splitter_disc, analytic_disc])
    assert chain.check_jacobian(input_data={"x": array([0.0, 0.0])})


def test_non_ndarray_inputs():
    """Check that MDOParallelChain handles inputs that are not NumPy arrays."""

    class StringDuplicator(Discipline):
        """A discipline duplicating an input string, e.g. "foo" -> "foofoo"."""

        def __init__(self):  # noqa: D107
            super().__init__()
            self.io.input_grammar.update_from_types({"in": str})
            self.io.output_grammar.update_from_types({"out": str})
            self.io.input_grammar.defaults["in"] = "foo"

        def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
            self.io.data["out"] = self.io.data["in"] * 2

    mdo_parallel_chain = MDOParallelChain([StringDuplicator()])
    mdo_parallel_chain.execute()
    assert mdo_parallel_chain.io.data["out"] == "foofoo"
    mdo_parallel_chain.execute({"in": "bar"})
    assert mdo_parallel_chain.io.data["out"] == "barbar"
