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
from itertools import permutations

import pytest
from numpy import ones

from gemseo.core.derivative.jacobian_operator import JacobianOperator
from gemseo.discipline.analytic import AnalyticDiscipline
from gemseo.discipline.chain.chain import ChainDerivationMode
from gemseo.discipline.chain.chain import DisciplineChain
from gemseo.problem.mdo.scalable.linear.disciplines_generator import (
    create_disciplines_from_desc,
)
from gemseo.problem.mdo.scalable.linear.linear_discipline import LinearDiscipline
from gemseo.util.derivative.approximation_mode import ApproximationMode
from gemseo.util.derivative.check.discipline import DisciplineJacobianChecker

_APPROXIMATION_MODE = ApproximationMode.COMPLEX_STEP
_APPROXIMATION_STEP = 1e-30
_CHECK_JACOBIAN_KWARGS = {"atol": 1e-6, "rtol": 1e-6}

# The two explicit sweeps. AUTO only dispatches to one of these based on the
# I/O sizes, so parametrizing correctness tests over it is redundant; AUTO's
# dispatch is covered end-to-end by test_jacobian_with_heterogeneous_sizes and
# directly by test_auto_linearization_mode_selection.
_SWEEP_MODES = [ChainDerivationMode.FORWARD, ChainDerivationMode.REVERSE]

_DESCRIPTION = [
    ("A", ["x1", "x2"], ["a1", "a2", "a3"]),
    ("B", ["x3", "x4"], ["b1", "b2"]),
    ("C", ["a1", "b1"], ["c1", "c2"]),
    ("D", ["a2", "b1"], ["d1"]),
    ("E", ["a3", "b2", "n"], ["e1", "e2"]),
    ("F", ["a1", "a2"], ["f1"]),
    ("G", ["c1", "d1"], ["g1", "g2"]),
    ("H", ["c2", "e1"], ["h1", "h2"]),
    ("I", ["d1", "f1"], ["i1"]),
    ("J", ["e2", "b2"], ["j1", "j2"]),
    ("K", ["g1", "h1", "m"], ["k1", "k2"]),
    ("L", ["g2", "i1", "x5"], ["l1"]),
    ("M", ["h2", "j1"], ["m", "y1"]),
    ("N", ["j2", "k2"], ["n", "y2"]),
    ("P", ["p", "k1", "x6"], ["p", "y_p"]),
    ("Q", ["l1", "m", "n", "y_p"], ["y3", "y4"]),
]

_IO_SUBSETS = [
    # Full inputs/outputs
    (["x1", "x2", "x3", "x4", "x5", "x6", "p"], ["y1", "y2", "y3", "y4", "y_p"]),
    # A-stream: x1 and x2 drive g/h/i paths to y3/y4
    (["x1", "x2"], ["y3", "y4"]),
    # B-stream: x3 and x4 drive c/d/e/j paths to y1/y2
    (["x3", "x4"], ["y1", "y2"]),
    # Skip connection: x5 feeds L directly → y3/y4
    (["x5"], ["y3", "y4"]),
    # Backward coupling: m/n enter K and E before M and N produce them
    (["m", "n"], ["y1", "y2", "y3", "y4"]),
    # Self-coupled terminal: p feeds back into P, x6 is a direct P input
    (["p", "x6"], ["y_p"]),
]


@pytest.fixture
def heterogeneous_sizes_disciplines() -> list[LinearDiscipline]:
    """3-discipline chain: A(1x3), B(3x2), C(2x1) Jacobian block shapes."""
    return [
        LinearDiscipline(
            "A",
            ["x1"],
            ["a"],
            inputs_size=1,
            outputs_size=3,
            matrix_format=LinearDiscipline.MatrixFormat.DENSE,
        ),
        LinearDiscipline(
            "B",
            ["x2", "a"],
            ["b"],
            inputs_size=3,
            outputs_size=2,
            matrix_format=LinearDiscipline.MatrixFormat.CSR,
        ),
        LinearDiscipline(
            "C",
            ["x3", "b"],
            ["y"],
            inputs_size=2,
            outputs_size=1,
        ),
    ]


@pytest.fixture
def heterogeneous_jacobian_type_disciplines() -> list[LinearDiscipline]:
    """3 all-to-all coupled disciplines with dense, CSR, and matrix-free Jacobians.

    Each discipline consumes the other two's outputs, so a forward coupling path
    exists in every ordering.
    """
    return [
        LinearDiscipline(
            "Dense",
            ["x", "b", "c"],
            ["a"],
            matrix_format=LinearDiscipline.MatrixFormat.DENSE,
        ),
        LinearDiscipline(
            "CSR",
            ["x", "a", "c"],
            ["b"],
            matrix_format=LinearDiscipline.MatrixFormat.CSR,
        ),
        LinearDiscipline(
            "MatrixFree",
            ["x", "a", "b"],
            ["c", "y"],
            matrix_free_jacobian=True,
        ),
    ]


@pytest.mark.parametrize("mode", _SWEEP_MODES)
@pytest.mark.parametrize(
    "order",
    [
        (0, 1, 2, 3),  # natural
        (3, 2, 1, 0),  # reversed
        (2, 0, 3, 1),  # interleaved
    ],
)
def test_linearization_varying_discipline_orders(
    sobieski_disciplines, order, mode
) -> None:
    """DisciplineChain Jacobian is correct regardless of discipline ordering."""
    chain = DisciplineChain([sobieski_disciplines[p] for p in order])
    checker = DisciplineJacobianChecker(chain)
    assert checker.check(
        linearization_mode=mode,
        approximation_mode=_APPROXIMATION_MODE,
        step=_APPROXIMATION_STEP,
        **_CHECK_JACOBIAN_KWARGS,
    )


@pytest.mark.parametrize("mode", _SWEEP_MODES)
@pytest.mark.parametrize("expression", ["x", "2*x"])
def test_jacobian_with_passthrough_discipline(expression, mode) -> None:
    """Jacobian is correct when a discipline re-emits an input under the same name.

    The identity case (x) is degenerate: ∂x/∂x = 1 masks a spurious extra
    application of the chain rule. The scaling case (2*x) catches it.
    """
    chain = DisciplineChain([
        AnalyticDiscipline({"x": expression}, name="a"),
        AnalyticDiscipline({"o": "x+y"}, name="o"),
    ])
    checker = DisciplineJacobianChecker(chain)
    assert checker.check(
        {"x": ones(1), "y": ones(1)},
        linearization_mode=mode,
        approximation_mode=_APPROXIMATION_MODE,
        step=_APPROXIMATION_STEP,
        **_CHECK_JACOBIAN_KWARGS,
    )


@pytest.mark.parametrize("mode", _SWEEP_MODES)
@pytest.mark.parametrize("permutation", list(permutations(range(3))))
def test_jacobian_with_heterogeneous_jacobian_formats(
    heterogeneous_jacobian_type_disciplines, permutation, mode
) -> None:
    """JacobianOperator contamination propagates to each discipline after MatrixFree."""
    chain = DisciplineChain([
        heterogeneous_jacobian_type_disciplines[i] for i in permutation
    ])
    checker = DisciplineJacobianChecker(chain)
    assert checker.check(
        linearization_mode=mode,
        approximation_mode=_APPROXIMATION_MODE,
        step=_APPROXIMATION_STEP,
        **_CHECK_JACOBIAN_KWARGS,
    )

    mf_idx = permutation.index(2)
    mf_before_dense = mf_idx < permutation.index(0)  # MatrixFree before Dense
    mf_before_csr = mf_idx < permutation.index(1)  # MatrixFree before CSR

    # MatrixFree's own outputs are always JacobianOperators.
    assert all(isinstance(v, JacobianOperator) for v in chain.jac["c"].values())
    assert all(isinstance(v, JacobianOperator) for v in chain.jac["y"].values())

    # Dense's output a is contaminated iff MatrixFree precedes Dense.
    # MatrixFree produces c, which Dense consumes — the c-path injects a
    # JacobianOperator into every (a, chain_input) entry via _add/_matmul.
    a_are_operators = [isinstance(v, JacobianOperator) for v in chain.jac["a"].values()]
    test = all(a_are_operators) if mf_before_dense else not any(a_are_operators)
    assert test

    # CSR's output b is contaminated iff MatrixFree precedes CSR — same mechanism.
    b_are_operators = [isinstance(v, JacobianOperator) for v in chain.jac["b"].values()]
    test = all(b_are_operators) if mf_before_csr else not any(b_are_operators)
    assert test


# Full ChainDerivationMode (incl. AUTO): asymmetric block sizes make AUTO
# dispatch to either sweep, covering the AUTO path through linearize.
@pytest.mark.parametrize("mode", ChainDerivationMode)
def test_jacobian_with_heterogeneous_sizes(
    heterogeneous_sizes_disciplines, mode
) -> None:
    """Jacobian is correct across non-square and rectangular coupling blocks."""
    chain = DisciplineChain(heterogeneous_sizes_disciplines)
    checker = DisciplineJacobianChecker(chain)
    assert checker.check(
        linearization_mode=mode,
        approximation_mode=_APPROXIMATION_MODE,
        step=_APPROXIMATION_STEP,
        **_CHECK_JACOBIAN_KWARGS,
    )


def test_discipline_chain_serialization(
    tmp_wd, heterogeneous_sizes_disciplines
) -> None:
    """DisciplineChain survives a pickle round-trip.

    Verifies that the deserialized chain can be executed and linearized.
    """
    chain = DisciplineChain(heterogeneous_sizes_disciplines)

    with open("chain.pkl", "wb") as file:
        pickle.dump(chain, file)

    with open("chain.pkl", "rb") as file:
        chain = pickle.load(file)

    chain.execute()
    assert "y" in chain.io.output_data

    checker = DisciplineJacobianChecker(chain)
    assert checker.check()


def test_discipline_chain_execution_with_virtual_disciplines(
    two_virtual_disciplines,
) -> None:
    """Virtual disciplines use default_output_data and propagate through the chain."""
    chain = DisciplineChain(two_virtual_disciplines)
    chain.execute()
    assert chain.io.output_data["z"] == 4.0
    assert chain.io.output_data["y"] == 2.0


@pytest.mark.parametrize("mode", _SWEEP_MODES)
def test_discipline_chain_skips_backward_coupling(mode) -> None:
    """Regression test for issue #1660.

    In chain A1(x,a2→a1), A2(a1→a2), B(a2→z), A2's output a2 feeds back into A1.
    For ∂z/∂x only ∂a1/∂x is needed; ∂a1/∂a2 must not appear in A1's Jacobian.
    """
    disciplines = create_disciplines_from_desc([
        ("A1", ["x", "a2"], ["a1"]),
        ("A2", ["a1"], ["a2"]),
        ("B", ["a2"], ["z"]),
    ])
    chain = DisciplineChain(disciplines)
    chain.linearization_mode = mode
    chain.add_differentiated_inputs(["x"])
    chain.add_differentiated_outputs(["z"])
    chain.linearize()

    assert set(disciplines[0].jac["a1"]) == {"x"}


@pytest.mark.parametrize("mode", _SWEEP_MODES)
def test_discipline_chain_coupling_variable_as_chain_input(mode) -> None:
    """Regression test for issue #1670.

    In chain A(x,b→a), B(a→b), C(b→z), b is both a chain input (A consumes it)
    and a coupling variable (B produces it for C). ∂z/∂b must propagate through
    B and A, not be taken directly from C's local ∂z/∂b.
    """
    disciplines = create_disciplines_from_desc([
        ("A", ["x", "b"], ["a"]),
        ("B", ["a"], ["b"]),
        ("C", ["b"], ["z"]),
    ])
    chain = DisciplineChain(disciplines)
    checker = DisciplineJacobianChecker(chain)
    assert checker.check(
        inputs=["b"],
        outputs=["z"],
        linearization_mode=mode,
        approximation_mode=_APPROXIMATION_MODE,
        step=_APPROXIMATION_STEP,
        **_CHECK_JACOBIAN_KWARGS,
    )


@pytest.mark.parametrize("mode", _SWEEP_MODES)
def test_discipline_chain_self_coupling_with_downstream_consumer(mode) -> None:
    """In chain P(p,x→p,w), R(p→z), p is self-coupled and consumed downstream.

    The row of w is seeded at P in terms of P's own input p and must not be
    expanded by ∂p/∂p again. Only the row of z, seeded downstream at R,
    references P's output p and requires the ∂z/∂p·∂p/∂(p,x) expansion.
    """
    disciplines = create_disciplines_from_desc([
        ("P", ["p", "x"], ["p", "w"]),
        ("R", ["p"], ["z"]),
    ])
    chain = DisciplineChain(disciplines)
    checker = DisciplineJacobianChecker(chain)
    assert checker.check(
        inputs=["p", "x"],
        outputs=["w", "z"],
        linearization_mode=mode,
        approximation_mode=_APPROXIMATION_MODE,
        step=_APPROXIMATION_STEP,
        **_CHECK_JACOBIAN_KWARGS,
    )


@pytest.mark.parametrize("mode", _SWEEP_MODES)
def test_discipline_chain_self_coupled_variable_as_output(mode) -> None:
    """A self-coupled variable differentiated as a chain output.

    In the single-discipline chain P(p,x→p,w) with p differentiated as both
    input and output, ∂p/∂p must be P's local ∂p_out/∂p_in, not its square.
    """
    disciplines = create_disciplines_from_desc([("P", ["p", "x"], ["p", "w"])])
    chain = DisciplineChain(disciplines)
    checker = DisciplineJacobianChecker(chain)
    assert checker.check(
        inputs=["p", "x"],
        outputs=["p", "w"],
        linearization_mode=mode,
        approximation_mode=_APPROXIMATION_MODE,
        step=_APPROXIMATION_STEP,
        **_CHECK_JACOBIAN_KWARGS,
    )


@pytest.mark.parametrize("mode", _SWEEP_MODES)
def test_discipline_chain_unreachable_input_output_pair(mode) -> None:
    """Zero Jacobian when no coupling path connects the input to the output.

    The active discipline set is empty, no sweep runs, and _init_jacobian
    fills the (y, x) block with sparse zeros.
    """
    disciplines = create_disciplines_from_desc([
        ("A", ["x"], ["a"]),
        ("B", ["u"], ["y"]),
    ])
    chain = DisciplineChain(disciplines)
    checker = DisciplineJacobianChecker(chain)
    assert checker.check(
        inputs=["x"],
        outputs=["y"],
        linearization_mode=mode,
        approximation_mode=_APPROXIMATION_MODE,
        step=_APPROXIMATION_STEP,
        **_CHECK_JACOBIAN_KWARGS,
    )
    assert chain.jac["y"]["x"].nnz == 0


@pytest.mark.parametrize("mode", _SWEEP_MODES)
def test_discipline_chain_blind_overwrite(mode) -> None:
    """Regression test for issue #1821.

    A discipline overwriting o without consuming it shadows upstream paths.
    C consumes B's o, which does not depend on x: ∂z/∂x must be zero, with no
    spurious contribution through A's o, while ∂z/∂u flows through B.
    """
    disciplines = create_disciplines_from_desc([
        ("A", ["x"], ["o"]),
        ("B", ["u"], ["o"]),
        ("C", ["o"], ["z"]),
    ])
    chain = DisciplineChain(disciplines)
    checker = DisciplineJacobianChecker(chain)
    assert checker.check(
        inputs=["x", "u"],
        outputs=["z"],
        linearization_mode=mode,
        approximation_mode=_APPROXIMATION_MODE,
        step=_APPROXIMATION_STEP,
        **_CHECK_JACOBIAN_KWARGS,
    )
    assert chain.jac["z"]["x"].nnz == 0


@pytest.mark.parametrize("mode", _SWEEP_MODES)
def test_discipline_chain_overwrite_with_intermediate_consumer(mode) -> None:
    """Regression test for issue #1821.

    Each consumer binds to the closest upstream producer of its inputs.
    C1 consumes A's o (before B overwrites it) and C2 consumes B's o:
    ∂z1/∂x ≠ 0 and ∂z2/∂u ≠ 0, while ∂z1/∂u = 0 and ∂z2/∂x = 0.
    """
    disciplines = create_disciplines_from_desc([
        ("A", ["x"], ["o"]),
        ("C1", ["o"], ["z1"]),
        ("B", ["u"], ["o"]),
        ("C2", ["o"], ["z2"]),
    ])
    chain = DisciplineChain(disciplines)
    checker = DisciplineJacobianChecker(chain)
    assert checker.check(
        inputs=["x", "u"],
        outputs=["z1", "z2"],
        linearization_mode=mode,
        approximation_mode=_APPROXIMATION_MODE,
        step=_APPROXIMATION_STEP,
        **_CHECK_JACOBIAN_KWARGS,
    )
    assert chain.jac["z1"]["u"].nnz == 0
    assert chain.jac["z2"]["x"].nnz == 0


@pytest.mark.parametrize("mode", _SWEEP_MODES)
def test_discipline_chain_sequential_variable_update(mode) -> None:
    """A discipline consuming and re-producing o chains both producers.

    B updates o in place (consumes A's o, produces a new o), so
    ∂z/∂x = ∂z/∂o·∂o_B/∂o_A·∂o_A/∂x and ∂z/∂u = ∂z/∂o·∂o_B/∂u.
    """
    disciplines = create_disciplines_from_desc([
        ("A", ["x"], ["o"]),
        ("B", ["o", "u"], ["o"]),
        ("C", ["o"], ["z"]),
    ])
    chain = DisciplineChain(disciplines)
    checker = DisciplineJacobianChecker(chain)
    assert checker.check(
        inputs=["x", "u"],
        outputs=["z"],
        linearization_mode=mode,
        approximation_mode=_APPROXIMATION_MODE,
        step=_APPROXIMATION_STEP,
        **_CHECK_JACOBIAN_KWARGS,
    )


@pytest.mark.parametrize("mode", _SWEEP_MODES)
def test_discipline_chain_output_independent_of_upstream_coupling(mode) -> None:
    """A downstream output independent of an upstream coupling is skipped.

    In chain A(x→c), B(c,x2→o1,o2), o1 depends on the coupling c while o2
    depends only on x2. Both sweeps must skip the (o2, c) term, so ∂o2/∂x is
    zero while ∂o1/∂x flows through c. AnalyticDiscipline provides the
    per-output sparsity that a fully-coupled LinearDiscipline cannot.
    """
    chain = DisciplineChain([
        AnalyticDiscipline({"c": "x"}, name="A"),
        AnalyticDiscipline({"o1": "c+x2", "o2": "2*x2"}, name="B"),
    ])
    checker = DisciplineJacobianChecker(chain)
    assert checker.check(
        {"x": ones(1), "x2": ones(1)},
        inputs=["x", "x2"],
        outputs=["o1", "o2"],
        linearization_mode=mode,
        approximation_mode=_APPROXIMATION_MODE,
        step=_APPROXIMATION_STEP,
        **_CHECK_JACOBIAN_KWARGS,
    )


@pytest.mark.parametrize("mode", _SWEEP_MODES)
def test_discipline_chain_single_discipline(mode) -> None:
    """A chain of one discipline reproduces its Jacobian in every mode."""
    chain = DisciplineChain(
        create_disciplines_from_desc([("A", ["x", "u"], ["y", "w"])])
    )
    checker = DisciplineJacobianChecker(chain)
    assert checker.check(
        linearization_mode=mode,
        approximation_mode=_APPROXIMATION_MODE,
        step=_APPROXIMATION_STEP,
        **_CHECK_JACOBIAN_KWARGS,
    )


@pytest.mark.parametrize(
    ("input_names", "output_names", "expected_mode"),
    [
        # Boundary: total output size == total input size favors reverse mode.
        (["x3"], ["b"], ChainDerivationMode.REVERSE),
        (["x1"], ["a"], ChainDerivationMode.FORWARD),
        (["x2"], ["y"], ChainDerivationMode.REVERSE),
    ],
)
def test_auto_linearization_mode_selection(
    heterogeneous_sizes_disciplines, input_names, output_names, expected_mode
) -> None:
    """AUTO picks reverse mode iff total output size ≤ total input size."""
    chain = DisciplineChain(heterogeneous_sizes_disciplines)
    chain.execute()
    assert (
        chain._DisciplineChain__select_linearization_mode(
            frozenset(input_names), frozenset(output_names)
        )
        == expected_mode
    )


def test_auto_linearization_mode_selection_without_data(
    heterogeneous_sizes_disciplines,
) -> None:
    """AUTO defaults to reverse mode when no data is available before execution."""
    chain = DisciplineChain(heterogeneous_sizes_disciplines)
    assert (
        chain._DisciplineChain__select_linearization_mode(
            frozenset(["x1"]), frozenset(["y"])
        )
        == ChainDerivationMode.REVERSE
    )


def test_discipline_chain_data_flow(heterogeneous_sizes_disciplines) -> None:
    """The process data flow lists the couplings between successive disciplines."""
    a, b, c = heterogeneous_sizes_disciplines
    chain = DisciplineChain([a, b, c])
    assert chain.get_process_flow().get_data_flow() == [(a, b, ["a"]), (b, c, ["b"])]


def test_jacobian_recomputed_on_io_subset_change(
    heterogeneous_sizes_disciplines,
) -> None:
    """Jacobian is correct after switching the differentiated I/O subset."""
    chain = DisciplineChain(heterogeneous_sizes_disciplines)
    checker = DisciplineJacobianChecker(chain)
    assert checker.check(
        inputs=["x1"],
        outputs=["y"],
        approximation_mode=_APPROXIMATION_MODE,
        step=_APPROXIMATION_STEP,
        **_CHECK_JACOBIAN_KWARGS,
    )
    assert checker.check(
        inputs=["x2", "x3"],
        outputs=["y"],
        approximation_mode=_APPROXIMATION_MODE,
        step=_APPROXIMATION_STEP,
        **_CHECK_JACOBIAN_KWARGS,
    )


def test_differentiated_ios_cached_on_repeated_io(
    heterogeneous_sizes_disciplines,
) -> None:
    """Repeated linearization with the same I/O reuses the cached coupling I/Os.

    _compute_jacobian rebuilds _discipline_to_ios only when the differentiated
    I/O key changes; a same-key call must reuse the previously built mapping
    instead of recomputing it via set_differentiated_ios.
    """
    chain = DisciplineChain(heterogeneous_sizes_disciplines)
    chain.execute()

    chain._compute_jacobian(frozenset(["x1"]), frozenset(["y"]))
    cached = chain._DisciplineChain__discipline_to_ios

    # set_differentiated_ios returns a fresh dict each call, so identity is
    # preserved iff the cache-hit branch skipped the rebuild.
    chain._compute_jacobian(frozenset(["x1"]), frozenset(["y"]))
    assert chain._DisciplineChain__discipline_to_ios is cached


@pytest.mark.parametrize("mode", _SWEEP_MODES)
def test_chain_jacobian_does_not_alias_subdiscipline_jacobians(
    heterogeneous_sizes_disciplines, mode
) -> None:
    """No chain Jacobian block shares its object with a sub-discipline block.

    Both sweeps seed chain outputs from the sub-disciplines' local Jacobians.
    These blocks must be copied, not referenced: mutating DisciplineChain.jac
    in place must never corrupt the Jacobian of a chained discipline.
    """
    chain = DisciplineChain(heterogeneous_sizes_disciplines)
    chain.linearization_mode = mode
    chain.execute()
    # x3 is a direct input of C, so C's local (y, x3) block is seeded as-is.
    chain._compute_jacobian(frozenset(["x1", "x2", "x3"]), frozenset(["y"]))

    discipline_block_ids = {
        id(block)
        for discipline in chain._disciplines
        for row in discipline.jac.values()
        for block in row.values()
    }
    chain_block_ids = {
        id(block) for row in chain.jac.values() for block in row.values()
    }
    assert discipline_block_ids.isdisjoint(chain_block_ids)


def test_adjoint_sweep_prunes_stale_coupling_entries(
    heterogeneous_sizes_disciplines,
) -> None:
    """Adjoint sweep prunes stale coupling entries from chain.jac.

    add_differentiated_inputs unions across calls: if a sub-discipline accumulated
    an input from a prior call, its Jacobian carries stale entries that would leak
    into chain.jac without explicit pruning.
    """
    _, b, _ = heterogeneous_sizes_disciplines
    # Pre-register "a" on disc_b as a stale entry from a fictitious prior call.
    # "a" is an internal coupling (A→B), not a chain-level input.
    b.add_differentiated_inputs(["a"])

    chain = DisciplineChain(heterogeneous_sizes_disciplines)
    chain.execute()
    chain._compute_jacobian(frozenset(["x2", "x3"]), frozenset(["y"]))

    # "a" leaked from B.jac into the adjoint sweep and must be pruned.
    for jac_row in chain.jac.values():
        assert "a" not in jac_row


@pytest.mark.parametrize("mode", _SWEEP_MODES)
@pytest.mark.parametrize(("input_names", "output_names"), _IO_SUBSETS)
def test_discipline_chain_complex_topology_jacobian(
    input_names, output_names, mode
) -> None:
    """Jacobian correctness on a 16-discipline topology, in forward and reverse.

    Covers multi-path couplings, skip connections, self-coupling (P→P),
    and backward coupling (M→K via m, N→E via n), in the forward and reverse
    sweeps (AUTO dispatches to one of them, exercised elsewhere).
    """
    chain = DisciplineChain(create_disciplines_from_desc(_DESCRIPTION))
    checker = DisciplineJacobianChecker(chain)
    assert checker.check(
        inputs=input_names,
        outputs=output_names,
        linearization_mode=mode,
        approximation_mode=_APPROXIMATION_MODE,
        step=_APPROXIMATION_STEP,
        **_CHECK_JACOBIAN_KWARGS,
    )
