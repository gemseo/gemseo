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
#        :author: Gilberto Ruiz Jimenez
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import array
from numpy.testing import assert_almost_equal

from gemseo import create_discipline
from gemseo import create_mda
from gemseo.core.discipline.namespace import namespaces_separator
from gemseo.discipline import AnalyticDiscipline
from gemseo.discipline import propagate_namespace
from gemseo.discipline.namespace import _compute_affected_ios
from gemseo.mda import MDAGaussSeidel
from gemseo.util.testing.helper import assert_exception

if TYPE_CHECKING:
    from gemseo.core.discipline import Discipline


def _create_sobieski_disciplines() -> list[Discipline]:
    """Return the four Sobieski disciplines.

    Returns:
        The Sobieski structure, aerodynamics, propulsion and mission disciplines.
    """
    return create_discipline([
        "SobieskiStructure",
        "SobieskiAerodynamics",
        "SobieskiPropulsion",
        "SobieskiMission",
    ])


def _create_two_step_chain() -> tuple[Discipline, Discipline]:
    """Return a two-discipline chain.

    The chain is ``disc_1`` (``x`` -> ``y``) followed by ``disc_2`` (``y`` -> ``z``).

    Returns:
        The disciplines ``disc_1`` and ``disc_2``.
    """
    disc_1 = AnalyticDiscipline(expressions={"y": "x + 1"})
    disc_2 = AnalyticDiscipline(expressions={"z": "y + 1"})
    return disc_1, disc_2


def _create_chain() -> tuple[Discipline, Discipline, Discipline]:
    """Return a linear chain plus a disconnected discipline.

    The chain is ``a`` (``x`` -> ``y``) followed by ``b`` (``y`` -> ``z``).
    The ``c`` discipline (``w`` -> ``v``) is disconnected from the chain.

    Returns:
        The disciplines ``a``, ``b`` and ``c``.
    """
    disc_a, disc_b = _create_two_step_chain()
    disc_c = create_discipline("AnalyticDiscipline", expressions={"v": "w + 1"})
    return disc_a, disc_b, disc_c


def _create_branching_group() -> tuple[Discipline, Discipline, Discipline, Discipline]:
    """Return a group whose seed discipline feeds two branches that never merge.

    The group is ``a`` (``x`` -> ``y``, ``z``), ``b`` (``y`` -> ``u``),
    ``c`` (``u`` -> ``v``) and ``d`` (``z`` -> ``w``).

    Returns:
        The disciplines ``a``, ``b``, ``c`` and ``d``.
    """
    disc_a = AnalyticDiscipline(expressions={"y": "x + 1", "z": "x + 2"})
    disc_b = AnalyticDiscipline(expressions={"u": "y + 1"})
    disc_c = AnalyticDiscipline(expressions={"v": "u + 1"})
    disc_d = AnalyticDiscipline(expressions={"w": "z + 1"})
    return disc_a, disc_b, disc_c, disc_d


def _create_partially_excluded_group() -> tuple[Discipline, Discipline]:
    """Return a group where a discipline consumes an excluded and a non-excluded input.

    The group is ``a`` (``x`` -> ``y``, ``z``) followed by ``b`` (``y``, ``z``
    -> ``u``), so the edge from ``a`` to ``b`` carries both an excluded and a
    non-excluded coupling.

    Returns:
        The disciplines ``a`` and ``b``.
    """
    disc_a = AnalyticDiscipline(expressions={"y": "x + 1", "z": "x + 2"})
    disc_b = AnalyticDiscipline(expressions={"u": "y + z"})
    return disc_a, disc_b


def test_seed_propagates_to_whole_coupling_graph() -> None:
    """The whole Sobieski coupling graph is reached from the aerodynamics seed."""
    disciplines = _create_sobieski_disciplines()

    affected = propagate_namespace(disciplines, "platform", {"x_2"})

    assert set(affected) == set(disciplines)


def test_all_outputs_of_reached_disciplines_are_namespaced() -> None:
    """Every output of every reached discipline carries the namespace."""
    disciplines = _create_sobieski_disciplines()
    prefix = f"platform{namespaces_separator}"

    propagate_namespace(disciplines, "platform", {"x_2"})

    for discipline in disciplines:
        for output_name in discipline.io.output_grammar:
            assert output_name.startswith(prefix)


def test_variables_not_produced_in_reached_set_stay_bare() -> None:
    """Inputs that are neither seeds nor produced within the group stay bare."""
    disciplines = _create_sobieski_disciplines()
    prefix = f"platform{namespaces_separator}"

    propagate_namespace(disciplines, "platform", {"x_2"})

    all_input_names = set()
    for discipline in disciplines:
        all_input_names.update(discipline.io.input_grammar)

    for global_name in ("x_shared", "x_1", "x_3"):
        assert global_name in all_input_names
        assert f"{prefix}{global_name}" not in all_input_names


def test_seed_and_couplings_are_namespaced_where_consumed() -> None:
    """The seed and all the couplings are renamed wherever they are consumed."""
    disciplines = _create_sobieski_disciplines()
    prefix = f"platform{namespaces_separator}"

    propagate_namespace(disciplines, "platform", {"x_2"})

    all_input_names = set()
    for discipline in disciplines:
        all_input_names.update(discipline.io.input_grammar)

    couplings = (
        "y_12",
        "y_14",
        "y_21",
        "y_23",
        "y_24",
        "y_31",
        "y_32",
        "y_34",
    )
    for name in (*couplings, "x_2"):
        assert f"{prefix}{name}" in all_input_names
        assert name not in all_input_names


def test_namespaced_group_solves_the_same_mda() -> None:
    """An MDA over the namespaced group returns the same solution as before."""
    prefix = f"platform{namespaces_separator}"
    reference = create_mda("MDAGaussSeidel", _create_sobieski_disciplines()).execute()

    disciplines = _create_sobieski_disciplines()
    affected = propagate_namespace(disciplines, "platform", {"x_2"})
    namespaced = create_mda("MDAGaussSeidel", disciplines).execute()

    output_names = set()
    for ios in affected.values():
        output_names |= ios.outputs

    # The couplings survived the renaming: the MDA still converges to the same point.
    assert output_names
    for name in output_names:
        assert_almost_equal(namespaced[f"{prefix}{name}"], reference[name])


def test_compute_affected_ios_does_not_mutate() -> None:
    """The dry-run computation leaves the grammars untouched."""
    disciplines = _create_sobieski_disciplines()
    before = {
        discipline: (
            sorted(discipline.io.input_grammar),
            sorted(discipline.io.output_grammar),
        )
        for discipline in disciplines
    }

    affected, _ = _compute_affected_ios(disciplines, {"x_2"}, set())
    assert affected

    for discipline in disciplines:
        assert sorted(discipline.io.input_grammar) == before[discipline][0]
        assert sorted(discipline.io.output_grammar) == before[discipline][1]
        # No namespace separator was introduced anywhere.
        for name in (*discipline.io.input_grammar, *discipline.io.output_grammar):
            assert namespaces_separator not in name


def test_empty_seed_names_is_a_no_op() -> None:
    """Propagating an empty set of seed names leaves the disciplines untouched."""
    disciplines = _create_chain()

    affected = propagate_namespace(disciplines, "ns", ())

    assert affected == {}
    for discipline in disciplines:
        for name in (*discipline.io.input_grammar, *discipline.io.output_grammar):
            assert namespaces_separator not in name


def test_partial_reach() -> None:
    """A discipline disconnected from the seed is left untouched."""
    disc_a, disc_b, disc_c = _create_chain()
    prefix = f"ns{namespaces_separator}"

    affected = propagate_namespace([disc_a, disc_b, disc_c], "ns", {"x"})

    # Only the disciplines downstream of the seed are reached.
    assert set(affected) == {disc_a, disc_b}
    assert disc_c not in affected

    assert list(disc_a.io.input_grammar) == [f"{prefix}x"]
    assert list(disc_a.io.output_grammar) == [f"{prefix}y"]
    assert list(disc_b.io.input_grammar) == [f"{prefix}y"]
    assert list(disc_b.io.output_grammar) == [f"{prefix}z"]

    # The disconnected discipline keeps bare names.
    assert list(disc_c.io.input_grammar) == ["w"]
    assert list(disc_c.io.output_grammar) == ["v"]


def test_seed_on_coupling_variable() -> None:
    """Seeding a coupling variable still propagates to downstream disciplines."""
    disc_a, disc_b, disc_c = _create_chain()
    prefix = f"ns{namespaces_separator}"

    affected = propagate_namespace([disc_a, disc_b, disc_c], "ns", {"y"})

    assert set(affected) == {disc_a, disc_b}

    # y is an output of a and an input of b: both are sources and get namespaced.
    assert list(disc_a.io.output_grammar) == [f"{prefix}y"]
    assert list(disc_b.io.input_grammar) == [f"{prefix}y"]
    assert list(disc_b.io.output_grammar) == [f"{prefix}z"]

    # x is not the seed and is not produced within the group, so it stays bare.
    assert list(disc_a.io.input_grammar) == ["x"]

    # The disconnected discipline is untouched.
    assert list(disc_c.io.input_grammar) == ["w"]
    assert list(disc_c.io.output_grammar) == ["v"]


def test_unknown_seed_raises_without_mutation(snapshot) -> None:
    """A seed owned by no discipline raises and leaves the group untouched."""
    disc_a, disc_b, disc_c = _create_chain()

    with assert_exception(ValueError, snapshot):
        propagate_namespace([disc_a, disc_b, disc_c], "ns", {"other", "unknown"})

    for discipline in (disc_a, disc_b, disc_c):
        for name in (*discipline.io.input_grammar, *discipline.io.output_grammar):
            assert namespaces_separator not in name


def test_already_namespaced_raises_without_mutation(snapshot) -> None:
    """Pre-namespaced affected variables raise and leave the group untouched.

    Both `disc_a` and `disc_e` own the seed `x` directly, so both are sources and
    stay reached regardless of their own output already carrying a namespace,
    which lets this test exercise two offenders at once.
    """
    disc_a = create_discipline(
        "AnalyticDiscipline", expressions={"y": "x + 1"}, name="a"
    )
    disc_e = create_discipline(
        "AnalyticDiscipline", expressions={"q": "x + 2"}, name="e"
    )
    disc_c = create_discipline(
        "AnalyticDiscipline", expressions={"v": "w + 1"}, name="c"
    )
    disc_a.add_namespace_to_output("y", "pre1")
    disc_e.add_namespace_to_output("q", "pre2")
    disciplines = [disc_a, disc_e, disc_c]
    before = {
        discipline: (
            sorted(discipline.io.input_grammar),
            sorted(discipline.io.output_grammar),
        )
        for discipline in disciplines
    }

    with assert_exception(ValueError, snapshot):
        propagate_namespace(disciplines, "ns", {"x"})

    for discipline in disciplines:
        assert sorted(discipline.io.input_grammar) == before[discipline][0]
        assert sorted(discipline.io.output_grammar) == before[discipline][1]


def test_producer_outside_reached_set_raises_without_mutation(snapshot) -> None:
    """Couplings produced both inside and outside the reached set raise."""
    disc_a = create_discipline(
        "AnalyticDiscipline", expressions={"u": "x + 1"}, name="a"
    )
    disc_f = create_discipline(
        "AnalyticDiscipline", expressions={"u": "w + 1"}, name="f"
    )
    disc_d = create_discipline(
        "AnalyticDiscipline", expressions={"z": "u + 1"}, name="d"
    )
    disc_g = create_discipline(
        "AnalyticDiscipline", expressions={"z": "t + 1"}, name="g"
    )
    disciplines = [disc_a, disc_f, disc_d, disc_g]
    before = {
        discipline: (
            sorted(discipline.io.input_grammar),
            sorted(discipline.io.output_grammar),
        )
        for discipline in disciplines
    }

    with assert_exception(ValueError, snapshot):
        propagate_namespace(disciplines, "ns", {"x"})

    for discipline in disciplines:
        assert sorted(discipline.io.input_grammar) == before[discipline][0]
        assert sorted(discipline.io.output_grammar) == before[discipline][1]


def test_excluded_variable_cuts_a_branch() -> None:
    """An excluded variable stops the propagation through the edge carrying only it."""
    disc_a, disc_b, disc_c, disc_d = _create_branching_group()
    prefix = f"left{namespaces_separator}"
    before_b = (
        sorted(disc_b.io.input_grammar),
        sorted(disc_b.io.output_grammar),
    )
    before_c = (
        sorted(disc_c.io.input_grammar),
        sorted(disc_c.io.output_grammar),
    )

    affected, _ = _compute_affected_ios([disc_a, disc_b, disc_c, disc_d], {"x"}, {"y"})

    # b and c are only reachable through the edge carrying the excluded y, so
    # they are not reached at all.
    assert set(affected) == {disc_a, disc_d}
    assert affected[disc_a].inputs == frozenset({"x"})
    assert affected[disc_a].outputs == frozenset({"y", "z"})
    assert affected[disc_d].inputs == frozenset({"z"})
    assert affected[disc_d].outputs == frozenset({"w"})

    propagate_namespace([disc_a, disc_b, disc_c, disc_d], "left", {"x"}, {"y"})

    assert list(disc_a.io.input_grammar) == [f"{prefix}x"]
    # y is excluded, but a still produces it, so leaving it bare would give the
    # combined group two producers of the bare name: it is namespaced too,
    # even though its namespaced copy is then consumed by nobody.
    assert sorted(disc_a.io.output_grammar) == [f"{prefix}y", f"{prefix}z"]
    assert list(disc_d.io.input_grammar) == [f"{prefix}z"]
    assert list(disc_d.io.output_grammar) == [f"{prefix}w"]

    # b and c are untouched.
    assert (
        sorted(disc_b.io.input_grammar),
        sorted(disc_b.io.output_grammar),
    ) == before_b
    assert (
        sorted(disc_c.io.input_grammar),
        sorted(disc_c.io.output_grammar),
    ) == before_c


def test_reached_discipline_with_only_excluded_outputs_is_pruned() -> None:
    """A reached discipline whose outputs are all excluded is pruned and untouched."""
    disc_a, disc_b = _create_two_step_chain()
    before_b = (
        sorted(disc_b.io.input_grammar),
        sorted(disc_b.io.output_grammar),
    )

    affected, _ = _compute_affected_ios([disc_a, disc_b], {"x"}, {"z"})

    # b's only output, z, is excluded: its namespaced copy would be consumed by
    # nobody, so b is pruned from the reached set even though it is reachable
    # from the seed through the non-excluded y.
    assert set(affected) == {disc_a}

    propagate_namespace([disc_a, disc_b], "left", {"x"}, {"z"})

    assert (
        sorted(disc_b.io.input_grammar),
        sorted(disc_b.io.output_grammar),
    ) == before_b


def test_seed_discipline_with_only_excluded_outputs_is_pruned() -> None:
    """A seed discipline whose outputs are all excluded is pruned too."""
    disc_a, disc_b = _create_two_step_chain()

    affected, _ = _compute_affected_ios([disc_a, disc_b], {"x"}, {"y"})

    # a is the seed, but its only output y is excluded, so it is pruned like any
    # other discipline whose outputs are all excluded; b is then unreachable
    # since the edge carrying only y is cut.
    assert affected == {}


def test_excluded_input_stays_bare_when_discipline_reached_through_other_path() -> None:
    """An excluded input stays bare even when its discipline is reached otherwise."""
    disc_a, disc_b = _create_partially_excluded_group()
    prefix = f"left{namespaces_separator}"

    affected = propagate_namespace([disc_a, disc_b], "left", {"x"}, {"y"})

    # The edge from a to b also carries z, which is not excluded, so it is not
    # cut and b is reached.
    assert disc_b in affected
    assert list(disc_b.io.output_grammar) == [f"{prefix}u"]
    input_names = set(disc_b.io.input_grammar)
    assert "y" in input_names
    assert f"{prefix}y" not in input_names
    assert f"{prefix}z" in input_names


def test_excluded_name_outside_reached_set_is_a_no_op() -> None:
    """An excluded variable outside the reached set changes nothing."""
    disciplines = _create_chain()

    with_excluded, _ = _compute_affected_ios(disciplines, {"x"}, {"w"})
    without_excluded, _ = _compute_affected_ios(disciplines, {"x"}, set())

    assert with_excluded == without_excluded


def test_seed_and_excluded_overlap_raises_without_mutation(snapshot) -> None:
    """A name that is both seed and excluded raises and leaves the group untouched."""
    disciplines = _create_chain()
    before = {
        discipline: (
            sorted(discipline.io.input_grammar),
            sorted(discipline.io.output_grammar),
        )
        for discipline in disciplines
    }

    with assert_exception(ValueError, snapshot):
        propagate_namespace(disciplines, "ns", {"x"}, {"x"})

    for discipline in disciplines:
        assert sorted(discipline.io.input_grammar) == before[discipline][0]
        assert sorted(discipline.io.output_grammar) == before[discipline][1]


def test_unknown_excluded_name_raises_without_mutation(snapshot) -> None:
    """An excluded name owned by no discipline raises and leaves the group untouched."""
    disciplines = _create_chain()
    before = {
        discipline: (
            sorted(discipline.io.input_grammar),
            sorted(discipline.io.output_grammar),
        )
        for discipline in disciplines
    }

    with assert_exception(ValueError, snapshot):
        propagate_namespace(disciplines, "ns", {"x"}, {"unknown"})

    for discipline in disciplines:
        assert sorted(discipline.io.input_grammar) == before[discipline][0]
        assert sorted(discipline.io.output_grammar) == before[discipline][1]


def test_excluded_variable_reuses_the_original_discipline_in_an_mda() -> None:
    """An MDA over original and namespaced disciplines reuses the excluded coupling."""
    prefix = f"left{namespaces_separator}"
    original_input = {"x": array([1.0])}
    reference = MDAGaussSeidel(list(_create_branching_group())).execute(original_input)

    original_a, original_b, original_c, original_d = _create_branching_group()
    copy_a, copy_b, copy_c, copy_d = _create_branching_group()
    affected = propagate_namespace(
        [copy_a, copy_b, copy_c, copy_d], "left", {"x"}, {"y"}
    )

    # copy_b and copy_c dropped out of the reached set: a's only
    # non-excluded output is z, so b and c, only reachable through the excluded
    # y, are not reached at all. They are not part of the assembled group below;
    # the original b and c, which produce the same bare names, are reused
    # instead of being duplicated.
    assert set(affected) == {copy_a, copy_d}

    # The two seeds take different values, so the original branch and the
    # namespaced one compute different numbers.
    combined = MDAGaussSeidel([
        original_a,
        original_b,
        original_c,
        original_d,
        copy_a,
        copy_d,
    ]).execute({**original_input, f"{prefix}x": array([10.0])})

    # The MDA solves, and original_b and original_c consume the bare "y" produced
    # by original_a: the excluded variable keeps the original value, whatever
    # the namespaced branch computes for its own "left:y".
    assert_almost_equal(combined["u"], reference["u"])
    assert_almost_equal(combined["v"], reference["v"])
    assert f"{prefix}y" in combined
    assert f"{prefix}w" in combined


def test_propagate_namespace_grammar_order_is_sorted() -> None:
    """The renamed grammar elements end up in sorted order, reproducibly."""
    disciplines = _create_sobieski_disciplines()
    prefix = f"platform{namespaces_separator}"

    affected = propagate_namespace(disciplines, "platform", {"x_2"})

    for discipline in disciplines:
        namespaced_inputs = [
            name for name in discipline.io.input_grammar if name.startswith(prefix)
        ]
        namespaced_outputs = [
            name for name in discipline.io.output_grammar if name.startswith(prefix)
        ]
        assert namespaced_inputs == sorted(namespaced_inputs)
        assert namespaced_outputs == sorted(namespaced_outputs)
        assert set(namespaced_inputs) == {
            f"{prefix}{name}" for name in affected[discipline].inputs
        }
        assert set(namespaced_outputs) == {
            f"{prefix}{name}" for name in affected[discipline].outputs
        }
