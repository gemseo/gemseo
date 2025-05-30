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
#        :author: François Gallard
#        :author: Alexandre Scotto Di Perrotolo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import re

import pytest
from numpy import ndarray

from gemseo import create_mda
from gemseo.core.discipline import Discipline
from gemseo.problems.mdo.scalable.linear.disciplines_generator import DESC_3_DISC_WEAK
from gemseo.problems.mdo.scalable.linear.disciplines_generator import DESC_4_DISC_WEAK
from gemseo.problems.mdo.scalable.linear.disciplines_generator import DESC_16_DISC
from gemseo.problems.mdo.scalable.linear.disciplines_generator import DESC_DISC_REPEATED
from gemseo.problems.mdo.scalable.linear.disciplines_generator import _get_disc_names
from gemseo.problems.mdo.scalable.linear.disciplines_generator import (
    create_disciplines_from_desc,
)
from gemseo.problems.mdo.scalable.linear.disciplines_generator import (
    create_disciplines_from_sizes,
)
from gemseo.problems.mdo.scalable.linear.linear_discipline import LinearDiscipline

DESCRIPTIONS = [
    DESC_3_DISC_WEAK,
    DESC_4_DISC_WEAK,
    DESC_16_DISC,
    DESC_DISC_REPEATED,
]


def test_fail_no_output() -> None:
    """Test that the LinearDiscipline fails when there are no inputs or outputs in the
    description."""
    with pytest.raises(ValueError, match=re.escape("output_names must not be empty.")):
        create_disciplines_from_desc([("A", ["x"], [])])
    with pytest.raises(ValueError, match=re.escape("input_names must not be empty.")):
        create_disciplines_from_desc([("A", [], ["y"])])


@pytest.mark.parametrize("descriptions", DESCRIPTIONS)
def test_creation(descriptions) -> None:
    """Test that the disciplines are well generated according to spec."""
    disciplines = create_disciplines_from_desc(descriptions)
    assert len(disciplines) == len(descriptions)
    for disc, description in zip(disciplines, descriptions):
        assert disc.name == description[0]
        out = disc.execute()
        assert sorted(out.keys()) == sorted(list(description[2]) + list(description[1]))


@pytest.mark.parametrize("desc", DESCRIPTIONS[:-1])
def test_mda_convergence(desc) -> None:
    """Test that the generated disciplines have an equilibrium point."""
    disciplines = create_disciplines_from_desc(desc)
    tolerance = 1e-12
    mda = create_mda(
        "MDAGaussSeidel", disciplines, tolerance=tolerance, max_mda_iter=1000
    )
    mda.execute()

    assert mda.normed_residual <= tolerance


def test_lin_disc_jac() -> None:
    """Test _compute_jacobian."""
    desc = [("A", ["x"], ["a"]), ("B", ["a", "x"], ["b", "c"])]
    disciplines = create_disciplines_from_desc(desc)
    for disc in disciplines:
        assert disc.check_jacobian()


@pytest.mark.parametrize(
    "grammar_type", [Discipline.GrammarType.JSON, Discipline.GrammarType.SIMPLE]
)
def test_create_disciplines_from_sizes(grammar_type) -> None:
    """Test that the disciplines are well created according to the specifications."""
    nb_of_disc = 2
    nb_of_total_disc_io = 10
    nb_of_disc_inputs = 4
    nb_of_disc_outputs = 5
    inputs_size = 6
    outputs_size = 7

    disciplines = create_disciplines_from_sizes(
        nb_of_disc,
        nb_of_total_disc_io,
        nb_of_disc_inputs,
        nb_of_disc_outputs,
        inputs_size,
        outputs_size,
        grammar_type=grammar_type,
    )

    assert len(disciplines) == nb_of_disc

    for disc in disciplines:
        assert len(disc.io.input_grammar) == nb_of_disc_inputs
        assert len(disc.io.output_grammar) == nb_of_disc_outputs
        for d in disc.io.input_grammar.defaults.values():
            assert d.size == inputs_size

        out = disc.execute()
        for output_name in disc.io.output_grammar:
            assert out[output_name].size == outputs_size


@pytest.mark.parametrize(
    ("nb_of_disc_inputs", "nb_of_disc_outputs", "kind"),
    [
        (3, 1, "inputs"),
        (1, 3, "outputs"),
    ],
)
def test_sizes_errors(nb_of_disc_inputs, nb_of_disc_outputs, kind) -> None:
    """Test that the inputs consistency errors."""
    with pytest.raises(ValueError, match=f"The number of disciplines {kind}"):
        create_disciplines_from_sizes(
            1,
            2,
            nb_of_disc_inputs=nb_of_disc_inputs,
            nb_of_disc_outputs=nb_of_disc_outputs,
        )


@pytest.mark.parametrize("nb_of_names", [1, 27])
def test_get_disc_names(nb_of_names) -> None:
    """Test names generator."""
    names = _get_disc_names(nb_of_names)
    assert len(set(names)) == nb_of_names


@pytest.mark.parametrize(
    "matrix_format",
    LinearDiscipline.MatrixFormat,
)
def test_jacobian_format(matrix_format: LinearDiscipline.MatrixFormat) -> None:
    """Test that the Jacobian matrices have the specified format."""
    disciplines = create_disciplines_from_desc(
        DESC_16_DISC,
        inputs_size=10,
        outputs_size=20,
        matrix_format=matrix_format,
        matrix_density=0.1,
    )

    for discipline in disciplines:
        if matrix_format == LinearDiscipline.MatrixFormat.DENSE:
            assert isinstance(discipline.mat, ndarray)
        else:
            assert discipline.mat.format == matrix_format

        assert discipline.mat.shape == (discipline.size_out, discipline.size_in)
