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
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import pytest
from numpy import array
from numpy import ndarray
from numpy.testing import assert_equal
from scipy.sparse import issparse

from gemseo.problem.mdo.scalable.linear.linear_discipline import LinearDiscipline
from gemseo.util.testing.helper import assert_exception


@pytest.mark.parametrize(
    "matrix_format",
    LinearDiscipline.MatrixFormat,
)
def test_jacobian_format(matrix_format: LinearDiscipline.MatrixFormat) -> None:
    """Test that the Jacobian matrix has the specified size."""
    discipline = LinearDiscipline(
        name="A",
        input_names=["i1", "i2", "i3"],
        output_names=["o1", "o2", "o3"],
        inputs_size=10,
        outputs_size=20,
        matrix_format=matrix_format,
    )
    # Check format and size of the underlying matrix
    if matrix_format == LinearDiscipline.MatrixFormat.DENSE:
        assert isinstance(discipline.mat, ndarray)
    else:
        assert discipline.mat.format == matrix_format

    assert discipline.mat.shape == (60, 30)

    # Check format and size of the different blocks
    jac = discipline.linearize(compute_all_jacobians=True)

    for output_name in ["o1", "o2", "o3"]:
        for input_name in ["i1", "i2", "i3"]:
            if matrix_format == LinearDiscipline.MatrixFormat.DENSE:
                assert isinstance(discipline.mat, ndarray)
            else:
                assert issparse(jac[output_name][input_name])

            assert jac[output_name][input_name].shape == (20, 10)


@pytest.mark.parametrize("linearize_first", [True, False])
def test_compute_jac_at_run(linearize_first: bool):
    """Tests the computation of the Jacobian at execute()"""
    discipline = LinearDiscipline(
        name="A",
        input_names=["i"],
        output_names=["o"],
        compute_jac_at_run=True,
    )
    discipline.add_differentiated_outputs("o")
    discipline.add_differentiated_inputs("i")

    if linearize_first:
        discipline.linearize(execute=False)
    else:
        discipline.execute()

    assert "o" in discipline.jac
    assert "i" in discipline.jac["o"]


@pytest.fixture(params=[False, True], ids=["jac_at_linearize", "jac_at_run"])
def namespaced_discipline(request) -> LinearDiscipline:
    """A LinearDiscipline whose first input and first output have a namespace."""
    discipline = LinearDiscipline(
        "A", ["i1", "i2"], ["o1", "o2"], compute_jac_at_run=request.param
    )
    discipline.add_namespace_to_input("i1", "ns")
    discipline.add_namespace_to_output("o1", "ns")
    return discipline


def test_namespaces(namespaced_discipline):
    """Check that the namespaces do not change the order of the variables."""
    namespaced_discipline.execute({"ns:i1": array([1.0]), "i2": array([2.0])})
    output_vector = namespaced_discipline.mat @ array([1.0, 2.0])
    assert_equal(namespaced_discipline.io.output_data["ns:o1"], output_vector[:1])
    assert_equal(namespaced_discipline.io.output_data["o2"], output_vector[1:])


def test_namespaces_jacobian(namespaced_discipline):
    """Check the Jacobian when an input and an output have a namespace."""
    namespaced_discipline.execute({"ns:i1": array([1.0]), "i2": array([2.0])})
    matrix = namespaced_discipline.mat
    jac = namespaced_discipline.linearize(compute_all_jacobians=True)
    assert_equal(jac["ns:o1"]["ns:i1"], matrix[:1, :1])
    assert_equal(jac["ns:o1"]["i2"], matrix[:1, 1:])
    assert_equal(jac["o2"]["ns:i1"], matrix[1:, :1])
    assert_equal(jac["o2"]["i2"], matrix[1:, 1:])


@pytest.mark.parametrize(
    ("input_names", "output_names"),
    [(["x", "y", "x"], ["z"]), (["x"], ["z", "z"])],
    ids=["input", "output"],
)
def test_repeated_name(input_names, output_names, snapshot) -> None:
    """Check the error when an input name or an output name is repeated."""
    with assert_exception(ValueError, snapshot):
        LinearDiscipline("foo", input_names, output_names)
