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
from numpy import ndarray
from scipy.sparse import issparse

from gemseo.problems.scalable.linear.linear_discipline import LinearDiscipline


@pytest.mark.parametrize(
    "matrix_format",
    LinearDiscipline.MatrixFormat,
)
def test_jacobian_format(matrix_format: LinearDiscipline.MatrixFormat):
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
