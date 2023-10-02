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

from typing import Mapping
from typing import Sequence

import pytest
from gemseo.algos.sequence_transformer.acceleration import AccelerationMethod
from gemseo.core.discipline import MDODiscipline
from gemseo.mda.gauss_seidel import MDAGaussSeidel
from gemseo.mda.jacobi import MDAJacobi
from gemseo.mda.mda import MDA
from gemseo.problems.scalable.linear.disciplines_generator import (
    create_disciplines_from_desc,
)
from gemseo.problems.scalable.linear.linear_discipline import LinearDiscipline
from numpy import concatenate
from numpy.linalg import norm
from numpy.random import seed

seed(1)


@pytest.fixture(scope="module")
def disciplines() -> Sequence[LinearDiscipline]:
    """Return a set of coupled disciplines.

    The disciplines are defined such that the set of input couplings differs from the
    one of strong couplings.
    """
    disciplines = create_disciplines_from_desc(
        [
            ("A", ["x"], ["a"]),
            ("B", ["a", "y"], ["b"]),
            ("C", ["b"], ["y"]),
        ],
        inputs_size=10,
        outputs_size=10,
    )

    # Scale the disciplines to illustrate the different stopping criteria
    disciplines[1].mat /= 100.0
    disciplines[2].mat *= 100.0

    for discipline in disciplines:
        discipline.set_cache_policy(MDODiscipline.CacheType.MEMORY_FULL)

    return disciplines


def get_jacobi_reference_residuals(
    disciplines: list[MDODiscipline],
) -> tuple[Mapping, Mapping]:
    """Compute the initial and final residual without scaling for MDAJacobi."""
    mda = MDAJacobi(
        disciplines,
        max_mda_iter=5,
        acceleration_method=AccelerationMethod.NONE,
    )

    for discipline in disciplines:
        discipline.execute()

    # Compute the initial residual
    input, output = {}, {}
    for discipline in disciplines:
        input.update(discipline.get_input_data())
        output.update(discipline.get_output_data())

    initial_residual = {}
    for coupling in ["a", "b", "y"]:
        initial_residual[coupling] = input[coupling] - output[coupling]

    mda.scaling = MDA.ResidualScaling.NO_SCALING
    mda.execute()

    # Compute the final residual
    input, output = {}, {}
    for discipline in disciplines:
        input.update(discipline.get_input_data())
        output.update(discipline.get_output_data())

    final_residual = {}
    for coupling in ["a", "b", "y"]:
        final_residual[coupling] = input[coupling] - output[coupling]

    return initial_residual, final_residual


@pytest.mark.parametrize("scaling_strategy", MDA.ResidualScaling)
def test_scaling_strategy_jacobi(disciplines: list[MDODiscipline], scaling_strategy):
    """Tests the different scaling strategies for MDAJacobi."""
    initial_residual, final_residual = get_jacobi_reference_residuals(disciplines)

    initial_residual_vector = concatenate(list(initial_residual.values()))
    final_residual_vector = concatenate(list(final_residual.values()))

    mda = MDAJacobi(
        disciplines,
        max_mda_iter=5,
        acceleration_method=AccelerationMethod.NONE,
    )

    mda.scaling = scaling_strategy
    mda.execute()

    if scaling_strategy == MDA.ResidualScaling.NO_SCALING:
        assert mda._scaling_data is None
        assert mda.residual_history[-1] == norm(final_residual_vector)

    elif scaling_strategy == MDA.ResidualScaling.INITIAL_RESIDUAL_NORM:
        assert mda._scaling_data == norm(initial_residual_vector)
        assert mda.residual_history[-1] == norm(final_residual_vector) / norm(
            initial_residual_vector
        )

    elif scaling_strategy == MDA.ResidualScaling.INITIAL_RESIDUAL_COMPONENT:
        assert norm(mda._scaling_data - initial_residual_vector) == 0
        assert (
            mda.residual_history[-1]
            == abs(final_residual_vector / initial_residual_vector).max()
        )

    elif scaling_strategy == MDA.ResidualScaling.N_COUPLING_VARIABLES:
        assert mda._scaling_data == 30**0.5
        assert mda.residual_history[-1] == norm(final_residual_vector) / 30**0.5

    elif scaling_strategy == MDA.ResidualScaling.INITIAL_SUBRESIDUAL_NORM:
        for i, subres in enumerate(initial_residual.values()):
            subres_norm = mda._scaling_data[i][1]
            assert subres_norm == norm(subres)

        # Check that gathering all the sub-residuals norms yields the initial norm
        result = 0.0
        for _, subres_norm in mda._scaling_data:
            result += subres_norm**2
        assert abs(result**0.5 - norm(initial_residual_vector)) < 1e-15

        subres_norms = []
        for coupling in initial_residual.keys():
            subres_norms.append(
                norm(final_residual[coupling]) / norm(initial_residual[coupling])
            )
        assert mda.residual_history[-1] == max(subres_norms)

    elif scaling_strategy == MDA.ResidualScaling.SCALED_INITIAL_RESIDUAL_COMPONENT:
        assert norm(mda._scaling_data - initial_residual_vector) == 0
        assert (
            abs(
                mda.residual_history[-1]
                - norm(final_residual_vector / initial_residual_vector) / 30**0.5
            )
            < 1e-15
        )


def get_gauss_seidel_reference_residuals(
    disciplines: list[MDODiscipline],
) -> tuple[Mapping, Mapping]:
    """Compute the initial and final residual without scaling for MDAGaussSeidel."""
    mda = MDAGaussSeidel(
        disciplines,
        max_mda_iter=5,
        acceleration_method=AccelerationMethod.NONE,
    )
    mda.scaling = MDA.ResidualScaling.NO_SCALING
    mda.execute()

    _b = []
    for value in disciplines[1].cache:
        _b.append(value.outputs["b"])

    _y = []
    for value in disciplines[2].cache:
        _y.append(value.outputs["y"])

    initial_residual = {"b": _b[0] - _b[1], "y": _y[0] - _y[1]}
    final_residual = {"b": _b[-2] - _b[-1], "y": _y[-2] - _y[-1]}

    for discipline in disciplines:
        discipline.cache.clear()

    return initial_residual, final_residual


@pytest.mark.parametrize("scaling_strategy", MDA.ResidualScaling)
def test_scaling_strategy_gauss_seidel(
    disciplines: list[MDODiscipline], scaling_strategy
):
    """Tests the different scaling strategies for MDAGaussSeidel."""
    initial_residual, final_residual = get_gauss_seidel_reference_residuals(disciplines)

    initial_residual_vector = concatenate(list(initial_residual.values()))
    final_residual_vector = concatenate(list(final_residual.values()))

    mda = MDAGaussSeidel(
        disciplines,
        max_mda_iter=5,
        acceleration_method=AccelerationMethod.NONE,
    )

    mda.scaling = scaling_strategy
    mda.execute()

    if scaling_strategy == MDA.ResidualScaling.NO_SCALING:
        assert mda._scaling_data is None
        assert mda.residual_history[-1] == norm(final_residual_vector)

    elif scaling_strategy == MDA.ResidualScaling.INITIAL_RESIDUAL_NORM:
        assert mda._scaling_data == norm(initial_residual_vector)
        assert mda.residual_history[-1] == norm(final_residual_vector) / norm(
            initial_residual_vector
        )

    elif scaling_strategy == MDA.ResidualScaling.INITIAL_RESIDUAL_COMPONENT:
        assert norm(mda._scaling_data - initial_residual_vector) == 0
        assert (
            mda.residual_history[-1]
            == abs(final_residual_vector / initial_residual_vector).max()
        )

    elif scaling_strategy == MDA.ResidualScaling.N_COUPLING_VARIABLES:
        assert mda._scaling_data == 20**0.5
        assert mda.residual_history[-1] == norm(final_residual_vector) / 20**0.5

    elif scaling_strategy == MDA.ResidualScaling.INITIAL_SUBRESIDUAL_NORM:
        for i, subres in enumerate(initial_residual.values()):
            subres_norm = mda._scaling_data[i][1]
            assert subres_norm == norm(subres)

        # Check that gathering all the sub-residuals norms yields the initial norm
        result = 0.0
        for _, subres_norm in mda._scaling_data:
            result += subres_norm**2
        assert abs(result - norm(initial_residual_vector) ** 2) < 1e-12

        subres_norms = []
        for coupling in initial_residual.keys():
            subres_norms.append(
                norm(final_residual[coupling]) / norm(initial_residual[coupling])
            )
        assert mda.residual_history[-1] == max(subres_norms)

    elif scaling_strategy == MDA.ResidualScaling.SCALED_INITIAL_RESIDUAL_COMPONENT:
        assert norm(mda._scaling_data - initial_residual_vector) == 0
        assert (
            mda.residual_history[-1]
            == norm(final_residual_vector / initial_residual_vector) / 20**0.5
        )
