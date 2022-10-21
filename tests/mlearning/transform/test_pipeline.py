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
#        :author: Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Test transformer pipeline module."""
from __future__ import annotations

import pytest
from gemseo.mlearning.transform.pipeline import Pipeline
from gemseo.mlearning.transform.scaler.scaler import Scaler
from gemseo.mlearning.transform.transformer import Transformer
from numpy import allclose
from numpy import arange
from numpy import array
from numpy import diag
from numpy import eye
from numpy import ndarray


@pytest.fixture
def data() -> ndarray:
    """Test data."""
    return arange(30).reshape((10, 3))


@pytest.fixture
def transformers() -> list[Transformer]:
    """Transformers for pipeline."""
    return [Scaler(coefficient=2), Scaler(offset=3), Scaler(coefficient=5)]


C_1 = array([2, 1, 1])
OFF_2 = 3
OFF_3 = array([0, 10, 100])
C_3 = array([5, 1, 2])


@pytest.fixture
def other_transformers():
    """Transformers for pipeline."""
    return [
        Scaler(coefficient=C_1),
        Scaler(offset=OFF_2),
        Scaler(offset=OFF_3, coefficient=C_3),
    ]


def test_constructor(transformers, other_transformers):
    """Test constructor."""
    pipeline = Pipeline()
    assert pipeline.name == "Pipeline"
    assert pipeline.transformers == []
    another_pipeline = Pipeline(transformers=transformers)
    assert another_pipeline.transformers == transformers
    yet_another_pipeline = Pipeline(transformers=other_transformers)
    assert yet_another_pipeline.transformers == other_transformers


def test_duplicate(data, transformers):
    """Test pipeline duplicate method."""
    pipeline = Pipeline()
    pipeline.fit(data)
    pipeline_dup = pipeline.duplicate()
    for transformer, transformer_dup in zip(
        pipeline.transformers, pipeline_dup.transformers
    ):
        assert transformer != transformer_dup

    pipeline = Pipeline(transformers=transformers)
    pipeline.fit(data)
    pipeline_dup = pipeline.duplicate()
    for transformer, transformer_dup in zip(
        pipeline.transformers, pipeline_dup.transformers
    ):
        assert transformer != transformer_dup


def test_fit(data, transformers, other_transformers):
    """Test fit method."""
    pipeline = Pipeline()
    pipeline.fit(data)

    another_pipeline = Pipeline(transformers=transformers)
    another_pipeline.fit(data)

    yet_another_pipeline = Pipeline(transformers=other_transformers)
    yet_another_pipeline.fit(data)


def test_transform(data, transformers, other_transformers):
    """Test transform method."""
    pipeline = Pipeline()
    pipeline.fit(data)
    transformed_data = pipeline.transform(data)
    assert allclose(transformed_data, data)

    another_pipeline = Pipeline(transformers=transformers)
    another_pipeline.fit(data)
    transformed_data = another_pipeline.transform(data)
    assert allclose(transformed_data, 5 * (3 + 2 * data))

    yet_another_pipeline = Pipeline(transformers=other_transformers)
    yet_another_pipeline.fit(data)
    transformed_data = yet_another_pipeline.transform(data)
    assert allclose(transformed_data, OFF_3 + C_3 * (OFF_2 + C_1 * data))


def test_inverse_transform(data, transformers, other_transformers):
    """Test inverse_transform method."""
    pipeline = Pipeline()
    pipeline.fit(data)
    transformed_data = pipeline.inverse_transform(data)
    assert allclose(transformed_data, data)

    another_pipeline = Pipeline(transformers=transformers)
    another_pipeline.fit(data)
    transformed_data = another_pipeline.inverse_transform(data)
    assert allclose(transformed_data, (data / 5 - 3) / 2)

    yet_another_pipeline = Pipeline(transformers=other_transformers)
    yet_another_pipeline.fit(data)
    transformed_data = yet_another_pipeline.inverse_transform(data)
    assert allclose(transformed_data, ((data - OFF_3) / C_3 - OFF_2) / C_1)


def test_compute_jacobian(data, transformers, other_transformers):
    """Test compute_jacobian method."""
    iden = eye(data.shape[1])

    pipeline = Pipeline()
    pipeline.fit(data)
    jacobian = pipeline.compute_jacobian(data)
    assert allclose(jacobian, iden)

    another_pipeline = Pipeline(transformers=transformers)
    another_pipeline.fit(data)
    jacobian = another_pipeline.compute_jacobian(data)
    assert allclose(jacobian, 5 * 2 * iden)

    yet_another_pipeline = Pipeline(transformers=other_transformers)
    yet_another_pipeline.fit(data)
    jacobian = yet_another_pipeline.compute_jacobian(data)
    assert allclose(jacobian, diag(C_3 * C_1))


def test_compute_jacobian_inverse(data, transformers, other_transformers):
    """Test compute_jacobian_inverse method."""
    iden = eye(data.shape[1])

    pipeline = Pipeline()
    pipeline.fit(data)
    inv_jacobian = pipeline.compute_jacobian_inverse(data)
    assert allclose(inv_jacobian, iden)

    another_pipeline = Pipeline(transformers=transformers)
    another_pipeline.fit(data)
    inv_jacobian = another_pipeline.compute_jacobian_inverse(data)
    assert allclose(inv_jacobian, 1 / 2 * 1 / 5 * iden)

    yet_another_pipeline = Pipeline(transformers=other_transformers)
    yet_another_pipeline.fit(data)
    inv_jacobian = yet_another_pipeline.compute_jacobian_inverse(data)
    assert allclose(inv_jacobian, diag(1 / C_1 * 1 / C_3))
