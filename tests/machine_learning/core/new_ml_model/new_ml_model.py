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

from gemseo.machine_learning.core.models.ml_model import BaseMLModel
from gemseo.machine_learning.core.models.ml_model_settings import BaseMLModelSettings


class NewMLModel(BaseMLModel):
    """New machine learning model class."""

    LIBRARY = "NewLibrary"
    settings_class = BaseMLModelSettings

    def learn(self, samples=()) -> None:
        super().learn(samples=samples)
        self._trained = True

    def _learn(self, indices, fit_transformers) -> None:
        pass
