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
"""A squarified treemap of the variables of a dataset."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from numpy import isfinite
from numpy import issubdtype
from numpy import number

from gemseo.post.dataset.base import BaseDatasetPlot
from gemseo.post.dataset.treemap_settings import Treemap_Settings
from gemseo.util.matplotlib_figure import _get_colors
from gemseo.util.string import pretty_repr

if TYPE_CHECKING:
    from gemseo.post.dataset.treemap_settings import _Color


class Treemap(BaseDatasetPlot[Treemap_Settings]):
    """A squarified treemap of the variables of a dataset.

    The dataset must have a single sample
    whose values are non-negative.
    The treemap fills a unit square
    with one tile per scalar variable,
    the area of a tile being the share of the value of the variable
    in the sum of all the values;
    the values do not need to sum to one.
    The value written on a tile is the raw value of the variable.
    The tiles are laid out so that they are as close to squares as possible.

    The tiles are ordered from the largest to the smallest one,
    the residual tile coming last.
    """

    settings_class = Treemap_Settings

    def _create_specific_data_from_dataset(
        self,
    ) -> tuple[
        tuple[str, ...],
        tuple[float, ...],
        tuple[tuple[_Color, ...], ...],
    ]:
        """
        Returns:
            The labels of the tiles,
            the values to be written on the tiles
            and the colors of the tiles, with one or several colors per tile.

        Raises:
            ValueError: If the dataset has more than one sample,
                if two columns have the same name
                (e.g. a variable in two groups),
                if a value is not numeric,
                if a value is not finite or negative,
                if the sum of the values is zero
                or if `residual_name` is not a column of the dataset.
        """  # noqa: D205, D212, D415
        settings = self.settings
        data = self.dataset.get_view().to_numpy()
        if len(data) != 1:
            msg = (
                "Treemap requires a dataset with a single sample; "
                f"got {len(data)} samples."
            )
            raise ValueError(msg)

        names = list(self.dataset.get_columns())
        duplicated_names = [name for name, count in Counter(names).items() if count > 1]
        if duplicated_names:
            msg = (
                "Treemap requires distinct column names; "
                f"got duplicates: {pretty_repr(duplicated_names)}."
            )
            raise ValueError(msg)

        raw_values = data[0]
        if not issubdtype(raw_values.dtype, number):
            msg = "Treemap requires numeric values."
            raise ValueError(msg)

        if not isfinite(raw_values).all():
            msg = "Treemap requires finite values."
            raise ValueError(msg)

        if (raw_values < 0).any():
            msg = "Treemap requires non-negative values."
            raise ValueError(msg)

        total = raw_values.sum()
        if total == 0:
            msg = "Treemap requires at least one positive value."
            raise ValueError(msg)

        residual_name = settings.residual_name
        if residual_name and residual_name not in names:
            msg = f"The residual tile {residual_name!r} is not a column of the dataset."
            raise ValueError(msg)

        name_to_color = {
            name: (color,)
            for name, color in zip(
                names, _get_colors(settings.colormap, len(names)), strict=True
            )
        }
        # The residual tile is not colored from the colormap.
        name_to_color[residual_name] = (settings.residual_color,)
        name_to_color.update(settings.tile_colors)
        name_to_value = dict(zip(names, map(float, raw_values), strict=True))

        kept_names = []
        merged_names = []
        for name in names:
            if name == residual_name:
                continue

            if name_to_value[name] / total < settings.min_area:
                merged_names.append(name)
            else:
                kept_names.append(name)

        kept_names.sort(key=lambda name: -name_to_value[name])
        values = [name_to_value[name] for name in kept_names]
        colors = [name_to_color[name] for name in kept_names]
        labels = [settings.labels.get(name, name) for name in kept_names]
        if merged_names or residual_name:
            if residual_name:
                residual_label = settings.labels.get(residual_name, residual_name)
                suffix = f" + {settings.residual_label}"
                if merged_names and not residual_label.endswith(suffix):
                    residual_label = f"{residual_label}{suffix}"

                merged_names.append(residual_name)
            else:
                residual_label = settings.residual_label

            labels.append(residual_label)
            values.append(sum(name_to_value[name] for name in merged_names))
            colors.append(name_to_color[residual_name])

        return (
            tuple(labels),
            tuple(values),
            tuple(colors),
        )
