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
"""The discipline of the wing weight problem."""

from __future__ import annotations

from numpy import array

from gemseo.disciplines.analytic import AnalyticDiscipline


class WingWeightDiscipline(AnalyticDiscipline):
    """The discipline of the wing weight problem."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__(
            {
                "Ww": "0.036 * Sw ** 0.758 "
                "* Wfw ** 0.0035 "
                "* (A / cos(pi / 180 * Lamda) ** 2) ** 0.6 "
                "* q ** 0.006 "
                "* ell ** 0.04 "
                "* (100 * tc / cos(pi / 180 * Lamda)) ** (-0.3) "
                "* (Nz * Wdg) ** 0.49 + Sw * Wp"
            },
        )
        self.io.input_grammar.defaults.update({
            "A": array([8]),
            "Lamda": array([0.0]),
            "Nz": array([4.25]),
            "Sw": array([175]),
            "Wdg": array([2100]),
            "Wfw": array([260]),
            "Wp": array([0.0525]),
            "ell": array([0.75]),
            "q": array([14.5]),
            "tc": array([0.13]),
        })

        self.io.input_grammar.descriptions.update({
            "A": "The aspect ratio (-).",
            "Lamda": "The quarter-chord sweep angle (deg).",
            "Nz": "The ultimate load factor (-).",
            "Sw": "The wing area (ft²).",
            "Wdg": "The flight design gross weight (lb).",
            "Wfw": "The weight of fuel in the wing (lb).",
            "Wp": "The paint weight (lb/ft²).",
            "ell": "The taper ratio (-).",
            "q": "The dynamic pressure at cruise (lb/ft²).",
            "tc": "The airfoil thickness to chord ratio (-).",
        })

        self.io.output_grammar.descriptions.update({"Ww": "The wing weight (lb)."})
