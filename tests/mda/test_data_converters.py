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

# import numpy as np
# from gemseo.mda.gauss_seidel import MDAGaussSeidel
# from gemseo.problems.sellar.sellar import Y_1
# from gemseo.problems.sellar.sellar import Y_2
#
# from .sellar_for_data_converter import Sellar1
# from .sellar_for_data_converter import Sellar2
# from .sellar_for_data_converter import SellarSystem
# from .sellar_for_data_converter import ToXConverter
#
#
# def test_mda_gauss_seidel_jac(input_data):
#     """Test linearization of GS MDA."""
#     converter = ToXConverter()
#     discipline1 = Sellar1()
#     discipline2 = Sellar2()
#     system = SellarSystem()
#     input_data[Y_1] = np.ones([1])
#     input_data[Y_2] = np.ones([1])
#
#     disciplines = [converter, discipline1, discipline2, system]
#     mda = MDAGaussSeidel(
#         disciplines,
#         max_mda_iter=100,
#         tolerance=1e-14,
#         over_relax_factor=0.99,
#         grammar_type=MDAGaussSeidel.GrammarType.PYDANTIC,
#     )
#     input_data = mda.execute(input_data)
#     for discipline in disciplines:
#         assert discipline.check_jacobian(
#             input_data, derr_approx="complex_step", step=1e-30
#         )
#
#     assert mda.check_jacobian(
#         input_data,
#         threshold=1e-4,
#         derr_approx="complex_step",
#         step=1e-30,
#     )
