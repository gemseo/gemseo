# -*- coding: utf-8 -*-
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
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest
from copy import deepcopy
from math import cos, exp, log10, sin

import pytest
from future import standard_library
from numpy import array, zeros
from numpy.linalg import norm
from scipy.optimize import rosen, rosen_der

from gemseo import LOGGER, SOFTWARE_NAME
from gemseo.api import configure_logger, create_discipline
from gemseo.core.analytic_discipline import AnalyticDiscipline
from gemseo.problems.sobieski.wrappers import SobieskiMission, SobieskiStructure
from gemseo.utils.data_conversion import DataConversion
from gemseo.utils.derivatives_approx import (
    ComplexStep,
    DisciplineJacApprox,
    FirstOrderFD,
    comp_best_step,
)

standard_library.install_aliases()


configure_logger(SOFTWARE_NAME)


def test_initFirstOrderFD():
    """ """
    FirstOrderFD(rosen)


def test_initComplexStep():
    """ """
    cplx = ComplexStep(rosen, 1e-30j)
    assert cplx.step == 1e-30

    grad = cplx.f_gradient(zeros(3))
    assert norm(grad - rosen_der(zeros(3))) < 1e-3

    with pytest.raises(ValueError):
        cplx.f_gradient(zeros(3) + 1j)


def get_x_tests():
    """ """
    return [
        [0.0, 0.0],
        [1.0, 3.0, 5.0],
        [-1.9, 3.7, 4.0, 7, -1.9, 3.7, 4.0, 7],
        [-1.0, 5.0],
    ]


def run_tests(xs, fd_app):
    """

    :param xs: param fd_app:
    :param fd_app:

    """
    for x in xs:
        xa = array(x)
        appeox = fd_app.f_gradient(xa)
        exact = rosen_der(xa)
        err = norm(appeox - exact) / norm(exact)
        assert err < 1e-4


def test_approx_FirstOrderFD():
    run_tests(get_x_tests(), FirstOrderFD(rosen, 1e-8))


def test_approx_ComplexStep():
    run_tests(get_x_tests(), ComplexStep(rosen))


def test_approx_ComplexStep_diff_steps_e60():
    run_tests(get_x_tests(), ComplexStep(rosen, 1e-60))


def test_approx_ComplexStep_diff_steps_e200():
    run_tests(get_x_tests(), ComplexStep(rosen, 1e-200))


def test_approx_ComplexStep_diff_steps_e30():
    run_tests(get_x_tests(), ComplexStep(rosen, 1e-30))


def test_abs_der():
    discipline = AnalyticDiscipline("name", {"y": "x", "z": "x"})
    discipline.execute()
    apprx = DisciplineJacApprox(discipline)
    apprx.compute_approx_jac(["z"], ["x"])

    discipline.linearize()
    discipline.jac["z"]["x"] = array([[2.0]])

    assert not apprx.check_jacobian(discipline.jac, ["z"], ["x"], discipline)

    discipline.linearize()
    discipline.jac["z"]["x"] = array([[2.0, 3.0]])

    assert not apprx.check_jacobian(discipline.jac, ["z"], ["x"], discipline)


def test_complex_fail():
    discipline = SobieskiMission("complex128")
    assert discipline.check_jacobian(derr_approx=discipline.COMPLEX_STEP)

    data = deepcopy(discipline.default_inputs)
    data["x_shared"] += 0.1j
    with pytest.raises(ValueError):
        discipline.check_jacobian(data, derr_approx=discipline.COMPLEX_STEP)


@pytest.mark.parametrize("discipline_name", ["Sellar1", "Sellar2"])
def test_auto_step(discipline_name):
    discipline = create_discipline(discipline_name)

    ok = discipline.check_jacobian(auto_set_step=True, threshold=1e-2, step=1e-7)
    assert ok


def test_opt_step():
    x = 0.1
    step = 1e-6
    funcs = [sin, cos, exp]
    jacs = [cos, lambda x: -sin(x), exp]

    for func, jac in zip(funcs, jacs):
        for mult in [1.0, 1e2, 1e-2]:
            for x in [0.0, 1.0, 3.0]:

                f_p = func(mult * (x + step))
                f_x = func(mult * (x))
                f_m = func(mult * (x - step))
                trunc_error, cancel_error, opt_step = comp_best_step(
                    f_p, f_x, f_m, step
                )
                if trunc_error is None:
                    continue

                df_app = (func(mult * (x + opt_step)) - f_x) / opt_step
                err = abs(df_app - mult * jac(mult * x))
                full_error = trunc_error + cancel_error
                rel_erro_on_err = abs(log10(abs(full_error)) - log10(abs(err))) < 5
                assert rel_erro_on_err
