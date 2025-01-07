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
from __future__ import annotations

import re
from copy import deepcopy
from math import cos
from math import exp
from math import isinf
from math import log10
from math import sin
from typing import TYPE_CHECKING

import pytest
from numpy import array
from numpy import inf
from numpy import ndarray
from numpy import zeros
from numpy.linalg import norm
from numpy.testing import assert_equal
from scipy.optimize import rosen
from scipy.optimize import rosen_der

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt.factory import OptimizationLibraryFactory
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.core.discipline import Discipline
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.problems.mdo.scalable.linear.linear_discipline import LinearDiscipline
from gemseo.problems.mdo.sobieski.disciplines import SobieskiMission
from gemseo.utils.derivatives.approximation_modes import ApproximationMode
from gemseo.utils.derivatives.centered_differences import CenteredDifferences
from gemseo.utils.derivatives.complex_step import ComplexStep
from gemseo.utils.derivatives.derivatives_approx import DisciplineJacApprox
from gemseo.utils.derivatives.error_estimators import compute_best_step
from gemseo.utils.derivatives.factory import GradientApproximatorFactory
from gemseo.utils.derivatives.finite_differences import FirstOrderFD

if TYPE_CHECKING:
    from gemseo.typing import StrKeyMapping


def test_init_first_order_fd() -> None:
    """"""
    FirstOrderFD(rosen)


def test_init_centred_differences() -> None:
    """"""
    CenteredDifferences(rosen)


def test_init_complex_step() -> None:
    """"""
    cplx = ComplexStep(rosen, 1e-30j)
    assert cplx.step == 1e-30

    grad = cplx.f_gradient(zeros(3))
    assert norm(grad - rosen_der(zeros(3))) < 1e-3

    with pytest.raises(ValueError):
        cplx.f_gradient(zeros(3) + 1j)


@pytest.fixture
def x():
    """"""
    return [
        [0.0, 0.0],
        [1.0, 3.0, 5.0],
        [-1.9, 3.7, 4.0, 7, -1.9, 3.7, 4.0, 7],
        [-1.0, 5.0],
    ]


def run_tests(xs, fd_app) -> None:
    """:param xs: param fd_app:
    :param fd_app:

    """
    for x in xs:
        xa = array(x)
        appeox = fd_app.f_gradient(xa)
        exact = rosen_der(xa)
        err = norm(appeox - exact) / norm(exact)
        assert err < 1e-4


def test_approx_first_order_fd(x) -> None:
    run_tests(x, FirstOrderFD(rosen, 1e-8))


def test_approx_centred_differences(x) -> None:
    run_tests(x, CenteredDifferences(rosen, 1e-8))


def test_approx_complex_step(x) -> None:
    run_tests(x, ComplexStep(rosen))


def test_approx_complex_step_diff_steps_e60(x) -> None:
    run_tests(x, ComplexStep(rosen, 1e-60))


def test_approx_complex_step_diff_steps_e200(x) -> None:
    run_tests(x, ComplexStep(rosen, 1e-200))


def test_approx_complex_step_diff_steps_e30(x) -> None:
    run_tests(x, ComplexStep(rosen, 1e-30))


def test_abs_der() -> None:
    discipline = AnalyticDiscipline({"y": "x", "z": "x"})
    discipline.execute()
    apprx = DisciplineJacApprox(discipline)
    apprx.compute_approx_jac(["z"], ["x"])

    discipline.linearize()
    discipline.jac["z"]["x"] = array([[2.0]])

    assert not apprx.check_jacobian(["z"], ["x"])

    discipline.linearize()
    discipline.jac["z"]["x"] = array([[2.0, 3.0]])

    assert not apprx.check_jacobian(["z"], ["x"])


def test_cache_is_none() -> None:
    """Check that compute_approx_jac works when cache is None"""
    discipline = AnalyticDiscipline({"y": "x", "z": "x"})
    discipline.set_cache(discipline.CacheType.NONE)
    discipline.execute()
    apprx = DisciplineJacApprox(discipline)
    apprx.compute_approx_jac(["z"], ["x"])

    discipline.linearize()


def test_complex_fail() -> None:
    discipline = SobieskiMission("complex128")
    assert discipline.check_jacobian(
        derr_approx=discipline.ApproximationMode.COMPLEX_STEP
    )

    data = deepcopy(discipline.io.input_grammar.defaults)
    data["x_shared"] += 0.1j
    with pytest.raises(ValueError):
        discipline.check_jacobian(
            data, derr_approx=discipline.ApproximationMode.COMPLEX_STEP
        )


@pytest.mark.parametrize("parallel", [True, False])
@pytest.mark.parametrize(
    "method",
    [ApproximationMode.FINITE_DIFFERENCES, ApproximationMode.CENTERED_DIFFERENCES],
)
@pytest.mark.integration
def test_auto_step(parallel, method, sellar_disciplines) -> None:
    for discipline in sellar_disciplines:
        assert discipline.check_jacobian(
            auto_set_step=True,
            threshold=1e-2,
            step=1e-7,
            parallel=parallel,
            derr_approx=method,
        )


def test_opt_step() -> None:
    x = 0.1
    step = 1e-6
    funcs = [sin, cos, exp]
    jacs = [cos, lambda x: -sin(x), exp]

    for func, jac in zip(funcs, jacs):
        for mult in [1.0, 1e2, 1e-2]:
            for x in [0.0, 1.0, 3.0]:
                f_p = func(mult * (x + step))
                f_x = func(mult * x)
                f_m = func(mult * (x - step))
                trunc_error, cancel_error, opt_step = compute_best_step(
                    f_p, f_x, f_m, step
                )
                if trunc_error is None:
                    continue

                df_app = (func(mult * (x + opt_step)) - f_x) / opt_step
                err = abs(df_app - mult * jac(mult * x))
                full_error = trunc_error + cancel_error
                rel_erro_on_err = abs(log10(abs(full_error)) - log10(abs(err))) < 5
                assert rel_erro_on_err


@pytest.mark.parametrize(
    ("indices", "expected_sequence", "expected_variable_indices"),
    [
        ({"y": None}, [0, 1, 2, 3, 4], {"x": [0, 1], "y": [0, 1, 2]}),
        ({"y": Ellipsis}, [0, 1, 2, 3, 4], {"x": [0, 1], "y": [0, 1, 2]}),
        ({"y": 1}, [0, 1, 3], {"x": [0, 1], "y": [1]}),
        ({"y": [2, 4]}, [0, 1, 4, 6], {"x": [0, 1], "y": [2, 4]}),
        ({"y": slice(0, 2)}, [0, 1, 2, 3], {"x": [0, 1], "y": [0, 1]}),
        ({}, [0, 1, 2, 3, 4], {"x": [0, 1], "y": [0, 1, 2]}),
    ],
)
def test_compute_io_indices(
    indices, expected_sequence, expected_variable_indices
) -> None:
    """Check that input and output indices are correctly computed from indices."""
    (
        indices_sequence,
        variable_indices,
    ) = DisciplineJacApprox._compute_variable_indices(
        indices, ["x", "y"], {"y": 3, "x": 2}
    )
    assert indices_sequence == expected_sequence
    assert variable_indices == expected_variable_indices


@pytest.mark.parametrize(
    "method",
    [ApproximationMode.FINITE_DIFFERENCES, ApproximationMode.CENTERED_DIFFERENCES],
)
def test_load_and_dump(tmp_wd, method) -> None:
    """Check the loading and dumping of a reference Jacobian."""
    discipline = AnalyticDiscipline({"y": "x", "z": "x"})
    discipline.execute()
    apprx = DisciplineJacApprox(discipline, approx_method=method)
    apprx.compute_approx_jac(["z"], ["x"])
    discipline.linearize()
    discipline.jac["z"]["x"] = array([[2.0]])
    file_name = "reference_jacobian.pkl"
    assert not apprx.check_jacobian(
        ["z"],
        ["x"],
        reference_jacobian_path=file_name,
        save_reference_jacobian=True,
    )

    assert not apprx.check_jacobian(
        ["z"],
        ["x"],
        reference_jacobian_path=file_name,
    )


class ToyDiscipline(Discipline):
    default_grammar_type = Discipline.GrammarType.SIMPLE

    def __init__(self, dtype) -> None:
        super().__init__()
        self.io.input_grammar.update_from_types({"x1": dtype, "x2": ndarray})
        self.io.output_grammar.update_from_types({"y1": dtype, "y2": ndarray})
        self.io.input_grammar.defaults = {
            "x1": dtype(1.0),
            "x2": array([1.0, 1.0], dtype=dtype),
        }
        self.dtype = dtype

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        self.io.data["y1"] = self.io.data["x1"] + 2 * self.io.data["x2"][0]
        self.io.data["y2"] = array([
            self.io.data["x1"] + 2 * self.io.data["x2"][0] + 3 * self.io.data["x2"][1],
            2 * self.io.data["x1"]
            + 4 * self.io.data["x2"][0]
            + 6 * self.io.data["x2"][1],
        ])

    def _compute_jacobian(self, input_names=(), output_names=()) -> None:
        self.jac = {
            "y1": {
                "x1": array([[1.0]], dtype=self.dtype),
                "x2": array([[2.0, 0.0]], dtype=self.dtype),
            },
            "y2": {
                "x1": array([[1.0], [2.0]], dtype=self.dtype),
                "x2": array([[2.0, 3.0], [4.0, 6.0]], dtype=self.dtype),
            },
        }


@pytest.mark.parametrize("inputs", [["x1"], ["x2"], ["x1", "x2"]])
@pytest.mark.parametrize("outputs", [["y1"], ["y2"], ["y1", "y2"]])
@pytest.mark.parametrize(
    "indices",
    [
        None,
        {"x1": 0},
        {"y2": 1},
        {"x1": 0, "y2": 1},
        {"x2": [0, 1], "y2": [0, 1]},
        {"x2": 1, "y2": [0, 1]},
    ],
)
@pytest.mark.parametrize("dtype", [float, complex])
@pytest.mark.parametrize(
    "method",
    [ApproximationMode.FINITE_DIFFERENCES, ApproximationMode.CENTERED_DIFFERENCES],
)
def test_indices(inputs, outputs, indices, dtype, method) -> None:
    """Test the option to check the Jacobian by indices.

    Args:
        inputs: The input variables to be checked.
        outputs: The output variables to be checked.
        dtype: The data type of the variables for the test discipline.
    """
    discipline = ToyDiscipline(dtype)
    discipline.linearize(compute_all_jacobians=True)
    apprx = DisciplineJacApprox(discipline, approx_method=method)
    assert apprx.check_jacobian(outputs, inputs, indices=indices)


@pytest.mark.parametrize("dtype", [float, complex])
@pytest.mark.parametrize(
    "method",
    [ApproximationMode.FINITE_DIFFERENCES, ApproximationMode.CENTERED_DIFFERENCES],
)
def test_wrong_step(dtype, method) -> None:
    """Test that an exception is raised if the step size length does not match inputs.

    Args:
        dtype: The data type of the variables for the test discipline.
    """
    discipline = ToyDiscipline(dtype)
    discipline.linearize(compute_all_jacobians=True)
    apprx = DisciplineJacApprox(discipline, step=[1e-7, 1e-7], approx_method=method)
    with pytest.raises(
        ValueError, match=re.escape("Inconsistent step size, expected 3 got 2.")
    ):
        apprx.compute_approx_jac(output_names=["y1", "y2"], input_names=["x1", "x2"])


def test_factory() -> None:
    factory = GradientApproximatorFactory()
    assert factory.is_available("ComplexStep")
    assert factory.is_available("CenteredDifferences")

    def function(x):
        return 2 * x

    assert isinstance(factory.create("FirstOrderFD", function), FirstOrderFD)
    assert isinstance(factory.create("finite_differences", function), FirstOrderFD)
    assert isinstance(
        factory.create("finite_differences", function, step=1e-3), FirstOrderFD
    )
    assert isinstance(factory.create("complex_step", function), ComplexStep)
    assert isinstance(
        factory.create("centered_differences", function), CenteredDifferences
    )


@pytest.mark.parametrize(
    ("normalize", "lower_bound", "upper_bound"),
    [
        (False, -2, 2),
        (True, -2, 2),
        (False, -2, inf),
        (True, -2, inf),
        (False, -inf, 2),
        (True, -inf, 2),
    ],
)
@pytest.mark.parametrize(
    "method",
    [ApproximationMode.FINITE_DIFFERENCES, ApproximationMode.CENTERED_DIFFERENCES],
)
def test_derivatives_on_design_boundaries(
    caplog, normalize, lower_bound, upper_bound, method
) -> None:
    """Check that finite differences on the design boundaries use a backward step."""
    design_space = DesignSpace()
    design_space.add_variable(
        "x", lower_bound=lower_bound, upper_bound=upper_bound, value=2.0
    )

    problem = OptimizationProblem(design_space, differentiation_method=method)
    problem.objective = MDOFunction(lambda x: x**2, "my_objective")

    OptimizationLibraryFactory().execute(
        problem, algo_name="SLSQP", max_iter=1, normalize_design_space=normalize
    )

    grad = problem.database.get_gradient_history("my_objective")[0, 0]
    if isinf(upper_bound) and (method != ApproximationMode.CENTERED_DIFFERENCES):
        assert grad > 4.0
    else:
        assert grad < 4.0

    assert grad == pytest.approx(4.0, abs=1e-6)

    assert "All components of the normalized vector " not in caplog.text


@pytest.mark.parametrize("output_size", [1, 10])
def test_derivatives_with_sparse_jacobians(tmp_wd, output_size) -> None:
    """Test check Jacobians with sparse Jacobians."""
    discipline = LinearDiscipline(
        "A", ["x"], ["y"], inputs_size=10, outputs_size=output_size, matrix_format="csr"
    )
    discipline.linearize(compute_all_jacobians=True)

    assert DisciplineJacApprox(discipline).check_jacobian(
        ["y"], ["x"], plot_result=True
    )


def f(x):
    return array([sum(x)])


@pytest.mark.parametrize("use_design_space", [False, True])
@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("cls", [CenteredDifferences, FirstOrderFD, ComplexStep])
def test_f_gradient(use_design_space, parallel, cls):
    """Check the BaseGradientApproximator.f_gradient method."""
    if use_design_space:
        design_space = DesignSpace()
        design_space.add_variable("x", size=2)
    else:
        design_space = None

    cd = cls(f, design_space=design_space, parallel=parallel)
    assert_equal(cd.f_gradient(array([0.0, 0.0])), array([[1.0, 1.0]]))
