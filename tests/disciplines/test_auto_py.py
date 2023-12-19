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
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import re

import pytest
from numpy import array
from numpy import atleast_1d
from numpy import ndarray
from numpy import ones
from numpy import sqrt
from numpy import zeros
from scipy.optimize import rosen
from scipy.optimize import rosen_der

from gemseo import MDODiscipline
from gemseo import create_design_space
from gemseo import create_mda
from gemseo import create_scenario
from gemseo import execute_algo
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.core.parallel_execution.disc_parallel_execution import DiscParallelExecution
from gemseo.disciplines.auto_py import AutoPyDiscipline
from gemseo.disciplines.auto_py import to_arrays_dict
from gemseo.problems.sellar.sellar import get_inputs

X_DIM = 4


@pytest.fixture()
def design_space():
    design_space = create_design_space()
    design_space.add_variable(
        "x", X_DIM, l_b=-2 * ones(X_DIM), u_b=2 * ones(X_DIM), value=zeros(X_DIM)
    )
    return design_space


def f1(y2=1.0, z=2.0):
    y1 = z + y2
    return y1  # noqa: RET504


def f2(y1=2.0, z=2.0):
    y2 = z + 2 * y1
    y3 = 14
    return y2, y3


def f3(x=1.0):
    if x > 0:
        y = -x
        return y  # noqa: RET504
    y = 2 * x
    return y  # noqa: RET504


def f4(x=1.0):
    if x > 0:
        y = -x
        return y  # noqa: RET504
    y = 2 * x
    return y, x


def f5(y=2.0):
    u = 1.0 - 0.01 * y
    f = 2 * u
    return u, f


def df5(y=2.0):
    return array([[-0.01], [-0.02]])


def test_basic():
    """Test a basic auto-discipline execution."""
    d1 = AutoPyDiscipline(f1)

    assert list(d1.get_input_data_names()) == ["y2", "z"]
    d1.execute()

    assert d1.local_data["y1"] == f1()

    d2 = AutoPyDiscipline(f2)
    assert list(d2.get_input_data_names()) == ["y1", "z"]
    assert list(d2.get_output_data_names()) == ["y2", "y3"]

    d2.execute()
    assert d2.local_data["y2"] == f2()[0]


@pytest.mark.parametrize(
    "grammar_type",
    [AutoPyDiscipline.GrammarType.SIMPLE, AutoPyDiscipline.GrammarType.JSON],
)
def test_jac(grammar_type):
    """Test a basic jacobian."""
    disc = AutoPyDiscipline(py_func=f5, py_jac=df5, grammar_type=grammar_type)
    assert disc.check_jacobian()


def test_use_arrays():
    """Test the use of arrays."""
    d1 = AutoPyDiscipline(f1, use_arrays=True)
    d1.execute()
    assert d1.local_data["y1"] == f1()
    d1.execute({"x1": array([1.0]), "z": array([2.0])})
    assert d1.local_data["y1"] == f1()


def test_fail_wrongly_formatted_function():
    """Test that a wrongly formatted function cannot be used."""
    AutoPyDiscipline(f3)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Two return statements use different variable names; ['y', 'x'] and ["
            "'y']."
        ),
    ):
        AutoPyDiscipline(f4)


def test_fail_not_a_python_function():
    """Test the failure if a Python function is not provided."""
    not_a_function = 2
    with pytest.raises(TypeError, match="py_func must be callable."):
        AutoPyDiscipline(not_a_function)


def test_jac_pb(design_space):
    """Test the AutoPyDiscipline with Jacobian provided."""
    max_iter = 100
    algo = "L-BFGS-B"

    pb = OptimizationProblem(design_space)
    pb.objective = MDOFunction(rosen, "rosen", jac=rosen_der)
    execute_algo(pb, algo, max_iter=max_iter)
    fopt_ref = pb.solution.f_opt

    scn = create_scenario(
        AutoPyDiscipline(rosen, rosen_der),
        "DisciplinaryOpt",
        "r",
        design_space,
    )
    scn.execute({"algo": algo, "max_iter": max_iter})

    assert fopt_ref == scn.optimization_result.f_opt


def test_missing_jacobian():
    auto_rosen = AutoPyDiscipline(rosen)
    with pytest.raises(RuntimeError, match="The analytic Jacobian is missing."):
        auto_rosen._compute_jacobian()


@pytest.mark.parametrize("input_", [{"a": [1.0]}, {"a": array([1.0])}])
def test_to_arrays_dict(input_):
    """Test the function to_arrays_dict."""
    output = to_arrays_dict(input_)
    assert output["a"] == array([1.0])


def test_multiprocessing():
    """Test the execution of an AutoPyDiscipline in multiprocessing."""
    d1 = AutoPyDiscipline(f1)
    d2 = AutoPyDiscipline(f2)

    parallel_execution = DiscParallelExecution([d1, d2], n_processes=2)
    parallel_execution.execute([
        {"y2": array([2.0]), "z": array([1.0])},
        {"y1": array([5.0]), "z": array([3.0])},
    ])

    assert d1.local_data["y1"] == f1(2.0, 1.0)
    assert d2.local_data["y2"] == f2(5.0, 3.0)[0]


@pytest.mark.parametrize(
    ("name", "expected"), [("custom_name", "custom_name"), (None, "f1")]
)
def test_auto_py_name(name, expected):
    """Test that the name of the AutoPyDiscipline is set correctly."""
    d1 = AutoPyDiscipline(f1, name=name)
    assert d1.name == expected


def obj(a=1.0, b=2.0, c=3.0):
    c1 = a + 2.0 * b + 3.0 * c
    return c1  # noqa: RET504


def jac(a=1.0, b=2.0, c=3.0):
    return array([[1.0, 2.0, 3.0]])


def jac_wrong_shape(a=1.0, b=2.0, c=3.0):
    return array([[1.0, 2.0, 3.0]]).T


def test_jacobian_shape_mismatch():
    """Tests the jacobian shape."""
    disc = AutoPyDiscipline(py_func=obj, py_jac=jac)

    assert disc.check_jacobian(threshold=1e-5)

    disc_wrong = AutoPyDiscipline(py_func=obj, py_jac=jac_wrong_shape)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The shape (3, 1) of the Jacobian matrix of the discipline obj "
            "provided by py_jac does not match (output_size, input_size)=(1, 3)."
        ),
    ):
        disc_wrong.linearize(compute_all_jacobians=True)


def test_multiline_return():
    """Check that AutoPyDiscipline can wrap a function with a multiline return."""

    def f(x):
        yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy = x
        zzzzzzzzzzzzzzzzz = x
        return (
            yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy,
            zzzzzzzzzzzzzzzzz,
        )

    discipline = AutoPyDiscipline(f)
    assert discipline.input_names == ["x"]
    assert discipline.output_names == [
        "yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy",
        "zzzzzzzzzzzzzzzzz",
    ]


def f6(x: int) -> int:
    z = 0
    return z  # noqa: RET504


def f6_with_defaults(x: int = 0) -> int:
    z = 0
    return z  # noqa: RET504


def f6_no_return(x: int):
    z = 0
    return z  # noqa: RET504


def f6_missing_type(x: int, y):
    z = 0
    return z  # noqa: RET504


def f6_multiple_returns(x: int) -> tuple[int, float]:
    z = 0
    zz = 0
    return z, zz


def f6_missing_return_tuple(x: int) -> int:
    z = 0
    zz = 0
    return z, zz


def f6_bad_multiple_returns(x: int) -> tuple[int]:
    z = 0
    zz = 0
    return z, zz


@pytest.mark.parametrize(
    ("func", "warnings", "input_names_to_types", "output_names_to_types"),
    [
        (f6, [], {"x": int}, {"z": int}),
        (f6_with_defaults, [], {"x": int}, {"z": int}),
        (
            f6_no_return,
            [
                "Discipline f6_no_return: py_func has inconsistent type "
                "hints: either both "
                "the signature arguments and the return values shall have "
                "type hints or none. "
                "The grammars will not use the type hints at all."
            ],
            {"x": ndarray},
            {"z": ndarray},
        ),
        (
            f6_missing_type,
            [
                "Discipline f6_missing_type: py_func has missing type hints "
                "for the arguments: y.",
                "Discipline f6_missing_type: py_func has inconsistent type "
                "hints: either both "
                "the signature arguments and the return values shall have "
                "type hints or none. "
                "The grammars will not use the type hints at all.",
            ],
            {"x": ndarray, "y": ndarray},
            {"z": ndarray},
        ),
        (f6_multiple_returns, [], {"x": int}, {"z": int, "zz": float}),
        (
            f6_missing_return_tuple,
            [
                "Discipline f6_missing_return_tuple: py_func has bad return "
                "type hints: expecting "
                "a tuple of types, got <class 'int'>.",
                "Discipline f6_missing_return_tuple: py_func has inconsistent "
                "type hints: "
                "either both the signature arguments and the return values "
                "shall have type "
                "hints or none. The grammars will not use the type hints at "
                "all.",
            ],
            {"x": ndarray},
            {"z": ndarray, "zz": ndarray},
        ),
        (
            f6_bad_multiple_returns,
            [
                "Discipline f6_bad_multiple_returns: py_func has bad return "
                "type hints: the number "
                "of return values and return types shall be equal: 2 return "
                "values but 1 "
                "return type hints.",
                "Discipline f6_bad_multiple_returns: py_func has inconsistent "
                "type hints: "
                "either both the signature arguments and the return values "
                "shall have type "
                "hints or none. The grammars will not use the type hints at "
                "all.",
            ],
            {"x": ndarray},
            {"z": ndarray, "zz": ndarray},
        ),
    ],
)
def test_type_hints_for_grammars(
    func,
    warnings,
    input_names_to_types,
    output_names_to_types,
    caplog,
):
    """Verify the type hints handling."""
    d = AutoPyDiscipline(func, grammar_type=MDODiscipline.GrammarType.SIMPLE)
    assert d.input_grammar == input_names_to_types
    assert d.output_grammar == output_names_to_types
    assert caplog.messages == warnings
    if caplog.records:
        assert caplog.records[0].levelname == "WARNING"


def compute_y_1(
    x_local: ndarray,
    x_shared: ndarray,
    y_2: ndarray,
) -> float:
    """Evaluate the first coupling equation in functional form.

    Args:
        x_local: The design variables local to first discipline.
        x_shared: The shared design variables.
        y_2: The coupling variable coming from the second discipline.

    Returns:
        The value of the coupling variable :math:`y_1`.
    """
    if x_shared.ndim != 1:
        # This handles running the test suite for checking data conversion.
        x_shared = x_shared.flatten()
    y_1 = float(sqrt(x_shared[0] ** 2 + x_shared[1] + x_local[0] - 0.2 * y_2[0]))
    return y_1  # noqa: RET504


def compute_jacobian_1(
    x_local: ndarray,
    x_shared: ndarray,
    y_2: ndarray,
) -> ndarray:
    jac = ones((1, 4))
    if x_shared.ndim != 1:
        # This handles running the test suite for checking data conversion.
        x_shared = x_shared.flatten()
    inv_denom = 1.0 / compute_y_1(x_local, x_shared, y_2)
    jac[0][0] = 0.5 * inv_denom
    jac[0][1] = x_shared[0] * inv_denom
    jac[0][2] = 0.5 * inv_denom
    jac[0][-1] = -0.1 * inv_denom
    return jac


def compute_y_2(
    x_shared: ndarray,
    y_1: float,
) -> ndarray:
    """Evaluate the second coupling equation in functional form.

    Args:
        x_shared: The shared design variables.
        y_1: The coupling variable coming from the first discipline.

    Returns:
        The value of the coupling variable :math:`y_2`.
    """
    if x_shared.ndim != 1:
        # This handles running the test suite for checking data conversion.
        x_shared = x_shared.flatten()
    out = x_shared[0] + x_shared[1]
    y_2 = array([y_1 + out])
    return y_2  # noqa: RET504


def compute_jacobian_2(
    x_shared: ndarray,
    y_1: float,
) -> ndarray:
    return ones((1, 3))


@pytest.mark.parametrize("x_local", range(3))
@pytest.mark.parametrize(
    "mda_name", ["MDAGaussSeidel", "MDAJacobi", "MDANewtonRaphson", "MDAQuasiNewton"]
)
def test_mda(x_local, mda_name, sellar_disciplines):
    """Verify MDA."""
    input_data_ref = get_inputs()
    mda_ref = create_mda(mda_name, sellar_disciplines[:-1])
    output_ref = mda_ref.execute(input_data_ref)

    input_data = input_data_ref.copy()
    for name, value in input_data.items():
        input_data[name] = atleast_1d(value)
    input_data["y_1"] = 1.0

    sellar_1 = AutoPyDiscipline(compute_y_1, py_jac=compute_jacobian_1)
    sellar_2 = AutoPyDiscipline(compute_y_2, py_jac=compute_jacobian_2)
    mda = create_mda(mda_name, [sellar_1, sellar_2])
    outputs = mda.execute(input_data)

    assert outputs["y_1"] == output_ref["y_1"]
    assert outputs["y_2"] == output_ref["y_2"]

    if mda_name != "MDAGaussSeidel":
        # No need to test more of the check jacobian.
        return

    del input_data["y_1"]
    mda.default_inputs = input_data

    assert mda.check_jacobian(
        input_data=input_data, inputs=["x_local", "x_shared"], outputs=["y_1", "y_2"]
    )
