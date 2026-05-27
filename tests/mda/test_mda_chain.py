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
#
# Copyright 2024 Capgemini
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from numpy import array
from numpy import inf
from numpy import isclose

from gemseo.algos.linear_solvers.scipy_linalg import LGMRES_Settings
from gemseo.algos.linear_solvers.scipy_linalg import TFQMR_Settings
from gemseo.core.chains.parallel_chain import ParallelDisciplineChain
from gemseo.core.coupling_structure import CouplingStructure
from gemseo.core.derivatives.jacobian_assembly import JacobianAssembly
from gemseo.core.discipline import Discipline
from gemseo.core.grammars.errors import InvalidDataError
from gemseo.core.grammars.simple import SimpleGrammar
from gemseo.mda.chain import MDAChain
from gemseo.mda.chain_settings import MDAChain_Settings
from gemseo.mda.factory import MDA_FACTORY
from gemseo.mda.gauss_seidel_settings import MDAGaussSeidel_Settings
from gemseo.mda.jacobi_settings import MDAJacobi_Settings
from gemseo.mda.newton_raphson_settings import MDANewtonRaphson_Settings
from gemseo.problems.mdo.scalable.linear.disciplines_generator import (
    create_disciplines_from_desc,
)
from gemseo.problems.mdo.scalable.linear.linear_discipline import LinearDiscipline
from gemseo.problems.mdo.sellar.sellar_1 import Sellar1
from gemseo.problems.mdo.sellar.utils import get_initial_data
from gemseo.problems.mdo.sobieski.disciplines import SobieskiPropulsion
from gemseo.utils.testing.helpers import assert_exception

from .test_mda import analytic_disciplines_from_desc
from .utils import generate_parallel_doe

if TYPE_CHECKING:
    from collections.abc import Sequence

DISC_DESCR_16D = [
    ("A", ["a"], ["b"]),
    ("B", ["c"], ["a", "n"]),
    ("C", ["b", "d"], ["c", "e"]),
    ("D", ["f"], ["d", "g"]),
    ("E", ["e"], ["f", "h", "o"]),
    ("F", ["g", "j"], ["i"]),
    ("G", ["i", "h"], ["k", "l"]),
    ("H", ["k", "m"], ["j"]),
    ("I", ["l"], ["m", "w"]),
    ("J", ["n", "o"], ["p", "q"]),
    ("K", ["y"], ["x"]),
    ("L", ["w", "x"], ["y", "z"]),
    ("M", ["p", "s"], ["r"]),
    ("N", ["r"], ["t", "u"]),
    ("O", ["q", "t"], ["s", "v"]),
    ("P", ["u", "v", "z"], ["xx"]),
]


@pytest.fixture
def coupled_disciplines() -> Sequence[Discipline]:
    """A set of 16 coupled disciplines."""
    return create_disciplines_from_desc(DISC_DESCR_16D)


def test_set_solver(sellar_with_2d_array, sellar_disciplines) -> None:
    """Test that the MDA tolerances can be set at the object instantiation."""
    mda_chain = MDAChain(
        sellar_disciplines,
        settings=MDAChain_Settings(
            tolerance=1e-3,
            linear_solver_settings=None,
        ),
    )
    assert mda_chain.settings.linear_solver_settings is None

    sub_mda1_settings = mda_chain.discipline_chain.disciplines[0].settings
    assert sub_mda1_settings.linear_solver_settings is None


def test_sellar(tmp_wd, sellar_with_2d_array, sellar_disciplines) -> None:
    """"""
    mda_chain = MDAChain(
        sellar_disciplines,
        settings=MDAChain_Settings(
            inner_mda_settings=MDAJacobi_Settings(), tolerance=1e-12
        ),
    )
    input_data = get_initial_data()
    inputs = ["x_1", "x_shared"]
    outputs = ["obj", "c_1", "c_2"]
    assert mda_chain.check_jacobian(
        input_data,
        derr_approx=Discipline.ApproximationMode.COMPLEX_STEP,
        input_names=inputs,
        output_names=outputs,
        threshold=1e-5,
    )
    mda_chain.plot_residual_history(filename="mda_chain_residuals")
    res_file = "MDAJacobi_mda_chain_residuals.png"
    assert Path(res_file).exists()


def test_sellar_chain_linearize(sellar_with_2d_array, sellar_disciplines) -> None:
    inputs = ["x_1", "x_shared"]
    outputs = ["obj", "c_1", "c_2"]
    mda_chain = MDAChain(
        sellar_disciplines,
        settings=MDAChain_Settings(
            tolerance=1e-13, max_mda_iter=30, chain_linearize=True, warm_start=True
        ),
    )

    assert mda_chain.check_jacobian(
        derr_approx=Discipline.ApproximationMode.FINITE_DIFFERENCES,
        input_names=inputs,
        output_names=outputs,
        step=1e-6,
        threshold=1e-5,
    )

    assert mda_chain.io.output_data[mda_chain.NORMALIZED_RESIDUAL_NORM][0] < 1e-13


def test_16_disc_parallel() -> None:
    disciplines = create_disciplines_from_desc(DISC_DESCR_16D)
    MDAChain(disciplines)


@pytest.mark.parametrize(
    "in_gtype", [Discipline.GrammarType.SIMPLE, Discipline.GrammarType.JSON]
)
def test_simple_grammar_type(in_gtype) -> None:
    disciplines = create_disciplines_from_desc(DISC_DESCR_16D)
    mda = MDAChain(disciplines)
    assert isinstance(mda.io.input_grammar, SimpleGrammar)
    assert isinstance(mda.discipline_chain.io.input_grammar, SimpleGrammar)
    for inner_mda in mda.inner_mdas:
        assert isinstance(inner_mda.io.input_grammar, SimpleGrammar)


@pytest.mark.parametrize("matrix_type", JacobianAssembly.JacobianType)
@pytest.mark.parametrize(
    "linearization_mode",
    [
        JacobianAssembly.DerivationMode.AUTO,
        JacobianAssembly.DerivationMode.DIRECT,
        JacobianAssembly.DerivationMode.ADJOINT,
    ],
)
def test_self_coupled_mda_jacobian(matrix_type, linearization_mode) -> None:
    """Tests a particular coupling structure."""
    disciplines = analytic_disciplines_from_desc((
        {"c1": "x+1.-0.2*c1"},
        {"obj": "x+c1"},
    ))
    mda = MDAChain(
        disciplines,
        settings=MDAChain_Settings(
            tolerance=1e-14, linear_solver_settings=LGMRES_Settings(rtol=1e-14)
        ),
    )
    mda.matrix_type = matrix_type
    assert mda.check_jacobian(
        input_names=["x"], output_names=["obj"], linearization_mode=linearization_mode
    )

    assert mda.normalized_residual_norm == mda.inner_mdas[0].normalized_residual_norm


def test_no_coupling_jac() -> None:
    """Tests a particular coupling structure."""
    disciplines = analytic_disciplines_from_desc(({"obj": "x"},))
    mda = MDAChain(disciplines)
    assert mda.check_jacobian(input_names=["x"], output_names=["obj"])


def test_sub_coupling_structures(sellar_with_2d_array, sellar_disciplines) -> None:
    """Check that an MDA is correctly instantiated from a coupling structure."""
    coupling_structure = CouplingStructure(sellar_disciplines)
    sub_coupling_structures = [CouplingStructure(sellar_disciplines)]
    mda_sellar = MDAChain(
        sellar_disciplines,
        settings=MDAChain_Settings(
            coupling_structure=coupling_structure,
            sub_coupling_structures=sub_coupling_structures,
        ),
    )
    assert mda_sellar.coupling_structure == coupling_structure
    assert (
        mda_sellar.discipline_chain.disciplines[0].coupling_structure
        == sub_coupling_structures[0]
    )


def test_log_convergence(sellar_with_2d_array, sellar_disciplines) -> None:
    mda_chain = MDAChain(sellar_disciplines)
    assert not mda_chain.settings.log_convergence
    for mda in mda_chain.inner_mdas:
        assert not mda.settings.log_convergence

    mda_chain.settings.log_convergence = True
    assert mda_chain.settings.log_convergence
    for mda in mda_chain.inner_mdas:
        assert mda.settings.log_convergence


def test_parallel_doe() -> None:
    """Test the execution of MDAChain in parallel."""
    obj = generate_parallel_doe("MDAChain", 7)
    assert isclose(array([-obj]), array([608.175]), atol=1e-3)


def test_mda_chain_self_coupling() -> None:
    """Test that a nested MDAChain is not detected as a self-coupled discipline."""
    disciplines = analytic_disciplines_from_desc((
        {"y1": "x"},
        {"c1": "y1+x+0.2*c2"},
        {"c2": "y1+x+1.-0.3*c1"},
        {"obj": "x+c1+c2"},
    ))
    mdachain_lower = MDAChain(
        disciplines, settings=MDAChain_Settings(name="mdachain_lower")
    )
    mdachain_root = MDAChain(
        [mdachain_lower], settings=MDAChain_Settings(name="mdachain_root")
    )

    assert mdachain_root.discipline_chain.disciplines[0] == mdachain_lower
    assert len(mdachain_root.discipline_chain.disciplines) == 1


def test_mdachain_parallel_discipline_chain() -> None:
    """Test that the MDAChain creates ParallelDisciplineChain for parallel tasks, if
    requested."""
    disciplines = analytic_disciplines_from_desc((
        {"a": "x"},
        {"y1": "x1", "b": "a+1"},
        {"x1": "1.-0.3*y1"},
        {"y2": "x2", "c": "a+2"},
        {"x2": "1.-0.3*y2"},
        {"obj1": "x1+x2"},
        {"obj2": "b+c"},
        {"obj": "obj1+obj2"},
    ))
    mdachain = MDAChain(
        disciplines,
        settings=MDAChain_Settings(
            name="mdachain_lower", mdachain_parallelize_tasks=True
        ),
    )
    assert mdachain.check_jacobian(input_names=["x"], output_names=["obj"])
    assert type(mdachain.discipline_chain.disciplines[1]) is ParallelDisciplineChain
    assert type(mdachain.discipline_chain.disciplines[2]) is ParallelDisciplineChain


PARALLEL_OPTIONS = [
    {
        "mdachain_parallelize_tasks": False,
        "mdachain_parallel_settings": {},
    },
    {
        "mdachain_parallelize_tasks": True,
        "mdachain_parallel_settings": {"use_threading": True, "n_processes": 1},
    },
    {
        "mdachain_parallelize_tasks": True,
        "mdachain_parallel_settings": {"use_threading": False, "n_processes": 1},
    },
    {
        "mdachain_parallelize_tasks": True,
        "mdachain_parallel_settings": {"use_threading": True, "n_processes": 2},
    },
    {
        "mdachain_parallelize_tasks": True,
        "mdachain_parallel_settings": {"use_threading": False, "n_processes": 2},
    },
]


@pytest.mark.parametrize("parallel_options", PARALLEL_OPTIONS)
def test_mdachain_paralleldisciplinechain_options(parallel_options) -> None:
    """Test the parallel discipline chain in a MDAChain with various arguments."""
    disciplines = analytic_disciplines_from_desc((
        {"a": "x"},
        {"y1": "x1", "b": "a+1"},
        {"x1": "1.-0.3*y1"},
        {"y2": "x2", "c": "a+2"},
        {"x2": "1.-0.3*y2"},
        {"obj1": "x1+x2"},
        {"obj2": "b+c"},
        {"obj": "obj1+obj2"},
    ))
    mdachain_parallelize_tasks = parallel_options["mdachain_parallelize_tasks"]
    mdo_parallel_chain_options = parallel_options["mdachain_parallel_settings"]
    mdachain = MDAChain(
        disciplines,
        settings=MDAChain_Settings(
            name="mdachain_lower",
            mdachain_parallelize_tasks=mdachain_parallelize_tasks,
            mdachain_parallel_settings=mdo_parallel_chain_options,
        ),
    )
    assert mdachain.check_jacobian(input_names=["x"], output_names=["obj"])


def test_scaling_setter(sellar_with_2d_array, sellar_disciplines) -> None:
    """Test that changing the scaling of a chain modifies all the inner mdas."""
    mda_chain = MDAChain(
        sellar_disciplines,
        settings=MDAChain_Settings(
            tolerance=1e-13, max_mda_iter=30, chain_linearize=True, warm_start=True
        ),
    )
    mda_chain.scaling = MDAChain.ResidualScaling.NO_SCALING
    assert mda_chain.scaling == MDAChain.ResidualScaling.NO_SCALING
    for mda in mda_chain.inner_mdas:
        assert mda.scaling == MDAChain.ResidualScaling.NO_SCALING


def test_initialize_defaults(snapshot) -> None:
    """Test the automated initialization of the default_input_data."""
    disciplines = create_disciplines_from_desc([
        ("A", ["x", "y"], ["z"]),
        ("B", ["a", "z"], ["y", "w"]),
    ])
    del disciplines[0].default_input_data["y"]
    chain = MDAChain(disciplines, settings=MDAChain_Settings(initialize_defaults=False))
    with assert_exception(InvalidDataError, snapshot):
        chain.execute()

    MDAChain(
        disciplines, settings=MDAChain_Settings(initialize_defaults=True)
    ).execute()

    del disciplines[1].default_input_data["z"]
    chain = MDAChain(disciplines, settings=MDAChain_Settings(initialize_defaults=True))
    with assert_exception(ValueError, snapshot):
        chain.execute()

    chain = MDAChain(disciplines, settings=MDAChain_Settings(initialize_defaults=True))
    assert "z" not in chain.io.input_grammar.defaults
    chain.execute({"z": array([0])})
    # Tests that the default inputs are well udapted
    assert "z" in chain.io.input_grammar.defaults
    chain.execute({"z": array([2])})


def test_set_bounds():
    """Test that bounds are properly dispatched to inner-MDAs."""
    mda = MDAChain([
        LinearDiscipline("A", ["x", "b"], ["a"]),
        LinearDiscipline("B", ["a"], ["b", "y"]),
        LinearDiscipline("C", ["b", "d"], ["c"]),
        LinearDiscipline("D", ["c"], ["d", "z"]),
    ])

    lower_bound = -array([1.0])
    upper_bound = array([1.0])

    mda.set_bounds({
        "a": (lower_bound, None),
        "c": (2.0 * lower_bound, None),
        "d": (-lower_bound, 4.0 * upper_bound),
    })

    mda.execute()

    mda_1 = mda.inner_mdas[0]
    assert (mda_1.lower_bound_vector == array([-1.0, -inf])).all()
    assert (mda_1.upper_bound_vector == array([+inf, +inf])).all()

    assert (mda_1._sequence_transformer.lower_bound == array([-1.0, -inf])).all()
    assert (mda_1._sequence_transformer.upper_bound == array([+inf, +inf])).all()

    mda_2 = mda.inner_mdas[1]
    assert (mda_2.lower_bound_vector == array([-2.0, 1.0])).all()
    assert (mda_2.upper_bound_vector == array([+inf, 4.0])).all()

    assert (mda_2._sequence_transformer.lower_bound == array([-2.0, 1.0])).all()
    assert (mda_2._sequence_transformer.upper_bound == array([+inf, 4.0])).all()


def test_settings_cascading(coupled_disciplines):
    """Test that settings are cascaded to inner mdas."""
    mda_chain = MDAChain(coupled_disciplines)

    for inner_mda in mda_chain.inner_mdas:
        assert inner_mda.settings.tolerance == 1e-6
        assert inner_mda.settings.max_mda_iter == 20
        assert not inner_mda.settings.log_convergence

    mda_chain.settings.tolerance = 1e-12
    mda_chain.settings.max_mda_iter = 30
    mda_chain.settings.log_convergence = True

    for inner_mda in mda_chain.inner_mdas:
        assert inner_mda.settings.tolerance == 1e-12
        assert inner_mda.settings.max_mda_iter == 30
        assert inner_mda.settings.log_convergence


@pytest.mark.parametrize(
    "inner_mda_settings",
    [MDAJacobi_Settings(), MDAGaussSeidel_Settings(), MDANewtonRaphson_Settings()],
)
def test_inner_mda_name_setting(coupled_disciplines, inner_mda_settings):
    """Test that the inner MDA name settings is properly used or ignored."""
    mda_chain = MDAChain(
        coupled_disciplines,
        settings=MDAChain_Settings(inner_mda_settings=inner_mda_settings),
    )
    cls = MDA_FACTORY.get_class(inner_mda_settings.target_class_name)
    for inner_mda in mda_chain.inner_mdas:
        assert isinstance(inner_mda, cls)
        assert inner_mda.settings == inner_mda_settings


def test_settings_precedence(coupled_disciplines, caplog):
    """Test the settings precedence of MDA chain over inner MDAs."""
    settings = {
        "linear_solver_settings": None,
        "log_convergence": True,
        "max_mda_iter": 17,
        "max_consecutive_unsuccessful_iterations": 13,
        "tolerance": 3e-14,
        "warm_start": True,
    }
    main_settings = MDAChain_Settings(**settings)
    expected_inner_mda_settings = MDAJacobi_Settings(**settings)
    mda_chain = MDAChain(coupled_disciplines, settings=main_settings)
    for inner_mda in mda_chain.inner_mdas:
        assert inner_mda.settings == expected_inner_mda_settings

    inner_mda_settings = MDAJacobi_Settings(
        linear_solver_settings=TFQMR_Settings(rtol=0.628),
        log_convergence=False,
        max_mda_iter=34,
        max_consecutive_unsuccessful_iterations=26,
        tolerance=6e-28,
        warm_start=False,
    )

    mda_chain = MDAChain(
        coupled_disciplines,
        settings=MDAChain_Settings(inner_mda_settings=inner_mda_settings),
    )
    for inner_mda in mda_chain.inner_mdas:
        assert inner_mda.settings == inner_mda_settings

    mda_chain = MDAChain(
        coupled_disciplines,
        settings=MDAChain_Settings(inner_mda_settings=inner_mda_settings, **settings),
    )
    expected_inner_mda_settings = inner_mda_settings.model_copy()
    for name, value in settings.items():
        setattr(expected_inner_mda_settings, name, value)
    for inner_mda in mda_chain.inner_mdas:
        assert inner_mda.settings == expected_inner_mda_settings

    for name, setting in settings.items():
        msg = (
            f"The {name!r} setting has been set for both the MDAChain "
            "and the inner MDA. The retained value is that of the MDAChain, "
            f"i.e. {setting}."
        )

        assert msg in caplog.text

    assert "linear_solver_settings" not in main_settings


@pytest.mark.parametrize(
    ("classes", "log"), [((Sellar1, SobieskiPropulsion), True), ((Sellar1,), False)]
)
def test_chain_linearize_warning(classes, log, caplog) -> None:
    """Check the warning about switching chain_linearize."""
    MDAChain([cls() for cls in classes])
    if log:
        assert caplog.record_tuples[0] == (
            "gemseo.mda.chain",
            30,
            "No coupling in MDA, switching chain_linearize to True.",
        )
    else:
        assert not caplog.record_tuples
