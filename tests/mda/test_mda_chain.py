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

import re
from pathlib import Path

import pytest
from numpy import array
from numpy import inf
from numpy import isclose

from gemseo.core.chains.parallel_chain import MDOParallelChain
from gemseo.core.coupling_structure import CouplingStructure
from gemseo.core.derivatives.jacobian_assembly import JacobianAssembly
from gemseo.core.discipline import Discipline
from gemseo.core.grammars.errors import InvalidDataError
from gemseo.core.grammars.simple_grammar import SimpleGrammar
from gemseo.mda.mda_chain import MDAChain
from gemseo.problems.mdo.scalable.linear.disciplines_generator import (
    create_disciplines_from_desc,
)
from gemseo.problems.mdo.scalable.linear.linear_discipline import LinearDiscipline
from gemseo.problems.mdo.sellar.utils import get_initial_data

from .test_mda import analytic_disciplines_from_desc
from .utils import generate_parallel_doe

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


def test_set_tolerances(sellar_disciplines) -> None:
    """Test that the MDA tolerances can be set at the object instantiation."""
    mda_chain = MDAChain(
        sellar_disciplines, tolerance=1e-3, linear_solver_tolerance=1e-6
    )
    assert mda_chain.settings.tolerance == 1e-3
    assert mda_chain.settings.linear_solver_tolerance == 1e-6

    assert mda_chain.mdo_chain.disciplines[0].settings.tolerance == 1e-3
    assert mda_chain.mdo_chain.disciplines[0].settings.linear_solver_tolerance == 1e-6


def test_set_solver(sellar_disciplines) -> None:
    """Test that the MDA tolerances can be set at the object instantiation."""
    mda_chain = MDAChain(
        sellar_disciplines,
        tolerance=1e-3,
        linear_solver_tolerance=1e-6,
        use_lu_fact=True,
        linear_solver="LGMRES",
        linear_solver_settings={"restart": 5},
    )
    assert mda_chain.settings.linear_solver == "LGMRES"
    assert mda_chain.settings.use_lu_fact
    assert mda_chain.settings.linear_solver_settings == {"restart": 5}

    sub_mda1_settings = mda_chain.mdo_chain.disciplines[0].settings
    assert sub_mda1_settings.linear_solver == "LGMRES"
    assert sub_mda1_settings.use_lu_fact
    assert sub_mda1_settings.linear_solver_settings == {"restart": 5}


def test_set_linear_solver_tolerance_from_options_constructor(
    sellar_disciplines,
) -> None:
    """Test that the tolerance cannot be set from the linear_solver_settings dictionary.

    In this test, we check that an exception is raised at the MDA instantiation.
    """
    linear_solver_settings = {"rtol": 1e-6}
    msg = (
        "The linear solver tolerance shall be set"
        " using the linear_solver_tolerance argument."
    )
    with pytest.raises(ValueError, match=msg):
        MDAChain(
            sellar_disciplines,
            tolerance=1e-12,
            linear_solver_settings=linear_solver_settings,
        )


def test_set_linear_solver_tolerance_from_options_set_attribute(
    sellar_disciplines,
) -> None:
    """Test that the tolerance cannot be set from the linear_solver_settings dictionary.

    In this test, we check that the exception is raised when linearizing the MDA.
    """
    linear_solver_settings = {"rtol": 1e-6}
    mda_chain = MDAChain(sellar_disciplines, tolerance=1e-12)
    mda_chain.settings.linear_solver_settings = linear_solver_settings
    input_data = get_initial_data()
    inputs = ["x_1", "x_shared"]
    outputs = ["obj", "c_1", "c_2"]
    mda_chain.add_differentiated_inputs(inputs)
    mda_chain.add_differentiated_outputs(outputs)
    msg = (
        "The linear solver tolerance shall be set"
        " using the linear_solver_tolerance argument."
    )
    with pytest.raises(ValueError, match=msg):
        mda_chain.linearize(input_data)


def test_sellar(tmp_wd, sellar_disciplines) -> None:
    """"""
    mda_chain = MDAChain(
        sellar_disciplines, inner_mda_name="MDAJacobi", tolerance=1e-12
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


def test_sellar_chain_linearize(sellar_disciplines) -> None:
    inputs = ["x_1", "x_shared"]
    outputs = ["obj", "c_1", "c_2"]
    mda_chain = MDAChain(
        sellar_disciplines,
        tolerance=1e-13,
        max_mda_iter=30,
        chain_linearize=True,
        warm_start=True,
    )

    assert mda_chain.check_jacobian(
        derr_approx=Discipline.ApproximationMode.FINITE_DIFFERENCES,
        input_names=inputs,
        output_names=outputs,
        step=1e-6,
        threshold=1e-5,
    )

    assert mda_chain.io.data[mda_chain.NORMALIZED_RESIDUAL_NORM][0] < 1e-13


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
    assert isinstance(mda.mdo_chain.io.input_grammar, SimpleGrammar)
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
    mda = MDAChain(disciplines, tolerance=1e-14, linear_solver_tolerance=1e-14)
    mda.matrix_type = matrix_type
    assert mda.check_jacobian(
        input_names=["x"], output_names=["obj"], linearization_mode=linearization_mode
    )

    assert mda.normed_residual == mda.inner_mdas[0].normed_residual


def test_no_coupling_jac() -> None:
    """Tests a particular coupling structure."""
    disciplines = analytic_disciplines_from_desc(({"obj": "x"},))
    mda = MDAChain(disciplines)
    assert mda.check_jacobian(input_names=["x"], output_names=["obj"])


def test_sub_coupling_structures(sellar_disciplines) -> None:
    """Check that an MDA is correctly instantiated from a coupling structure."""
    coupling_structure = CouplingStructure(sellar_disciplines)
    sub_coupling_structures = [CouplingStructure(sellar_disciplines)]
    mda_sellar = MDAChain(
        sellar_disciplines,
        coupling_structure=coupling_structure,
        sub_coupling_structures=sub_coupling_structures,
    )
    assert mda_sellar.coupling_structure == coupling_structure
    assert (
        mda_sellar.mdo_chain.disciplines[0].coupling_structure
        == sub_coupling_structures[0]
    )


def test_log_convergence(sellar_disciplines) -> None:
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
    mdachain_lower = MDAChain(disciplines, name="mdachain_lower")
    mdachain_root = MDAChain([mdachain_lower], name="mdachain_root")

    assert mdachain_root.mdo_chain.disciplines[0] == mdachain_lower
    assert len(mdachain_root.mdo_chain.disciplines) == 1


def test_mdachain_parallelmdochain() -> None:
    """Test that the MDAChain creates MDOParallelChain for parallel tasks, if
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
        disciplines, name="mdachain_lower", mdachain_parallelize_tasks=True
    )
    assert mdachain.check_jacobian(input_names=["x"], output_names=["obj"])
    assert type(mdachain.mdo_chain.disciplines[1]) is MDOParallelChain
    assert type(mdachain.mdo_chain.disciplines[2]) is MDOParallelChain


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
def test_mdachain_parallelmdochain_options(parallel_options) -> None:
    """Test the parallel MDO chain in a MDAChain with various arguments."""
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
        name="mdachain_lower",
        mdachain_parallelize_tasks=mdachain_parallelize_tasks,
        mdachain_parallel_settings=mdo_parallel_chain_options,
    )
    assert mdachain.check_jacobian(input_names=["x"], output_names=["obj"])


def test_max_mda_iter(sellar_disciplines) -> None:
    """Test that changing the max_mda_iter of a chain modifies all the inner mdas."""
    mda_chain = MDAChain(
        sellar_disciplines,
        tolerance=1e-13,
        max_mda_iter=30,
        chain_linearize=True,
        warm_start=True,
    )
    assert mda_chain.settings.max_mda_iter == 30
    for mda in mda_chain.inner_mdas:
        assert mda.settings.max_mda_iter == 30

    mda_chain.settings.max_mda_iter = 10
    assert mda_chain.settings.max_mda_iter == 10
    for mda in mda_chain.inner_mdas:
        assert mda.settings.max_mda_iter == 10


def test_scaling(sellar_disciplines) -> None:
    """Test that changing the scaling of a chain modifies all the inner mdas."""
    mda_chain = MDAChain(
        sellar_disciplines,
        tolerance=1e-13,
        max_mda_iter=30,
        chain_linearize=True,
        warm_start=True,
    )
    mda_chain.scaling = MDAChain.ResidualScaling.NO_SCALING
    assert mda_chain.scaling == MDAChain.ResidualScaling.NO_SCALING
    for mda in mda_chain.inner_mdas:
        assert mda.scaling == MDAChain.ResidualScaling.NO_SCALING


def test_initialize_defaults() -> None:
    """Test the automated initialization of the default_input_data."""
    disciplines = create_disciplines_from_desc([
        ("A", ["x", "y"], ["z"]),
        ("B", ["a", "z"], ["y", "w"]),
    ])
    del disciplines[0].default_input_data["y"]
    chain = MDAChain(disciplines, initialize_defaults=False)
    with pytest.raises(InvalidDataError, match=re.escape("Missing required names: y.")):
        chain.execute()

    MDAChain(disciplines, initialize_defaults=True).execute()

    del disciplines[1].default_input_data["z"]
    chain = MDAChain(disciplines, initialize_defaults=True)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Cannot compute the inputs a, x, y, z, for the following disciplines A, B."
        ),
    ):
        chain.execute()

    chain = MDAChain(disciplines, initialize_defaults=True)
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
