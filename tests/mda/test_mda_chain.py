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
#        :author: Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from gemseo.core.chain import MDOParallelChain
from gemseo.core.coupling_structure import MDOCouplingStructure
from gemseo.core.derivatives.jacobian_assembly import JacobianAssembly
from gemseo.core.discipline import MDODiscipline
from gemseo.core.grammars.json_grammar import JSONGrammar
from gemseo.core.grammars.simple_grammar import SimpleGrammar
from gemseo.mda.mda_chain import MDAChain
from numpy import array
from numpy import isclose
from numpy import ones

from .test_mda import analytic_disciplines_from_desc

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
    ("P", ["u", "v", "z"], []),
]


def test_set_tolerances(sellar_disciplines):
    """Test that the MDA tolerances can be set at the object instantiation."""
    mda_chain = MDAChain(
        sellar_disciplines, tolerance=1e-3, linear_solver_tolerance=1e-6
    )
    assert mda_chain.tolerance == 1e-3
    assert mda_chain.linear_solver_tolerance == 1e-6

    assert mda_chain.mdo_chain.disciplines[0].tolerance == 1e-3
    assert mda_chain.mdo_chain.disciplines[0].linear_solver_tolerance == 1e-6


def test_set_solver(sellar_disciplines):
    """Test that the MDA tolerances can be set at the object instantiation."""
    mda_chain = MDAChain(
        sellar_disciplines,
        tolerance=1e-3,
        linear_solver_tolerance=1e-6,
        use_lu_fact=True,
        linear_solver="LGMRES",
        linear_solver_options={"restart": 5},
    )
    assert mda_chain.linear_solver == "LGMRES"
    assert mda_chain.use_lu_fact
    assert mda_chain.linear_solver_options == {"restart": 5}

    assert mda_chain.mdo_chain.disciplines[0].linear_solver == "LGMRES"
    assert mda_chain.mdo_chain.disciplines[0].use_lu_fact
    assert mda_chain.mdo_chain.disciplines[0].linear_solver_options == {"restart": 5}


def test_set_linear_solver_tolerance_from_options_constructor(sellar_disciplines):
    """Test that the tolerance cannot be set from the linear_solver_options dictionary.

    In this test, we check that an exception is raised at the MDA instantiation.
    """
    linear_solver_options = {"tol": 1e-6}
    msg = (
        "The linear solver tolerance shall be set"
        " using the linear_solver_tolerance argument."
    )
    with pytest.raises(ValueError, match=msg):
        MDAChain(
            sellar_disciplines,
            tolerance=1e-12,
            linear_solver_options=linear_solver_options,
        )


def test_set_linear_solver_tolerance_from_options_set_attribute(sellar_disciplines):
    """Test that the tolerance cannot be set from the linear_solver_options dictionary.

    In this test, we check that the exception is raised when linearizing the MDA.
    """
    linear_solver_options = {"tol": 1e-6}
    mda_chain = MDAChain(sellar_disciplines, tolerance=1e-12)
    mda_chain.linear_solver_options = linear_solver_options
    input_data = {
        "x_local": np.array([0.7]),
        "x_shared": np.array([1.97763897, 0.2]),
        "y_0": np.array([1.0]),
        "y_1": np.array([1.0]),
    }
    inputs = ["x_local", "x_shared"]
    outputs = ["obj", "c_1", "c_2"]
    mda_chain.add_differentiated_inputs(inputs)
    mda_chain.add_differentiated_outputs(outputs)
    msg = (
        "The linear solver tolerance shall be set"
        " using the linear_solver_tolerance argument."
    )
    with pytest.raises(ValueError, match=msg):
        mda_chain.linearize(input_data)


def test_sellar(tmp_wd, sellar_disciplines):
    """"""
    mda_chain = MDAChain(sellar_disciplines, tolerance=1e-12)
    input_data = {
        "x_local": np.array([0.7]),
        "x_shared": np.array([1.97763897, 0.2]),
        "y_0": np.array([1.0]),
        "y_1": np.array([1.0]),
    }
    inputs = ["x_local", "x_shared"]
    outputs = ["obj", "c_1", "c_2"]
    assert mda_chain.check_jacobian(
        input_data,
        derr_approx=MDODiscipline.COMPLEX_STEP,
        inputs=inputs,
        outputs=outputs,
        threshold=1e-5,
    )
    mda_chain.plot_residual_history(filename="mda_chain_residuals")
    res_file = "MDAJacobi_mda_chain_residuals.png"
    assert Path(res_file).exists()


def test_sellar_chain_linearize(sellar_disciplines):
    inputs = ["x_local", "x_shared"]
    outputs = ["obj", "c_1", "c_2"]
    mda_chain = MDAChain(
        sellar_disciplines,
        tolerance=1e-13,
        max_mda_iter=30,
        chain_linearize=True,
        warm_start=True,
    )

    ok = mda_chain.check_jacobian(
        derr_approx=MDODiscipline.FINITE_DIFFERENCES,
        inputs=inputs,
        outputs=outputs,
        step=1e-6,
        threshold=1e-5,
    )
    assert ok

    assert mda_chain.local_data[mda_chain.RESIDUALS_NORM][0] < 1e-13


def generate_disciplines_from_desc(
    description_list, grammar_type=MDODiscipline.JSON_GRAMMAR_TYPE
):
    disciplines = []
    data = ones(1)
    for desc in description_list:
        name = desc[0]
        input_d = {k: data for k in desc[1]}
        output_d = {k: data for k in desc[2]}
        disc = MDODiscipline(name)
        disc.input_grammar.update_from_data(input_d)
        disc.output_grammar.update_from_data(output_d)
        disciplines.append(disc)
    return disciplines


def test_16_disc_parallel():
    disciplines = generate_disciplines_from_desc(DISC_DESCR_16D)
    MDAChain(disciplines)


@pytest.mark.parametrize(
    "in_gtype", [MDODiscipline.SIMPLE_GRAMMAR_TYPE, MDODiscipline.JSON_GRAMMAR_TYPE]
)
def test_simple_grammar_type(in_gtype):
    disciplines = generate_disciplines_from_desc(DISC_DESCR_16D)
    mda = MDAChain(disciplines, grammar_type=MDODiscipline.SIMPLE_GRAMMAR_TYPE)

    assert type(mda.input_grammar) == SimpleGrammar
    assert type(mda.mdo_chain.input_grammar) == SimpleGrammar
    for inner_mda in mda.inner_mdas:
        assert type(inner_mda.input_grammar) == SimpleGrammar


def test_mix_sim_jsongrammar(sellar_disciplines):
    mda_chain_s = MDAChain(
        sellar_disciplines,
        grammar_type=MDODiscipline.SIMPLE_GRAMMAR_TYPE,
    )
    assert type(mda_chain_s.input_grammar) == SimpleGrammar

    out_1 = mda_chain_s.execute()

    mda_chain = MDAChain(sellar_disciplines)
    assert type(mda_chain.input_grammar) == JSONGrammar

    out_2 = mda_chain.execute()

    assert out_1["obj"] == out_2["obj"]


@pytest.mark.parametrize(
    "matrix_type", [JacobianAssembly.SPARSE, JacobianAssembly.LINEAR_OPERATOR]
)
@pytest.mark.parametrize(
    "linearization_mode",
    [
        JacobianAssembly.AUTO_MODE,
        JacobianAssembly.DIRECT_MODE,
        JacobianAssembly.ADJOINT_MODE,
    ],
)
def test_self_coupled_mda_jacobian(matrix_type, linearization_mode):
    """Tests a particular coupling structure."""
    disciplines = analytic_disciplines_from_desc(
        (
            {"c1": "x+1.-0.2*c1"},
            {"obj": "x+c1"},
        )
    )
    mda = MDAChain(disciplines)
    mda.matrix_type = matrix_type
    assert mda.check_jacobian(
        inputs=["x"], outputs=["obj"], linearization_mode=linearization_mode
    )

    assert mda.normed_residual == mda.inner_mdas[0].normed_residual


def test_no_coupling_jac():
    """Tests a particular coupling structure."""
    disciplines = analytic_disciplines_from_desc(({"obj": "x"},))
    mda = MDAChain(disciplines)
    assert mda.check_jacobian(inputs=["x"], outputs=["obj"])


def test_sub_coupling_structures(sellar_disciplines):
    """Check that an MDA is correctly instantiated from a coupling structure."""
    coupling_structure = MDOCouplingStructure(sellar_disciplines)
    sub_coupling_structures = [MDOCouplingStructure(sellar_disciplines)]
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


def test_log_convergence(sellar_disciplines):
    mda_chain = MDAChain(sellar_disciplines)
    assert not mda_chain.log_convergence
    for mda in mda_chain.inner_mdas:
        assert not mda.log_convergence

    mda_chain.log_convergence = True
    assert mda_chain.log_convergence
    for mda in mda_chain.inner_mdas:
        assert mda.log_convergence


def test_parallel_doe(generate_parallel_doe_data):
    """Test the execution of MDAChain in parallel.

    Args:
        generate_parallel_doe_data: Fixture that returns the optimum solution to
            a parallel DOE scenario for a particular `main_mda_name`
            and n_samples.
    """
    obj = generate_parallel_doe_data("MDAChain", 7)
    assert isclose(array([-obj]), array([608.175]), atol=1e-3)


def test_mda_chain_self_coupling():
    """Test that a nested MDAChain is not detected as a self-coupled discipline."""
    disciplines = analytic_disciplines_from_desc(
        (
            {"y1": "x"},
            {"c1": "y1+x+0.2*c2"},
            {"c2": "y1+x+1.-0.3*c1"},
            {"obj": "x+c1+c2"},
        )
    )
    mdachain_lower = MDAChain(disciplines, name="mdachain_lower")
    mdachain_root = MDAChain([mdachain_lower], name="mdachain_root")

    assert mdachain_root.mdo_chain.disciplines[0] == mdachain_lower
    assert len(mdachain_root.mdo_chain.disciplines) == 1


def test_mdachain_parallelmdochain():
    """Test that the MDAChain creates MDOParallelChain for parallel tasks, if
    requested."""
    disciplines = analytic_disciplines_from_desc(
        (
            {"a": "x"},
            {"y1": "x1", "b": "a+1"},
            {"x1": "1.-0.3*y1"},
            {"y2": "x2", "c": "a+2"},
            {"x2": "1.-0.3*y2"},
            {"obj1": "x1+x2"},
            {"obj2": "b+c"},
            {"obj": "obj1+obj2"},
        )
    )
    mdachain = MDAChain(
        disciplines, name="mdachain_lower", mdachain_parallelize_tasks=True
    )
    assert mdachain.check_jacobian(inputs=["x"], outputs=["obj"])
    assert type(mdachain.mdo_chain.disciplines[1]) is MDOParallelChain
    assert type(mdachain.mdo_chain.disciplines[2]) is MDOParallelChain


PARALLEL_OPTIONS = [
    {
        "mdachain_parallelize_tasks": False,
        "mdachain_parallel_options": {},
    },
    {
        "mdachain_parallelize_tasks": True,
        "mdachain_parallel_options": {"use_threading": True, "n_processes": 1},
    },
    {
        "mdachain_parallelize_tasks": True,
        "mdachain_parallel_options": {"use_threading": False, "n_processes": 1},
    },
    {
        "mdachain_parallelize_tasks": True,
        "mdachain_parallel_options": {"use_threading": True, "n_processes": 2},
    },
    {
        "mdachain_parallelize_tasks": True,
        "mdachain_parallel_options": {"use_threading": False, "n_processes": 2},
    },
]


@pytest.mark.parametrize("parallel_options", PARALLEL_OPTIONS)
def test_mdachain_parallelmdochain_options(parallel_options):
    """Test the parallel MDO chain in a MDAChain with various arguments."""
    disciplines = analytic_disciplines_from_desc(
        (
            {"a": "x"},
            {"y1": "x1", "b": "a+1"},
            {"x1": "1.-0.3*y1"},
            {"y2": "x2", "c": "a+2"},
            {"x2": "1.-0.3*y2"},
            {"obj1": "x1+x2"},
            {"obj2": "b+c"},
            {"obj": "obj1+obj2"},
        )
    )
    mdachain_parallelize_tasks = parallel_options["mdachain_parallelize_tasks"]
    mdo_parallel_chain_options = parallel_options["mdachain_parallel_options"]
    mdachain = MDAChain(
        disciplines,
        name="mdachain_lower",
        mdachain_parallelize_tasks=mdachain_parallelize_tasks,
        mdachain_parallel_options=mdo_parallel_chain_options,
    )
    assert mdachain.check_jacobian(inputs=["x"], outputs=["obj"])
