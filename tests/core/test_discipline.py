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
#        :author: Damien Guenot
#                 Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import contextlib
import logging
import os
import platform
import re
from pathlib import Path
from pathlib import PurePosixPath
from pathlib import PureWindowsPath
from typing import TYPE_CHECKING

import pytest
from numpy import array
from numpy import complex128
from numpy import concatenate
from numpy import ndarray
from numpy import ones
from numpy.linalg import norm

from gemseo import create_discipline
from gemseo.caches.hdf5_cache import HDF5Cache
from gemseo.caches.simple_cache import SimpleCache
from gemseo.core.chains.chain import MDOChain
from gemseo.core.discipline import Discipline
from gemseo.core.discipline.data_processor import ComplexDataProcessor
from gemseo.core.execution_statistics import ExecutionStatistics
from gemseo.core.execution_status import ExecutionStatus
from gemseo.core.grammars.errors import InvalidDataError
from gemseo.core.grammars.json_grammar import JSONGrammar
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.disciplines.auto_py import AutoPyDiscipline
from gemseo.mda.base_mda import BaseMDA
from gemseo.problems.mdo.sellar.sellar_1 import Sellar1
from gemseo.problems.mdo.sellar.variables import X_1
from gemseo.problems.mdo.sellar.variables import X_SHARED
from gemseo.problems.mdo.sellar.variables import Y_2
from gemseo.problems.mdo.sobieski._disciplines_sg import SobieskiStructureSG
from gemseo.problems.mdo.sobieski.core.problem import SobieskiProblem
from gemseo.problems.mdo.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.mdo.sobieski.disciplines import SobieskiMission
from gemseo.problems.mdo.sobieski.disciplines import SobieskiPropulsion
from gemseo.problems.mdo.sobieski.disciplines import SobieskiStructure
from gemseo.utils.compatibility.scipy import sparse_classes
from gemseo.utils.discipline import DummyDiscipline
from gemseo.utils.pickle import from_pickle
from gemseo.utils.pickle import to_pickle
from gemseo.utils.repr_html import REPR_HTML_WRAPPER

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.typing import StrKeyMapping

Status = ExecutionStatus.Status


def check_jac_equals(
    jac_1: dict[str, ndarray],
    jac_2: dict[str, ndarray],
) -> bool:
    """Check that two Jacobian matrices are equals.

    Args:
        jac_1: A jacobian matrix.
        jac_2: A jacobian matrix.

    Returns:
        True if the two jacobian matrices are equal.
    """
    if sorted(jac_1.keys()) != sorted(jac_2.keys()):
        return False
    for out, jac_dict in jac_1.items():
        if sorted(jac_dict.keys()) != sorted(jac_2[out].keys()):
            return False
        for inpt, jac_loc in jac_dict.items():
            if not (jac_loc == jac_2[out][inpt]).all():
                return False

    return True


@pytest.fixture
def sobieski_chain() -> tuple[MDOChain, dict[str, ndarray]]:
    """Build a Sobieski chain.

    Returns:
         Tuple containing a Sobieski MDOChain instance
             and the defaults inputs of the chain.
    """
    chain = MDOChain([
        SobieskiStructure(),
        SobieskiAerodynamics(),
        SobieskiPropulsion(),
        SobieskiMission(),
    ])
    chain_inputs = chain.io.input_grammar
    indata = SobieskiProblem().get_default_inputs(names=chain_inputs)
    return chain, indata


@pytest.fixture
def hybrid_jacobian_discipline() -> Discipline:
    class HybridDiscipline(Discipline):
        def __init__(self) -> None:
            super().__init__()
            self.input_grammar.update_from_names(["x_1"])
            self.input_grammar.update_from_names(["x_2"])
            self.input_grammar.update_from_names(["x_3"])
            self.output_grammar.update_from_names(["y_1"])
            self.output_grammar.update_from_names(["y_2"])
            self.output_grammar.update_from_names(["y_3"])
            self.default_input_data = {
                "x_1": array([1.0]),
                "x_2": array([2.0]),
                "x_3": array([1.0]),
            }

        def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
            self.io.data["y_1"] = input_data["x_1"] * input_data["x_2"]
            self.io.data["y_2"] = (
                input_data["x_1"] * input_data["x_2"] * input_data["x_3"]
            )
            self.io.data["y_3"] = input_data["x_1"]

        def _compute_jacobian(
            self,
            input_names: Iterable[str] = (),
            output_names: Iterable[str] = (),
        ) -> None:
            self._init_jacobian()
            x1 = array([self.get_input_data(with_namespaces=False)["x_1"]])
            x2 = array([self.get_input_data(with_namespaces=False)["x_2"]])
            x3 = array([self.get_input_data(with_namespaces=False)["x_3"]])
            self.jac = {"y_1": {"x_1": x2}, "y_2": {"x_2": x1 * x3}}

    return HybridDiscipline()


@pytest.mark.xfail
def test_instantiate_grammars() -> None:
    """Test the instantiation of the grammars."""
    chain = MDOChain([SobieskiAerodynamics()])
    chain.disciplines[0]._instantiate_grammars(None, None, grammar_type="JSONGrammar")
    assert isinstance(chain.disciplines[0].input_grammar, JSONGrammar)


def test_execute_status_error(sobieski_chain) -> None:
    """Test the execution with a failed status."""
    chain, indata = sobieski_chain
    chain.execution_status.value = ExecutionStatus.Status.FAILED
    with pytest.raises(ValueError):
        chain.execute(indata)


def test_check_input_data_exception_chain(sobieski_chain) -> None:
    """Test the check input data exception."""
    chain, indata = sobieski_chain
    del indata["x_1"]
    with pytest.raises(InvalidDataError):
        chain.io.input_grammar.validate(indata)


@pytest.mark.parametrize(
    "grammar_type", [Discipline.GrammarType.JSON, Discipline.GrammarType.SIMPLE]
)
def test_check_input_data_exception(grammar_type) -> None:
    """Test the check input data exception."""
    if grammar_type == Discipline.GrammarType.SIMPLE:
        struct = SobieskiStructureSG()
    else:
        struct = SobieskiStructure()

    struct_inputs = struct.io.input_grammar
    indata = SobieskiProblem().get_default_inputs(names=struct_inputs)
    del indata["x_1"]

    with pytest.raises(InvalidDataError, match=r".*Missing required names: x_1"):
        struct.io.input_grammar.validate(indata)

    struct.execute(indata)

    del struct.io.input_grammar.defaults["x_1"]
    with pytest.raises(InvalidDataError, match=r".*Missing required names: x_1"):
        struct.execute(indata)


@pytest.mark.xfail
def test_outputs() -> None:
    """Test the execution of a Discipline."""
    struct = SobieskiStructure()
    with pytest.raises(InvalidDataError):
        struct.io.output_grammar.validate(struct.io.data)
    indata = SobieskiProblem().get_default_inputs()
    struct.execute(indata)
    in_array = struct.get_inputs_asarray()
    assert len(in_array) == 13


def test_get_input_data(sobieski_chain) -> None:
    """Test get_input_data."""
    chain, indata_ref = sobieski_chain
    chain.execute(indata_ref)
    indata = chain.get_input_data()
    assert sorted(indata.keys()) == sorted(indata_ref.keys())


def test_reset_statuses_for_run_error(sobieski_chain) -> None:
    """Test the reset of the discipline status."""
    chain, _ = sobieski_chain
    chain.execution_status.value = ExecutionStatus.Status.FAILED


def test_check_jac_fdapprox() -> None:
    """Test the finite difference approximation."""
    aero = SobieskiAerodynamics("complex128")
    inpts = aero.io.input_grammar.defaults
    aero.linearization_mode = aero.ApproximationMode.FINITE_DIFFERENCES
    aero.linearize(inpts, compute_all_jacobians=True)
    aero.check_jacobian(inpts)
    aero.linearization_mode = "auto"
    aero.check_jacobian(inpts)


def test_check_jac_csapprox() -> None:
    """Test the complex step approximation."""
    aero = SobieskiAerodynamics("complex128")
    aero.linearization_mode = aero.ApproximationMode.COMPLEX_STEP
    aero.linearize(compute_all_jacobians=True)
    aero.check_jacobian()


@pytest.mark.parametrize(
    "hybrid_approximation_mode",
    ["hybrid_complex_step", "hybrid_finite_differences", "hybrid_centered_differences"],
)
def test_check_jac_hybrid_approx(
    hybrid_jacobian_discipline, hybrid_approximation_mode
) -> None:
    """Test the hybrid finite difference approximation."""

    disc = hybrid_jacobian_discipline

    inputs = disc.default_input_data
    disc.linearization_mode = hybrid_approximation_mode
    disc.linearize(inputs, compute_all_jacobians=True)
    disc.check_jacobian(
        inputs, linearization_mode=disc.ApproximationMode.FINITE_DIFFERENCES
    )


def test_check_jac_approx_plot(tmp_wd) -> None:
    """Test the generation of the gradient plot."""
    aero = SobieskiAerodynamics()
    aero.linearize(compute_all_jacobians=True)
    file_path = "gradients_validation.pdf"
    aero.check_jacobian(step=10.0, plot_result=True, file_path=file_path)
    assert os.path.exists(file_path)


def test_check_lin_threshold() -> None:
    """Check the linearization threshold."""
    aero = SobieskiAerodynamics()
    problem = SobieskiProblem()
    indata = problem.get_default_inputs(names=aero.io.input_grammar)
    aero.check_jacobian(indata, threshold=1e-50)


def test_input_exist() -> None:
    """Test is_input_existing."""
    sr = SobieskiAerodynamics()
    problem = SobieskiProblem()
    indata = problem.get_default_inputs(names=sr.io.input_grammar)
    assert next(iter(indata.keys())) in sr.io.input_grammar
    assert "bidon" not in sr.io.input_grammar


def test_get_all_inputs_outputs_name() -> None:
    """Test get_all_input_outputs_name method."""
    aero = SobieskiAerodynamics()
    problem = SobieskiProblem()
    indata = problem.get_default_inputs(names=aero.io.input_grammar)
    for data_name in indata:
        assert data_name in aero.io.input_grammar


@pytest.mark.xfail
def test_get_all_inputs_outputs() -> None:
    """Test get all_inputs_outputs method."""
    aero = SobieskiAerodynamics()
    problem = SobieskiProblem()
    indata = problem.get_default_inputs(names=aero.io.input_grammar)
    aero.execute(indata)
    aero.get_all_inputs()
    aero.get_all_outputs()
    arr = concatenate(list(aero.get_all_outputs()))
    assert isinstance(arr, ndarray)
    assert len(arr) > 0
    arr = aero.get_inputs_asarray()
    assert isinstance(arr, ndarray)
    assert len(arr) > 0


def test_serialize_deserialize(tmp_wd) -> None:
    """Test the serialization/deserialization method."""
    aero = SobieskiAerodynamics()
    aero.io.data_processor = ComplexDataProcessor()
    out_file = "sellar1.o"
    input_data = SobieskiProblem().get_default_inputs()
    aero.execute(input_data)
    locd = aero.io.data
    to_pickle(aero, out_file)
    saero_u = from_pickle(out_file)
    for k, v in locd.items():
        assert k in saero_u.io.data
        assert (v == saero_u.io.data[k]).all()

    def attr_list():
        return ["numpy_test"]

    aero.get_attributes_to_serialize = attr_list
    with pytest.raises(AttributeError):
        to_pickle(aero, out_file)

    saero_u_dict = saero_u.__dict__
    ok = True
    for k in aero.__dict__:
        if k not in saero_u_dict and k != "get_attributes_to_serialize":
            ok = False
    assert ok


def test_serialize_run_deserialize(tmp_wd) -> None:
    """Test serialization, run and deserialization."""
    aero = SobieskiAerodynamics()
    out_file = "sellar1.o"
    input_data = SobieskiProblem().get_default_inputs()
    to_pickle(aero, out_file)
    saero_u = from_pickle(out_file)
    to_pickle(saero_u, out_file)
    saero_u.execute(input_data)
    to_pickle(saero_u, out_file)
    saero_loc = from_pickle(out_file)
    saero_loc.execution_status.value = "DONE"
    saero_loc.execute(input_data)

    for k, v in saero_loc.io.data.items():
        assert k in saero_u.io.data
        assert (v == saero_u.io.data[k]).all()


def test_serialize_hdf_cache(tmp_wd) -> None:
    """Test the serialization into a HDF5 cache."""
    aero = SobieskiAerodynamics()
    cache_hdf_file = "aero_cache.h5"
    aero.set_cache(aero.CacheType.HDF5, hdf_file_path=cache_hdf_file)
    aero.execute()
    out_file = "sob_aero.pckl"
    to_pickle(aero, out_file)
    saero_u = from_pickle(out_file)
    assert saero_u.cache.last_entry.outputs["y_2"] is not None


def test_data_processor() -> None:
    """Test the data processor."""
    aero = SobieskiAerodynamics()
    input_data = SobieskiProblem().get_default_inputs()
    aero.io.data_processor = ComplexDataProcessor()
    out_data = aero.execute(input_data)
    for v in aero.get_output_data().values():
        assert isinstance(v, ndarray)
        assert v.dtype == complex128
    # Mix data processor and cache
    out_data2 = aero.execute(input_data)
    for k, v in out_data.items():
        assert (out_data2[k] == v).all()


def test_diff_inputs_outputs() -> None:
    """Test the differentiation w.r.t inputs and outputs."""
    d = DummyDiscipline()
    with pytest.raises(
        ValueError,
        match=f"Cannot differentiate the discipline {d.name} w.r.t. the inputs "
        r"that are not among the discipline inputs: \[\]",
    ):
        d.add_differentiated_inputs(["toto"])
    with pytest.raises(
        ValueError,
        match=f"Cannot differentiate the discipline {d.name} for variables "
        r"that are not among the discipline outputs: \[\]",
    ):
        d.add_differentiated_outputs(["toto"])
    d.add_differentiated_inputs()


def test_linearize_errors() -> None:
    """Test the exceptions and errors during discipline linearization."""
    DummyDiscipline()._compute_jacobian()

    class LinDisc(Discipline):
        def __init__(self) -> None:
            super().__init__()
            self.io.input_grammar.update_from_names(["x"])
            self.io.output_grammar.update_from_names(["y"])

        def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
            self.io.data["y"] = array([2.0])

        def _compute_jacobian(self, input_names=(), output_names=()) -> None:
            self._init_jacobian()
            self.jac = {"y": {"x": array([0.0])}}

    d2 = LinDisc()
    d2.execute({"x": array([1.0])})
    # Shape is not 2D
    with pytest.raises(ValueError):
        d2.linearize({"x": array([1])}, compute_all_jacobians=True)

    d2.io.data["y"] = 1
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The shape (1,) of the Jacobian matrix dy/dx of the discipline LinDisc "
            "does not match (output_size, input_size)=(1, 1)."
        ),
    ):
        d2._check_jacobian_shape(["x"], ["y"])

    class SM(SobieskiMission):
        def _compute_jacobian(self, input_names=(), output_names=()) -> None:
            super()._compute_jacobian(
                input_names=input_names, output_names=output_names
            )
            self.jac["y_4"]["x_shared"] += 3.0

    sm = SM()
    success = sm.check_jacobian(input_names=["x_shared"], output_names=["y_4"])
    assert not success


def test_check_jacobian_errors() -> None:
    """Test the errors raised during check_jacobian."""
    sm = SobieskiMission()
    with pytest.raises(ValueError):
        sm._check_jacobian_shape([], [])

    sm.execute()
    sm.linearize(compute_all_jacobians=True)
    sm._check_jacobian_shape(sm.io.input_grammar, sm.io.output_grammar)
    sm.io.data.pop("x_shared")
    sm._check_jacobian_shape(sm.io.input_grammar, sm.io.output_grammar)
    sm.io.data.pop("y_4")
    sm._check_jacobian_shape(sm.io.input_grammar, sm.io.output_grammar)


def test_check_jacobian() -> None:
    """Test the check_jacobian method."""

    class SM(SobieskiMission):
        def _compute_jacobian(self, input_names=(), output_names=()) -> None:
            super()._compute_jacobian(
                input_names=input_names, output_names=output_names
            )
            del self.jac["y_4"]

    sm = SM()
    sm.execute()
    sm._compute_jacobian()

    msg = f"The discipline {sm.name} was not linearized."
    with pytest.raises(ValueError, match=msg):
        sm.linearize(compute_all_jacobians=True)

    class SM(SobieskiMission):
        def _compute_jacobian(self, input_names=(), output_names=()) -> None:
            super()._compute_jacobian(
                input_names=input_names, output_names=output_names
            )
            del self.jac["y_4"]["x_shared"]

    sm2 = SM()

    with pytest.raises(KeyError):
        sm2.linearize(compute_all_jacobians=True)


def test_check_jacobian_2() -> None:
    """Test check_jacobian."""
    x = array([1.0, 2.0])

    class LinDisc(Discipline):
        def __init__(self) -> None:
            super().__init__()
            self.io.input_grammar.update_from_names(["x"])
            self.io.output_grammar.update_from_names(["y"])
            self.io.input_grammar.defaults = {"x": x}
            self.jac_key = "x"
            self.jac_len = 2

        def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
            self.io.data["y"] = array([2.0])

        def _compute_jacobian(self, input_names=(), output_names=()) -> None:
            self._init_jacobian()
            self.jac = {"y": {self.jac_key: array([[0.0] * self.jac_len])}}

    disc = LinDisc()
    disc.jac_key = "z"
    with pytest.raises(KeyError):
        disc.linearize({"x": x}, compute_all_jacobians=True)
    disc.jac_key = "x"
    disc.jac_len = 3
    with pytest.raises(ValueError):
        disc.linearize({"x": x}, compute_all_jacobians=True)
    #         # Test not multiple d/dX
    disc.jac = {"y": {"x": array([[0.0], [1.0], [3.0]])}}
    with pytest.raises(ValueError):
        disc.linearize({"x": x}, compute_all_jacobians=True)
    #         # Test inconsistent output size for gradient
    #         # Test too small d/dX
    disc.jac = {"y": {"x": array([[0.0]])}}
    with pytest.raises(ValueError):
        disc.linearize({"x": x}, compute_all_jacobians=True)


def test_check_jacobian_input_data() -> None:
    sellar_1 = create_discipline("Sellar1")
    input_data = {
        X_1: array([3.0]),
        X_SHARED: array([3.0, 3.0]),
        Y_2: array([3.0]),
    }
    sellar_1.check_jacobian(
        input_data=input_data,
        input_names=[Y_2],
    )


def test_check_jacobian_parallel_fd() -> None:
    """Test check_jacobian in parallel."""
    sm = SobieskiMission()
    sm.check_jacobian(step=1e-6, threshold=1e-6, parallel=True, n_processes=6)


def test_check_jacobian_parallel_cplx() -> None:
    """Test check_jacobian in parallel with complex-step."""
    sm = SobieskiMission()
    sm.check_jacobian(
        derr_approx=sm.ApproximationMode.COMPLEX_STEP,
        step=1e-30,
        threshold=1e-6,
        parallel=True,
        n_processes=6,
    )


def test_execute_rerun_errors() -> None:
    """Test the execution and errors during re-run of Discipline."""

    class MyDisc(Discipline):
        def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
            self.io.data["b"] = array([1.0])

    d = MyDisc()
    d.io.input_grammar.update_from_names(["a"])
    d.io.output_grammar.update_from_names(["b"])
    d.execute({"a": [1]})
    d.execution_status.value = Status.RUNNING
    with pytest.raises(ValueError):
        d.execute({"a": [2]})

    d.execution_status.value = Status.DONE
    d.execute({"a": [1]})


def test_cache() -> None:
    """Test the Discipline cache."""
    sm = SobieskiMission(enable_delay=0.1)
    sm.cache.tolerance = 1e-6
    xs = sm.io.input_grammar.defaults["x_shared"]
    sm.execute({"x_shared": xs})
    t0 = sm.execution_statistics.duration
    sm.execute({"x_shared": xs + 1e-12})
    t1 = sm.execution_statistics.duration
    assert t0 == t1
    sm.execute({"x_shared": xs + 0.1})
    t2 = sm.execution_statistics.duration
    assert t2 > t1

    sm.execution_statistics.duration = 1.0
    assert sm.execution_statistics.duration == 1.0


def test_cache_h5(tmp_wd) -> None:
    """Test the HDF5 cache."""
    sm = SobieskiMission(enable_delay=0.1)
    hdf_file = sm.name + ".hdf5"
    sm.set_cache(sm.CacheType.HDF5, hdf_file_path=hdf_file)
    xs = sm.io.input_grammar.defaults["x_shared"]
    sm.execute({"x_shared": xs})
    t0 = sm.execution_statistics.duration
    sm.execute({"x_shared": xs})
    assert t0 == sm.execution_statistics.duration
    sm.cache.tolerance = 1e-6
    t0 = sm.execution_statistics.duration
    sm.execute({"x_shared": xs + 1e-12})
    assert t0 == sm.execution_statistics.duration
    sm.execute({"x_shared": xs + 1e12})
    assert t0 != sm.execution_statistics.duration
    # Read again the hashes
    sm.cache = HDF5Cache(hdf_file_path=hdf_file, hdf_node_path=sm.name)

    with pytest.raises(ImportError):
        sm.set_cache("toto")


def test_cache_h5_inpts(tmp_wd) -> None:
    """Test the HD5 cache for inputs."""
    sm = SobieskiMission()
    hdf_file = sm.name + ".hdf5"
    sm.set_cache(sm.CacheType.HDF5, hdf_file_path=hdf_file)
    xs = sm.io.input_grammar.defaults["x_shared"]
    sm.execute({"x_shared": xs})
    out_ref = sm.io.data["y_4"]
    sm.execute({"x_shared": xs + 1.0})
    sm.execute({"x_shared": xs})
    assert (sm.io.data["x_shared"] == xs).all()
    assert (sm.io.data["y_4"] == out_ref).all()


def test_cache_memory_inpts() -> None:
    """Test the CacheType.MEMORY_FULL."""
    sm = SobieskiMission()
    sm.set_cache(sm.CacheType.MEMORY_FULL)
    xs = sm.io.input_grammar.defaults["x_shared"]
    sm.execute({"x_shared": xs})
    out_ref = sm.io.data["y_4"]
    sm.execute({"x_shared": xs + 1.0})
    sm.execute({"x_shared": xs})
    assert (sm.io.data["x_shared"] == xs).all()
    assert (sm.io.data["y_4"] == out_ref).all()


def test_cache_h5_jac(tmp_wd) -> None:
    """Test the HDF5 cache for the Jacobian."""
    sm = SobieskiMission()
    hdf_file = sm.name + ".hdf5"
    sm.set_cache(sm.CacheType.HDF5, hdf_file_path=hdf_file)
    xs = sm.io.input_grammar.defaults["x_shared"]
    input_data = {"x_shared": xs}
    jac_1 = sm.linearize(input_data, compute_all_jacobians=True)
    sm.execute(input_data)
    jac_2 = sm.linearize(input_data, compute_all_jacobians=True)
    assert check_jac_equals(jac_1, jac_2)

    input_data = {"x_shared": xs + 2.0}
    sm.execute(input_data)
    jac_1 = sm.linearize(input_data, compute_all_jacobians=True, execute=False)

    input_data = {"x_shared": xs + 3.0}
    jac_2 = sm.linearize(input_data, compute_all_jacobians=True)
    assert not check_jac_equals(jac_1, jac_2)

    sm.execute(input_data)
    jac_3 = sm.linearize(input_data, compute_all_jacobians=True)
    assert check_jac_equals(jac_3, jac_2)

    jac_4 = sm.linearize(input_data, compute_all_jacobians=True, execute=False)
    assert check_jac_equals(jac_3, jac_4)

    sm.cache = HDF5Cache(hdf_file_path=hdf_file, hdf_node_path=sm.name)


def test_replace_h5_cache(tmp_wd) -> None:
    """Check that changing the HDF5 cache is correctly taken into account."""
    sm = SobieskiMission()
    hdf_file_1 = sm.name + "_1.hdf5"
    hdf_file_2 = sm.name + "_2.hdf5"
    sm.set_cache(sm.CacheType.HDF5, hdf_file_path=hdf_file_1)
    sm.set_cache(sm.CacheType.HDF5, hdf_file_path=hdf_file_2)
    assert sm.cache.hdf_file.hdf_file_path == hdf_file_2


def test_cache_run_and_linearize() -> None:
    """Check that the cache is filled with the Jacobian during linearization."""
    sm = SobieskiMission()
    run_orig = sm._run

    def run_and_lin(input_data) -> None:
        output_data = run_orig(input_data)
        sm._compute_jacobian()
        sm._has_jacobian = True
        return output_data

    sm._run = run_and_lin
    sm.set_cache(sm.CacheType.SIMPLE)
    sm.execute()
    assert sm.cache[sm.io.input_grammar.defaults].jacobian is not None

    sm.linearize()
    # Cache must be loaded
    assert sm.execution_statistics.n_linearizations == 0


def test_jac_approx_mix_fd() -> None:
    """Check the complex step method with parallel=True."""
    sm = SobieskiMission()
    sm.set_jacobian_approximation(
        sm.ApproximationMode.COMPLEX_STEP,
        jax_approx_step=1e-30,
        jac_approx_n_processes=4,
    )
    assert sm.check_jacobian(parallel=True, n_processes=4, threshold=1e-4)


def test_jac_set_optimal_fd_step_compute_all_jacobians() -> None:
    """Test the computation of the optimal time step with compute_all_jacobians=True."""
    sm = SobieskiMission()
    sm.set_jacobian_approximation()
    sm.set_optimal_fd_step(compute_all_jacobians=True)
    assert sm.check_jacobian(n_processes=1, threshold=1e-4)


def test_jac_set_optimal_fd_step_input_output() -> None:
    """Test the computation of the optimal time step with compute_all_jacobians=True."""
    sm = SobieskiMission()
    sm.set_jacobian_approximation()
    sm.set_optimal_fd_step(input_names=["y_14"], output_names=["y_4"])
    assert sm.check_jacobian(n_processes=1, threshold=1e-4)


def test_jac_set_optimal_fd_step_no_jac_approx() -> None:
    """Test that the optimal_fd_step cannot be called before settijng the approx
    method."""
    sm = SobieskiMission()
    msg = "set_jacobian_approximation must be called before setting an optimal step"
    with pytest.raises(ValueError, match=msg):
        sm.set_optimal_fd_step(compute_all_jacobians=True)


def test_jac_cache_trigger_shapecheck() -> None:
    """Test the check of cache shape."""
    # if cache is loaded and jacobian has already been computed for given i/o
    # and jacobian is called again but with new i/o
    # it will compute the jacobian with the new i/o
    aero = SobieskiAerodynamics("complex128")
    inpts = aero.io.input_grammar.defaults
    aero.linearization_mode = aero.ApproximationMode.FINITE_DIFFERENCES
    in_names = ["x_2", "y_12"]
    aero.add_differentiated_inputs(in_names)
    out_names = ["y_21"]
    aero.add_differentiated_outputs(out_names)
    aero.linearize(inpts)

    in_names = ["y_32", "x_shared"]
    out_names = ["g_2"]
    aero._has_jacobian = True
    aero.add_differentiated_inputs(in_names)
    aero.add_differentiated_outputs(out_names)
    aero.linearize(inpts, execute=False)


def test_has_jacobian() -> None:
    """Test that Discipline can be linearized."""
    # Test at the jacobian is not computed if _has_jacobian is
    # set to true by the discipline
    aero = SobieskiAerodynamics()
    aero.execute()
    aero.linearize(compute_all_jacobians=True)
    assert aero.execution_statistics.n_executions == 1
    assert aero.execution_statistics.n_linearizations == 1
    del aero

    class Aero2(SobieskiAerodynamics):
        def _run(self, input_data: StrKeyMapping):
            output_data = super()._run(input_data)
            self._compute_jacobian(self.io.input_grammar, self.io.output_grammar)
            self._has_jacobian = True
            return output_data

    aero2 = Aero2()
    aero2.execute()
    aero2.linearize(compute_all_jacobians=True)
    assert aero2.execution_statistics.n_executions == 1
    assert aero2.execution_statistics.n_linearizations == 0


def test_init_jacobian_with_incorrect_type() -> None:
    """Test the initialization of the jacobian matrix with incorrect type."""

    def myfunc(x=1.0, y=2.0):
        z = x + y
        return z  # noqa: RET504

    disc = AutoPyDiscipline(myfunc)

    with pytest.raises(ValueError):
        disc._init_jacobian(init_type="foo")


@pytest.mark.parametrize("init_method", Discipline.InitJacobianType)
@pytest.mark.parametrize("fill_missing_keys", [True, False])
def test_init_jacobian(init_method, fill_missing_keys) -> None:
    """Test the initialization of the jacobian matrix."""

    def myfunc(x=1.0, y=2.0):
        z = x + y
        return z  # noqa: RET504

    disc = AutoPyDiscipline(myfunc)

    disc.execute()
    disc._init_jacobian(
        output_names=["z"],
        input_names=["x"],
        init_type=init_method,
        fill_missing_keys=fill_missing_keys,
    )

    if not fill_missing_keys:
        assert not disc.jac["z"].get("y", None)

    if init_method == "empty":
        assert isinstance(disc.jac["z"]["x"], ndarray)
    elif init_method == "dense":
        assert isinstance(disc.jac["z"]["x"], ndarray)
        assert norm(disc.jac["z"]["x"]) == 0.0
    elif init_method == "sparse":
        assert isinstance(disc.jac["z"]["x"], sparse_classes)
        assert disc.jac["z"]["x"].size == 0


def test_repr_str() -> None:
    """Test the representation of a Discipline."""

    def myfunc(x=1, y=2):
        z = x + y
        return z  # noqa: RET504

    disc = AutoPyDiscipline(myfunc)
    assert str(disc) == "myfunc"
    assert repr(disc) == "myfunc\n   Inputs: x, y\n   Outputs: z"


def test_activate_counters() -> None:
    """Check that the discipline counters are active by default."""

    discipline = DummyDiscipline()
    assert discipline.execution_statistics.n_executions == 0
    assert discipline.execution_statistics.n_linearizations == 0
    assert discipline.execution_statistics.duration == 0

    discipline.execute()
    assert discipline.execution_statistics.n_executions == 1
    assert discipline.execution_statistics.n_linearizations == 0
    assert discipline.execution_statistics.duration > 0


def test_deactivate_counters() -> None:
    """Check that the discipline counters are set to None when deactivated."""
    activate_counters = ExecutionStatistics.is_enabled

    ExecutionStatistics.is_enabled = False

    discipline = DummyDiscipline()
    assert discipline.execution_statistics.n_executions is None
    assert discipline.execution_statistics.n_linearizations is None
    assert discipline.execution_statistics.duration is None

    discipline.execute()
    assert discipline.execution_statistics.n_executions is None
    assert discipline.execution_statistics.n_linearizations is None
    assert discipline.execution_statistics.duration is None

    match = "The execution statistics of the object named DummyDiscipline are disabled."
    with pytest.raises(RuntimeError, match=match):
        discipline.execution_statistics.n_executions = 1

    with pytest.raises(RuntimeError, match=match):
        discipline.execution_statistics.n_linearizations = 1

    with pytest.raises(RuntimeError, match=match):
        discipline.execution_statistics.duration = 1

    ExecutionStatistics.is_enabled = activate_counters


def test_cache_none() -> None:
    """Check that the discipline cache can be deactivated."""
    cache_type_before = Discipline.default_cache_type
    Discipline.default_cache_type = Discipline.CacheType.NONE
    discipline = DummyDiscipline()
    assert discipline.cache is None
    discipline.execute()
    assert BaseMDA.default_cache_type is BaseMDA.CacheType.SIMPLE
    Discipline.default_cache_type = cache_type_before


def test_grammar_inheritance() -> None:
    """Check that disciplines based on JSON grammar files inherit these files."""

    class NewSellar1(Sellar1):
        """A discipline whose parent uses IO grammar files."""

    # The discipline works correctly as the parent class has IO grammar files.
    discipline = NewSellar1()
    assert "x_1" in discipline.io.input_grammar


# @pytest.mark.parametrize(
#     ("grammar_directory", "comp_dir", "in_or_out", "expected"),
#     [
#         (
#             None,
#             None,
#             "in",
#             Path(sys.modules[Sellar1.__module__].__file__).parent.absolute()
#             / "foo_input.json",
#         ),
#         (None, "instance_gd", "out", Path("instance_gd") / "foo_output.json"),
#         ("class_gd", None, "out", Path("class_gd") / "foo_output.json"),
#         ("class_gd", "instance_gd", "out", Path("instance_gd") / "foo_output.json"),
#     ],
# )
# def test_get_grammar_file_path(
#     grammar_directory, comp_dir, in_or_out, expected
# ) -> None:
#     """Check the grammar file path."""
#     original_grammar_directory = Sellar1.GRAMMAR_DIRECTORY
#     Sellar1.GRAMMAR_DIRECTORY = grammar_directory
#     get_grammar_file_path = Discipline._Discipline
#     path = get_grammar_file_path(Sellar1, comp_dir, in_or_out, "foo")
#     assert path == expected
#     Sellar1.GRAMMAR_DIRECTORY = original_grammar_directory
#

# def test_residuals_fail():
#     """Tests the check of residual variables with state_equations_are_solved=False."""
#     disc = SobieskiMission()
#     disc.io.residual_to_state_variable = {"y_4": "x_shared"}
#     with pytest.raises(
#         RuntimeError,
#         match="Discipline that do not solve their residuals "
#         "are not supported yet.",
#     ):
#         disc.execute()


def test_activate_checks() -> None:
    out_ref = SobieskiMission().execute()["y_4"]
    SobieskiMission.validate_input_data = False
    SobieskiMission.validate_output_data = False
    disc = SobieskiMission()
    assert out_ref == disc.execute()["y_4"]
    SobieskiMission.validate_input_data = True
    SobieskiMission.validate_output_data = True


def test_no_cache() -> None:
    disc = SobieskiMission()
    disc.execute()
    disc.execute()
    assert disc.execution_statistics.n_executions == 1

    disc = DummyDiscipline()
    disc.cache = None
    disc.execute()
    disc.execute()
    assert disc.execution_statistics.n_executions == 2


@pytest.mark.parametrize(
    (
        "inputs",
        "outputs",
        "grammar_type",
        "expected_diff_inputs",
        "expected_diff_outputs",
    ),
    [
        (
            {"x": array([1.0]), "in_path": array(["some_string"])},
            {"y": array([0.0]), "out_path": array(["another_string"])},
            Discipline.GrammarType.JSON,
            ["x"],
            ["y"],
        ),
        (
            {"x": array([1.0]), "in_path": "some_string"},
            {"y": array([0.0]), "out_path": "another_string"},
            Discipline.GrammarType.JSON,
            ["x"],
            ["y"],
        ),
    ],
)
def test_add_differentiated_io_non_numeric(
    inputs, outputs, grammar_type, expected_diff_inputs, expected_diff_outputs
) -> None:
    """Check that non-numeric i/o are ignored in add_differentiated_inputs/outputs.

    If the discipline grammar type is :attr:`.Discipline.GrammarType.JSON` and
    an input/output is either a non-numeric array or not an array, it will be ignored.

    If the discipline grammar type is :attr:`.Discipline.GrammarType.SIMPLE` and
    an input/output is not an array, it will be ignored. Keep in mind that in this case
    the array subtype is not checked.

    Args:
        inputs: The inputs of the discipline.
        outputs: The outputs of the discipline.
        grammar_type: The discipline grammar type.
        expected_diff_inputs: The expected differentiated inputs.
        expected_diff_outputs: The expected differentiated outputs.
    """
    discipline = DummyDiscipline()
    discipline.io.input_grammar.update_from_data(inputs)
    discipline.io.output_grammar.update_from_data(outputs)
    discipline.add_differentiated_inputs()
    discipline.add_differentiated_outputs()
    assert discipline._differentiated_input_names == expected_diff_inputs
    assert discipline._differentiated_output_names == expected_diff_outputs


def test_hdf5cache_twice(tmp_wd, caplog) -> None:
    """Check what happens when the cache policy is set twice at HDF5Cache."""
    discipline = DummyDiscipline()
    discipline.set_cache("HDF5Cache", hdf_file_path="cache.hdf", hdf_node_path="foo")
    cache_id = id(discipline.cache)

    discipline.set_cache("HDF5Cache", hdf_file_path="cache.hdf", hdf_node_path="foo")
    assert id(discipline.cache) == cache_id
    _, log_level, log_message = caplog.record_tuples[0]
    assert log_level == logging.WARNING
    assert log_message == (
        "The cache policy is already set to HDF5Cache "
        "with the file path 'cache.hdf' and node name 'foo'; "
        "call discipline.cache.clear() to clear the cache."
    )

    discipline.set_cache("HDF5Cache", hdf_file_path="cache.hdf", hdf_node_path="bar")
    assert id(discipline.cache) != cache_id


class Observer:
    """This class will record the successive statuses a discipline will be in."""

    statuses: list[str]

    def __init__(self) -> None:
        self.statuses = []

    def update_status(self, execution_status: ExecutionStatus) -> None:
        self.statuses.append(execution_status.value)

    def reset(self) -> None:
        self.statuses.clear()


@pytest.fixture
def observer() -> Observer:
    return Observer()


@pytest.mark.xfail
def test_statuses(observer) -> None:
    """Verify the successive status."""
    disc = Sellar1()
    disc.execution_status.add_observer(observer)

    assert not observer.statuses

    disc.execution_status.value = Status.DONE
    assert observer.statuses == [Status.DONE]
    observer.reset()

    disc.execute()
    assert observer.statuses == [
        Status.RUNNING,
        Status.DONE,
    ]
    observer.reset()

    disc.linearize(compute_all_jacobians=True)
    assert observer.statuses == [
        Status.DONE,
        Status.LINEARIZING,
        Status.DONE,
    ]
    observer.reset()

    class KODisc(Discipline):
        def _run(self, input_data: StrKeyMapping):
            1 / 0  # noqa: B018

    disc = KODisc()
    disc.execution_status.add_observer(observer)
    with contextlib.suppress(Exception):
        disc.execute({})
    assert observer.statuses == [
        Status.RUNNING,
        Status.FAILED,
    ]


def test_statuses_linearize(observer) -> None:
    """Verify the successive status for linearize alone."""
    disc = Sellar1()
    disc.execution_status.add_observer(observer)

    disc.linearize(compute_all_jacobians=True)
    assert observer.statuses == [
        Status.RUNNING,
        Status.DONE,
        Status.LINEARIZING,
        Status.DONE,
    ]
    observer.reset()


@pytest.fixture(scope="module")
def self_coupled_disc() -> Discipline:
    """A minimalist self-coupled discipline, where the self-coupled variable is
    multiplied by two."""
    disc = AnalyticDiscipline({"x": "2*x", "y": "x"})
    disc.io.input_grammar.defaults["x"] = array([1])
    return disc


@pytest.mark.parametrize(
    ("name", "group", "value"),
    [
        ("x", "inputs", array([1])),
        ("x", "outputs", array([2])),
    ],
)
def test_self_coupled(self_coupled_disc, name, group, value) -> None:
    """Check that the value of each variable is equal to the prescribed value, and that
    each variable belongs to the prescribed group."""
    self_coupled_disc.execute()
    d = self_coupled_disc.cache.to_dataset()
    assert (
        d.get_view(variable_names=name, group_names=group).to_numpy() == value
    ).all()
    assert (
        d.get_view(variable_names=name, group_names=group).get_group_names(name)[0]
        == group
    )
    assert group in d.get_group_names(name)
    assert len(d.get_group_names(name)) > 1


def test_virtual_exe() -> None:
    """Tests the discipline virtual execution."""
    disc_1 = DummyDiscipline("d1")
    disc_1.io.input_grammar.update_from_names(["x"])
    disc_1.io.input_grammar.defaults = {"x": ones([1])}
    disc_1.io.output_grammar.update_from_names(["y"])
    disc_1.io.output_grammar.defaults = {"y": ones([1])}

    disc_1.status = Status.DONE
    disc_1.virtual_execution = True

    disc_1.execute()

    assert disc_1.io.data["y"] == ones([1])

    # Test with missing defaults
    disc_1.default_output_data.clear()
    # Ok with the cache
    disc_1.execute()

    disc_1.cache.clear()
    with pytest.raises(InvalidDataError, match=re.escape("Missing required names: y.")):
        disc_1.execute()


class DisciplineWithPaths(Discipline):
    default_grammar_type = Discipline.GrammarType.SIMPLE

    def __init__(self) -> None:
        super().__init__()
        self.io.input_grammar.update_from_types({"path": Path})
        self.io.output_grammar.update_from_types({"out_path": Path})
        self.local_path = Path()

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        self.io.data["out_path"] = self.io.data["path"]


def __is_path_correct(local_path: (Path, PurePosixPath, PureWindowsPath)) -> None:
    """Test the types of paths depending on the platform.

    Args:
        local_path: The path to test.

    Raises:
        AssertionError: if the path type is invalid.
    """
    if platform.platform().startswith("Windows"):
        assert isinstance(local_path, PureWindowsPath)
    else:
        assert isinstance(local_path, PurePosixPath)


def test_path_serialization(tmp_path) -> None:
    """Test the serialization of Paths in disciplines."""
    discipline = DisciplineWithPaths()
    disc_path = tmp_path / "mydisc.pckl"
    discipline.execute({"path": tmp_path})
    to_pickle(discipline, disc_path)
    deserialized = from_pickle(disc_path)

    assert isinstance(deserialized.local_path, Path)
    assert isinstance(deserialized.io.data["path"], Path)
    assert isinstance(deserialized.io.data["out_path"], Path)

    for local_path in [
        discipline.__getstate__()["local_path"],
    ]:
        __is_path_correct(local_path)

    data = deserialized.io.data
    __is_path_correct(data["path"])

    state = data.__getstate__()
    data.__setstate__(state)
    assert isinstance(data["path"], Path)


def test_repr_html() -> None:
    """Check Discipline._repr_html_."""
    assert AnalyticDiscipline(
        name="foo", expressions={"z": "b+a", "y": "c+d+e"}
    )._repr_html_() == REPR_HTML_WRAPPER.format(
        "foo<br/><ul><li>Inputs: a, b, c, d, e</li><li>Outputs: y, z</li></ul>"
    )


def test_create_cache_policy_to_none() -> None:
    """Check that set_cache can use CacheType.NONE as cache_type value."""
    discipline = DummyDiscipline()
    assert isinstance(discipline.cache, SimpleCache)
    discipline.set_cache(discipline.CacheType.NONE)
    assert discipline.cache is None
