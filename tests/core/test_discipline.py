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

import logging
import os
import sys
from pathlib import Path

import pytest
from gemseo.caches.hdf5_cache import HDF5Cache
from gemseo.core.chain import MDOChain
from gemseo.core.data_processor import ComplexDataProcessor
from gemseo.core.discipline import MDODiscipline
from gemseo.core.grammars.errors import InvalidDataException
from gemseo.core.grammars.json_grammar import JSONGrammar
from gemseo.core.scenario import Scenario
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.disciplines.auto_py import AutoPyDiscipline
from gemseo.mda.mda import MDA
from gemseo.problems.sellar.sellar import Sellar1
from gemseo.problems.sobieski._disciplines_sg import SobieskiStructureSG
from gemseo.problems.sobieski.core.problem import SobieskiProblem
from gemseo.problems.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.sobieski.disciplines import SobieskiMission
from gemseo.problems.sobieski.disciplines import SobieskiPropulsion
from gemseo.problems.sobieski.disciplines import SobieskiStructure
from numpy import allclose
from numpy import array
from numpy import complex128
from numpy import ndarray


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
    if not sorted(jac_1.keys()) == sorted(jac_2.keys()):
        return False
    for out, jac_dict in jac_1.items():
        if not sorted(jac_dict.keys()) == sorted(jac_2[out].keys()):
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
    chain = MDOChain(
        [
            SobieskiStructure(),
            SobieskiAerodynamics(),
            SobieskiPropulsion(),
            SobieskiMission(),
        ]
    )
    chain_inputs = chain.input_grammar.keys()
    indata = SobieskiProblem().get_default_inputs(names=chain_inputs)
    return chain, indata


def test_set_statuses(tmp_wd):
    """Test the setting of the statuses."""
    chain = MDOChain(
        [
            SobieskiAerodynamics(),
            SobieskiPropulsion(),
            SobieskiStructure(),
            SobieskiMission(),
        ]
    )
    chain.set_disciplines_statuses("FAILED")
    assert chain.disciplines[0].status == "FAILED"


def test_get_sub_disciplines():
    """Test the get_sub_disciplines method."""
    chain = MDOChain([SobieskiAerodynamics()])
    assert len(chain.disciplines[0].get_sub_disciplines()) == 0


def test_instantiate_grammars():
    """Test the instantiation of the grammars."""
    chain = MDOChain([SobieskiAerodynamics()])
    chain.disciplines[0]._instantiate_grammars(None, None, grammar_type="JSONGrammar")
    assert isinstance(chain.disciplines[0].input_grammar, JSONGrammar)


def test_execute_status_error(sobieski_chain):
    """Test the execution with a failed status."""
    chain, indata = sobieski_chain
    chain.set_disciplines_statuses("FAILED")
    with pytest.raises(Exception):
        chain.execute(indata)


def test_check_status_error(sobieski_chain):
    """Test the execution with a None status."""
    chain, _ = sobieski_chain
    with pytest.raises(Exception):
        chain._check_status("None")


def test_check_input_data_exception_chain(sobieski_chain):
    """Test the check input data exception."""
    chain, indata = sobieski_chain
    del indata["x_1"]
    with pytest.raises(InvalidDataException):
        chain.check_input_data(indata)


@pytest.mark.parametrize(
    "grammar_type", [MDODiscipline.JSON_GRAMMAR_TYPE, MDODiscipline.SIMPLE_GRAMMAR_TYPE]
)
def test_check_input_data_exception(grammar_type):
    """Test the check input data exception."""
    if grammar_type == MDODiscipline.SIMPLE_GRAMMAR_TYPE:
        struct = SobieskiStructureSG()
    else:
        struct = SobieskiStructure()

    struct_inputs = struct.input_grammar.keys()
    indata = SobieskiProblem().get_default_inputs(names=struct_inputs)
    del indata["x_1"]

    with pytest.raises(InvalidDataException, match=".*Missing required names: x_1"):
        struct.check_input_data(indata)

    struct.execute(indata)

    del struct.default_inputs["x_1"]
    with pytest.raises(InvalidDataException, match=".*Missing required names: x_1"):
        struct.execute(indata)


def test_outputs():
    """Test the execution of a MDODiscipline."""
    struct = SobieskiStructure()
    with pytest.raises(InvalidDataException):
        struct.check_output_data()
    indata = SobieskiProblem().get_default_inputs()
    struct.execute(indata)
    in_array = struct.get_inputs_asarray()
    assert len(in_array) == 13


def test_get_outputs_by_name_exception(sobieski_chain):
    """Test get_input_by_name with incorrect output var."""
    chain, indata = sobieski_chain
    chain.execute(indata)
    with pytest.raises(Exception):
        chain.get_outputs_by_name("toto")


def test_get_inputs_by_name_exception(sobieski_chain):
    """Test get_input_by_name with incorrect input var."""
    chain, _ = sobieski_chain
    with pytest.raises(Exception):
        chain.get_inputs_by_name("toto")


def test_get_input_data(sobieski_chain):
    """Test get_input_data."""
    chain, indata_ref = sobieski_chain
    chain.execute(indata_ref)
    indata = chain.get_input_data()
    assert sorted(indata.keys()) == sorted(indata_ref.keys())


def test_get_local_data_by_name_exception(sobieski_chain):
    """Test that an exception is raised when the var is not in the grammar."""
    chain, indata = sobieski_chain
    chain.execute(indata)
    with pytest.raises(Exception):
        chain.get_local_data_by_name("toto")


def test_reset_statuses_for_run_error(sobieski_chain):
    """Test the reset of the discipline status."""
    chain, _ = sobieski_chain
    chain.set_disciplines_statuses("FAILED")
    chain.reset_statuses_for_run()


def test_get_data_list_from_dict_error(sobieski_chain):
    """Test exception from get_data_list_from_dict."""
    _, indata = sobieski_chain
    with pytest.raises(TypeError):
        MDODiscipline.get_data_list_from_dict(2, indata)


def test_check_lin_error():
    """Test that an exception is raised when approx is inexistant."""
    aero = SobieskiAerodynamics()
    problem = SobieskiProblem()
    indata = problem.get_default_inputs(names=aero.get_input_data_names())
    with pytest.raises(Exception):
        aero.check_jacobian(indata, derr_approx="bidon")


def test_check_jac_fdapprox():
    """Test the finite difference approximation."""
    aero = SobieskiAerodynamics("complex128")
    inpts = aero.default_inputs
    aero.linearization_mode = aero.FINITE_DIFFERENCES
    aero.linearize(inpts, force_all=True)
    aero.check_jacobian(inpts)

    aero.linearization_mode = "auto"
    aero.check_jacobian(inpts)


def test_check_jac_csapprox():
    """Test the complex step approximation."""
    aero = SobieskiAerodynamics("complex128")
    aero.linearization_mode = aero.COMPLEX_STEP
    aero.linearize(force_all=True)
    aero.check_jacobian()


def test_check_jac_approx_plot(tmp_wd, pyplot_close_all):
    """Test the generation of the gradient plot."""
    aero = SobieskiAerodynamics()
    aero.linearize(force_all=True)
    file_path = "gradients_validation.pdf"
    aero.check_jacobian(step=10.0, plot_result=True, file_path=file_path)
    assert os.path.exists(file_path)


def test_check_lin_threshold():
    """Check the linearization threshold."""
    aero = SobieskiAerodynamics()
    problem = SobieskiProblem()
    indata = problem.get_default_inputs(names=aero.get_input_data_names())
    aero.check_jacobian(indata, threshold=1e-50)


def test_input_exist():
    """Test is_input_existing."""
    sr = SobieskiAerodynamics()
    problem = SobieskiProblem()
    indata = problem.get_default_inputs(names=sr.get_input_data_names())
    assert sr.is_input_existing(next(iter(indata.keys())))
    assert not sr.is_input_existing("bidon")


def test_get_all_inputs_outputs_name():
    """Test get_all_input_outputs_name method."""
    aero = SobieskiAerodynamics()
    problem = SobieskiProblem()
    indata = problem.get_default_inputs(names=aero.get_input_data_names())
    for data_name in indata:
        assert data_name in aero.get_input_data_names()


def test_get_all_inputs_outputs():
    """Test get all_inputs_outputs method."""
    aero = SobieskiAerodynamics()
    problem = SobieskiProblem()
    indata = problem.get_default_inputs(names=aero.get_input_data_names())
    aero.execute(indata)
    aero.get_all_inputs()
    aero.get_all_outputs()
    arr = aero.get_outputs_asarray()
    assert isinstance(arr, ndarray)
    assert len(arr) > 0
    arr = aero.get_inputs_asarray()
    assert isinstance(arr, ndarray)
    assert len(arr) > 0


def test_serialize_deserialize(tmp_wd):
    """Test the serialization/deserialization method."""
    aero = SobieskiAerodynamics()
    aero.data_processor = ComplexDataProcessor()
    out_file = "sellar1.o"
    input_data = SobieskiProblem().get_default_inputs()
    aero.execute(input_data)
    locd = aero.local_data
    aero.serialize(out_file)
    saero_u = MDODiscipline.deserialize(out_file)
    for k, v in locd.items():
        assert k in saero_u.local_data
        assert (v == saero_u.local_data[k]).all()

    def attr_list():
        return ["numpy_test"]

    aero.get_attributes_to_serialize = attr_list
    with pytest.raises(AttributeError):
        aero.serialize(out_file)

    saero_u_dict = saero_u.__dict__
    ok = True
    for k, _ in aero.__dict__.items():
        if k not in saero_u_dict and k != "get_attributes_to_serialize":
            ok = False
    assert ok


def test_serialize_run_deserialize(tmp_wd):
    """Test serialization, run and deserialization."""
    aero = SobieskiAerodynamics()
    out_file = "sellar1.o"
    input_data = SobieskiProblem().get_default_inputs()
    aero.serialize(out_file)
    saero_u = MDODiscipline.deserialize(out_file)
    saero_u.serialize(out_file)
    saero_u.execute(input_data)
    saero_u.serialize(out_file)
    saero_loc = MDODiscipline.deserialize(out_file)
    saero_loc.status = "PENDING"
    saero_loc.execute(input_data)

    for k, v in saero_loc.local_data.items():
        assert k in saero_u.local_data
        assert (v == saero_u.local_data[k]).all()


def test_serialize_hdf_cache(tmp_wd):
    """Test the serialization into a HDF5 cache."""
    aero = SobieskiAerodynamics()
    cache_hdf_file = "aero_cache.h5"
    aero.set_cache_policy(aero.HDF5_CACHE, cache_hdf_file=cache_hdf_file)
    aero.execute()
    out_file = "sob_aero.pckl"
    aero.serialize(out_file)
    saero_u = MDODiscipline.deserialize(out_file)
    assert saero_u.cache.last_entry.outputs["y_2"] is not None


def test_data_processor():
    """Test the data processor."""
    aero = SobieskiAerodynamics()
    input_data = SobieskiProblem().get_default_inputs()
    aero.data_processor = ComplexDataProcessor()
    out_data = aero.execute(input_data)
    for v in out_data.values():
        assert isinstance(v, ndarray)
        assert v.dtype == complex128
    # Mix data processor and cache
    out_data2 = aero.execute(input_data)
    for k, v in out_data.items():
        assert (out_data2[k] == v).all()


def test_diff_inputs_outputs():
    """Test the differentiation w.r.t inputs and outputs."""
    d = MDODiscipline()
    with pytest.raises(
        ValueError,
        match=f"Cannot differentiate the discipline {d.name} w.r.t. the inputs "
        r"that are not among the discipline inputs: \[\]",
    ):
        d.add_differentiated_inputs(["toto"])
    with pytest.raises(
        ValueError,
        match=f"Cannot differentiate the discipline {d.name} w.r.t. the outputs "
        r"that are not among the discipline outputs: \[\]",
    ):
        d.add_differentiated_outputs(["toto"])
    d.add_differentiated_inputs()


def test_run():
    """Test the execution of a abstract MDODiscipline."""
    d = MDODiscipline()
    with pytest.raises(NotImplementedError):
        d._run()


def test_load_default_inputs():
    """Test the load of the default inputs."""
    d = MDODiscipline()
    with pytest.raises(TypeError):
        d._filter_inputs(["toto"])
    notfailed = True
    try:
        d.default_inputs = ["toto"]
    except TypeError:
        notfailed = False
    if notfailed:
        raise Exception()


def test_linearize_errors():
    """Test the exceptions and errors during discipline linearization."""

    class LinDisc0(MDODiscipline):
        def __init__(self):
            super().__init__()

    LinDisc0()._compute_jacobian()

    class LinDisc(MDODiscipline):
        def __init__(self):
            super().__init__()
            self.input_grammar.update(["x"])
            self.output_grammar.update(["y"])

        def _run(self):
            self.local_data["y"] = array([2.0])

        def _compute_jacobian(self, inputs=None, outputs=None):
            self._init_jacobian()
            self.jac = {"y": {"x": array([0.0])}}

    d2 = LinDisc()
    d2.execute({"x": array([1.0])})
    # Shape is not 2D
    with pytest.raises(ValueError):
        d2.linearize({"x": array([1])}, force_all=True)

    with pytest.raises(ValueError):
        d2.__setattr__("linearization_mode", "toto")

    d2.local_data["y"] = 1
    with pytest.raises(ValueError):
        d2._check_jacobian_shape(["x"], ["y"])

    sm = SobieskiMission()

    def _compute_jacobian(inputs=None, outputs=None):
        SobieskiMission._compute_jacobian(sm, inputs=inputs, outputs=outputs)
        sm.jac["y_4"]["x_shared"] += 3.0

    sm._compute_jacobian = _compute_jacobian

    success = sm.check_jacobian(inputs=["x_shared"], outputs=["y_4"])
    assert not success


def test_check_jacobian_errors():
    """Test the errors raised during check_jacobian."""
    sm = SobieskiMission()
    with pytest.raises(ValueError):
        sm._check_jacobian_shape([], [])

    sm.execute()
    sm.linearize(force_all=True)
    sm._check_jacobian_shape(sm.get_input_data_names(), sm.get_output_data_names())
    sm.local_data.pop("x_shared")
    sm._check_jacobian_shape(sm.get_input_data_names(), sm.get_output_data_names())
    sm.local_data.pop("y_4")
    sm._check_jacobian_shape(sm.get_input_data_names(), sm.get_output_data_names())


def test_check_jacobian():
    """Test the check_jacobian method."""
    sm = SobieskiMission()
    sm.execute()
    sm._compute_jacobian()

    def _compute_jacobian(inputs=None, outputs=None):
        SobieskiMission._compute_jacobian(sm, inputs=inputs, outputs=outputs)
        del sm.jac["y_4"]

    sm._compute_jacobian = _compute_jacobian
    msg = f"The discipline {sm.name} was not linearized."
    with pytest.raises(ValueError, match=msg):
        sm.linearize(force_all=True)

    sm2 = SobieskiMission()

    def _compute_jacobian2(inputs=None, outputs=None):
        SobieskiMission._compute_jacobian(sm2, inputs=inputs, outputs=outputs)
        del sm2.jac["y_4"]["x_shared"]

    sm2._compute_jacobian = _compute_jacobian2
    with pytest.raises(KeyError):
        sm2.linearize(force_all=True)


def test_check_jacobian_2():
    """Test check_jacobian."""
    x = array([1.0, 2.0])

    class LinDisc(MDODiscipline):
        def __init__(self):
            super().__init__()
            self.input_grammar.update(["x"])
            self.output_grammar.update(["y"])
            self.default_inputs = {"x": x}
            self.jac_key = "x"
            self.jac_len = 2

        def _run(self):
            self.local_data["y"] = array([2.0])

        def _compute_jacobian(self, inputs=None, outputs=None):
            self._init_jacobian()
            self.jac = {"y": {self.jac_key: array([[0.0] * self.jac_len])}}

    disc = LinDisc()
    disc.jac_key = "z"
    with pytest.raises(KeyError):
        disc.linearize({"x": x}, force_all=True)
    disc.jac_key = "x"
    disc.jac_len = 3
    with pytest.raises(ValueError):
        disc.linearize({"x": x}, force_all=True)
    #         # Test not multiple d/dX
    disc.jac = {"y": {"x": array([[0.0], [1.0], [3.0]])}}
    with pytest.raises(ValueError):
        disc.linearize({"x": x}, force_all=True)
    #         # Test inconsistent output size for gradient
    #         # Test too small d/dX
    disc.jac = {"y": {"x": array([[0.0]])}}
    with pytest.raises(ValueError):
        disc.linearize({"x": x}, force_all=True)


@pytest.mark.skip_under_windows
def test_check_jacobian_parallel_fd():
    """Test check_jacobian in parallel."""
    sm = SobieskiMission()
    sm.check_jacobian(step=1e-6, threshold=1e-6, parallel=True, n_processes=6)


@pytest.mark.skip_under_windows
def test_check_jacobian_parallel_cplx():
    """Test check_jacobian in parallel with complex-step."""
    sm = SobieskiMission()
    sm.check_jacobian(
        derr_approx=sm.COMPLEX_STEP,
        step=1e-30,
        threshold=1e-6,
        parallel=True,
        n_processes=6,
    )


def test_execute_rerun_errors():
    """Test the execution and errors during re-run of MDODiscipline."""

    class MyDisc(MDODiscipline):
        def _run(self):
            self.local_data["b"] = array([1.0])

    d = MyDisc()
    d.input_grammar.update(["a"])
    d.output_grammar.update(["b"])
    d.execute({"a": [1]})
    d.status = d.STATUS_RUNNING
    with pytest.raises(ValueError):
        d.execute({"a": [2]})
    with pytest.raises(Exception):
        d.reset_statuses_for_run()

    d.status = d.STATUS_DONE
    d.execute({"a": [1]})
    d.re_exec_policy = d.RE_EXECUTE_NEVER_POLICY
    d.execute({"a": [1]})
    with pytest.raises(ValueError):
        d.execute({"a": [2]})


def test_cache():
    """Test the MDODiscipline cache."""
    sm = SobieskiMission(enable_delay=0.1)
    sm.cache_tol = 1e-6
    xs = sm.default_inputs["x_shared"]
    sm.execute({"x_shared": xs})
    t0 = sm.exec_time
    sm.execute({"x_shared": xs + 1e-12})
    t1 = sm.exec_time
    assert t0 == t1
    sm.execute({"x_shared": xs + 0.1})
    t2 = sm.exec_time
    assert t2 > t1

    sm.exec_time = 1.0
    assert sm.exec_time == 1.0


def test_cache_h5(tmp_wd):
    """Test the HDF5 cache."""
    sm = SobieskiMission(enable_delay=0.1)
    hdf_file = sm.name + ".hdf5"
    sm.set_cache_policy(sm.HDF5_CACHE, cache_hdf_file=hdf_file)
    xs = sm.default_inputs["x_shared"]
    sm.execute({"x_shared": xs})
    t0 = sm.exec_time
    sm.execute({"x_shared": xs})
    assert t0 == sm.exec_time
    sm.cache_tol = 1e-6
    t0 = sm.exec_time
    sm.execute({"x_shared": xs + 1e-12})
    assert t0 == sm.exec_time
    sm.execute({"x_shared": xs + 1e12})
    assert t0 != sm.exec_time
    # Read again the hashes
    sm.cache = HDF5Cache(hdf_file, sm.name)

    with pytest.raises(ImportError):
        sm.set_cache_policy(cache_type="toto")


def test_cache_h5_inpts(tmp_wd):
    """Test the HD5 cache for inputs."""
    sm = SobieskiMission()
    hdf_file = sm.name + ".hdf5"
    sm.set_cache_policy(sm.HDF5_CACHE, cache_hdf_file=hdf_file)
    xs = sm.default_inputs["x_shared"]
    sm.execute({"x_shared": xs})
    out_ref = sm.local_data["y_4"]
    sm.execute({"x_shared": xs + 1.0})
    sm.execute({"x_shared": xs})
    assert (sm.local_data["x_shared"] == xs).all()
    assert (sm.local_data["y_4"] == out_ref).all()


def test_cache_memory_inpts():
    """Test the MEMORY_FULL_CACHE."""
    sm = SobieskiMission()
    sm.set_cache_policy(sm.MEMORY_FULL_CACHE)
    xs = sm.default_inputs["x_shared"]
    sm.execute({"x_shared": xs})
    out_ref = sm.local_data["y_4"]
    sm.execute({"x_shared": xs + 1.0})
    sm.execute({"x_shared": xs})
    assert (sm.local_data["x_shared"] == xs).all()
    assert (sm.local_data["y_4"] == out_ref).all()


def test_cache_h5_jac(tmp_wd):
    """Test the HDF5 cache for the Jacobian."""
    sm = SobieskiMission()
    hdf_file = sm.name + ".hdf5"
    sm.set_cache_policy(sm.HDF5_CACHE, cache_hdf_file=hdf_file)
    xs = sm.default_inputs["x_shared"]
    input_data = {"x_shared": xs}
    jac_1 = sm.linearize(input_data, force_all=True)
    sm.execute(input_data)
    jac_2 = sm.linearize(input_data, force_all=True)
    assert check_jac_equals(jac_1, jac_2)

    input_data = {"x_shared": xs + 2.0}
    sm.execute(input_data)
    jac_1 = sm.linearize(input_data, force_all=True, force_no_exec=True)

    input_data = {"x_shared": xs + 3.0}
    jac_2 = sm.linearize(input_data, force_all=True)
    assert not check_jac_equals(jac_1, jac_2)

    sm.execute(input_data)
    jac_3 = sm.linearize(input_data, force_all=True)
    assert check_jac_equals(jac_3, jac_2)

    jac_4 = sm.linearize(input_data, force_all=True, force_no_exec=True)
    assert check_jac_equals(jac_3, jac_4)

    sm.cache = HDF5Cache(hdf_file, sm.name)


def test_replace_h5_cache(tmp_wd):
    """Check that changing the HDF5 cache is correctly taken into account."""
    sm = SobieskiMission()
    hdf_file_1 = sm.name + "_1.hdf5"
    hdf_file_2 = sm.name + "_2.hdf5"
    sm.set_cache_policy(sm.HDF5_CACHE, cache_hdf_file=hdf_file_1)
    sm.set_cache_policy(sm.HDF5_CACHE, cache_hdf_file=hdf_file_2)
    assert sm.cache.hdf_file.hdf_file_path == hdf_file_2


def test_cache_run_and_linearize():
    """Check that the cache is filled with the Jacobian during linearization."""
    sm = SobieskiMission()
    run_orig = sm._run

    def run_and_lin():
        run_orig()
        sm._compute_jacobian()
        sm._is_linearized = True

    sm._run = run_and_lin
    sm.set_cache_policy()
    sm.execute()
    assert sm.cache[sm.default_inputs].jacobian is not None

    sm.linearize()
    # Cache must be loaded
    assert sm.n_calls_linearize == 0


@pytest.mark.skip_under_windows
def test_jac_approx_mix_fd():
    """Check the complex step method with parallel=True."""
    sm = SobieskiMission()
    sm.set_jacobian_approximation(
        sm.COMPLEX_STEP, jax_approx_step=1e-30, jac_approx_n_processes=4
    )
    assert sm.check_jacobian(parallel=True, n_processes=4, threshold=1e-4)


def test_jac_set_optimal_fd_step_force_all():
    """Test the computation of the optimal time step with force_all=True."""
    sm = SobieskiMission()
    sm.set_jacobian_approximation()
    sm.set_optimal_fd_step(force_all=True)
    assert sm.check_jacobian(n_processes=1, threshold=1e-4)


def test_jac_set_optimal_fd_step_input_output():
    """Test the computation of the optimal time step with force_all=True."""
    sm = SobieskiMission()
    sm.set_jacobian_approximation()
    sm.set_optimal_fd_step(inputs=["y_14"], outputs=["y_4"])
    assert sm.check_jacobian(n_processes=1, threshold=1e-4)


def test_jac_set_optimal_fd_step_no_jac_approx():
    """Test that the optimal_fd_step cannot be called before settijng the approx
    method."""
    sm = SobieskiMission()
    msg = "set_jacobian_approximation must be called before setting an optimal step"
    with pytest.raises(ValueError, match=msg):
        sm.set_optimal_fd_step(force_all=True)


def test_jac_cache_trigger_shapecheck():
    """Test the check of cache shape."""
    # if cache is loaded and jacobian has already been computed for given i/o
    # and jacobian is called again but with new i/o
    # it will compute the jacobian with the new i/o
    aero = SobieskiAerodynamics("complex128")
    inpts = aero.default_inputs
    aero.linearization_mode = aero.FINITE_DIFFERENCES
    in_names = ["x_2", "y_12"]
    aero.add_differentiated_inputs(in_names)
    out_names = ["y_21"]
    aero.add_differentiated_outputs(out_names)
    aero.linearize(inpts)

    in_names = ["y_32", "x_shared"]
    out_names = ["g_2"]
    aero._cache_was_loaded = True
    aero.add_differentiated_inputs(in_names)
    aero.add_differentiated_outputs(out_names)
    aero.linearize(inpts, force_no_exec=True)


def test_is_linearized():
    """Test that MDODiscipline can be linearized."""
    # Test at the jacobian is not computed if _is_linearized is
    # set to true by the discipline
    aero = SobieskiAerodynamics()
    aero.execute()
    aero.linearize(force_all=True)
    assert aero.n_calls == 1
    assert aero.n_calls_linearize == 1
    del aero

    aero2 = SobieskiAerodynamics()
    aero_run = aero2._run
    aero_cjac = aero2._compute_jacobian

    def _run_and_jac():
        out = aero_run()
        aero_cjac(aero2.get_input_data_names(), aero2.get_output_data_names())
        aero2._is_linearized = True
        return out

    aero2._run = _run_and_jac

    aero2.execute()
    aero2.linearize(force_all=True)
    assert aero2.n_calls == 1
    assert aero2.n_calls_linearize == 0


def test_init_jacobian():
    """Test the initialization of the jacobian matrix."""

    def myfunc(x=1, y=2):
        z = x + y
        return z

    disc = AutoPyDiscipline(myfunc)
    disc.jac = {}
    disc.execute()
    disc._init_jacobian(outputs=["z"], fill_missing_keys=True)


def test_repr_str():
    """Test the representation of a MDODiscipline."""

    def myfunc(x=1, y=2):
        z = x + y
        return z

    disc = AutoPyDiscipline(myfunc)
    assert str(disc) == "myfunc"
    assert repr(disc) == "myfunc\n   Inputs: x, y\n   Outputs: z"


def test_activate_counters():
    """Check that the discipline counters are active by default."""
    discipline = MDODiscipline()
    assert discipline.n_calls == 0
    assert discipline.n_calls_linearize == 0
    assert discipline.exec_time == 0

    discipline._run = lambda: None
    discipline.execute()
    assert discipline.n_calls == 1
    assert discipline.n_calls_linearize == 0
    assert discipline.exec_time > 0


def test_deactivate_counters():
    """Check that the discipline counters are set to None when deactivated."""
    activate_counters = MDODiscipline.activate_counters

    MDODiscipline.activate_counters = False

    discipline = MDODiscipline()
    assert discipline.n_calls is None
    assert discipline.n_calls_linearize is None
    assert discipline.exec_time is None

    discipline._run = lambda: None
    discipline.execute()
    assert discipline.n_calls is None
    assert discipline.n_calls_linearize is None
    assert discipline.exec_time is None

    with pytest.raises(RuntimeError, match="The discipline counters are disabled."):
        discipline.n_calls = 1

    with pytest.raises(RuntimeError, match="The discipline counters are disabled."):
        discipline.n_calls_linearize = 1

    with pytest.raises(RuntimeError, match="The discipline counters are disabled."):
        discipline.exec_time = 1

    MDODiscipline.activate_counters = activate_counters


def test_cache_none():
    """Check that the discipline cache can be deactivate."""
    discipline = MDODiscipline(cache_type=None)
    assert discipline.activate_cache is True
    assert discipline.cache is None

    MDODiscipline.activate_cache = False
    discipline = MDODiscipline()
    assert discipline.cache is None

    discipline._run = lambda: None
    discipline.execute()

    assert MDA.activate_cache is True

    MDODiscipline.activate_cache = True


def test_grammar_inheritance():
    """Check that disciplines based on JSON grammar files inherit these files."""

    class NewSellar1(Sellar1):
        """A discipline whose parent uses IO grammar files."""

    # The discipline works correctly as the parent class has IO grammar files.
    discipline = NewSellar1()
    assert "x_local" in discipline.get_input_data_names()

    class NewScenario(Scenario):
        """A discipline whose parent forces its children to use IO grammar files."""

        def _init_algo_factory(self):
            pass

    # An error is raised as Scenario does not provide JSON grammar files.
    with pytest.raises(
        FileNotFoundError, match="The grammar file NewScenario_input.json is missing."
    ):
        NewScenario([discipline], "MDF", "y_1", "design_space_mock")


@pytest.mark.parametrize(
    "grammar_directory,comp_dir,in_or_out,expected",
    [
        (
            None,
            None,
            "in",
            Path(sys.modules[Sellar1.__module__].__file__).parent.absolute()
            / "foo_input.json",
        ),
        (None, "instance_gd", "out", Path("instance_gd") / "foo_output.json"),
        ("class_gd", None, "out", Path("class_gd") / "foo_output.json"),
        ("class_gd", "instance_gd", "out", Path("instance_gd") / "foo_output.json"),
    ],
)
def test_get_grammar_file_path(grammar_directory, comp_dir, in_or_out, expected):
    """Check the grammar file path."""
    original_grammar_directory = Sellar1.GRAMMAR_DIRECTORY
    Sellar1.GRAMMAR_DIRECTORY = grammar_directory
    get_grammar_file_path = MDODiscipline._MDODiscipline__get_grammar_file_path
    path = get_grammar_file_path(Sellar1, comp_dir, in_or_out, "foo")
    assert path == expected
    Sellar1.GRAMMAR_DIRECTORY = original_grammar_directory


def test_residuals_fail():
    """Tests the check of residual variables with run_solves_residuals=False."""
    disc = SobieskiMission()
    disc.residual_variables = {"y_4": "x_shared"}
    with pytest.raises(
        RuntimeError,
        match="Disciplines that do not solve their residuals are not supported yet.",
    ):
        disc.execute()


def test_activate_checks():
    out_ref = SobieskiMission().execute()["y_4"]
    disc = SobieskiMission()
    disc.activate_input_data_check = False
    disc.activate_output_data_check = False
    assert out_ref == disc.execute()["y_4"]


def test_no_cache():
    disc = SobieskiMission()
    disc.execute()
    disc.execute()
    assert disc.n_calls == 1

    disc = SobieskiMission()
    disc.cache = None
    disc.execute()
    disc.execute()
    assert disc.n_calls == 2

    with pytest.raises(ValueError, match="does not have a cache"):
        disc.cache_tol

    with pytest.raises(ValueError, match="does not have a cache"):
        disc.cache_tol = 1.0


@pytest.mark.parametrize(
    "recursive, expected", [(False, {"d1", "d2", "chain1"}), (True, {"d1", "d2", "d3"})]
)
def test_get_sub_disciplines_recursive(recursive, expected):
    """Test the recursive option of get_sub_disciplines.

    Args:
        recursive: Whether to list sub-disciplines recursively.
        expected: The expected disciplines.
    """
    d1 = MDODiscipline("d1")
    d2 = MDODiscipline("d2")
    d3 = MDODiscipline("d3")
    chain1 = MDOChain([d3], "chain1")
    chain2 = MDOChain([d2, chain1], "chain2")
    chain3 = MDOChain([d1, chain2], "chain3")

    classes = [
        discipline.name
        for discipline in chain3.get_sub_disciplines(recursive=recursive)
    ]

    assert set(classes) == expected


@pytest.mark.parametrize(
    "inputs, outputs, grammar_type, expected_diff_inputs, expected_diff_outputs",
    [
        (
            {"x": array([1.0]), "in_path": "some_string"},
            {"y": array([0.0]), "out_path": "another_string"},
            MDODiscipline.SIMPLE_GRAMMAR_TYPE,
            ["x"],
            ["y"],
        ),
        (
            {"x": array([1]), "in_path": "some_string"},
            {"y": 1, "out_path": "another_string"},
            MDODiscipline.SIMPLE_GRAMMAR_TYPE,
            ["x"],
            [],
        ),
        (
            {"x": array([1.0]), "in_path": array(["some_string"])},
            {"y": array([0.0]), "out_path": array(["another_string"])},
            MDODiscipline.JSON_GRAMMAR_TYPE,
            ["x"],
            ["y"],
        ),
        (
            {"x": array([1.0]), "in_path": "some_string"},
            {"y": array([0.0]), "out_path": "another_string"},
            MDODiscipline.JSON_GRAMMAR_TYPE,
            ["x"],
            ["y"],
        ),
        (
            {"x": array([1]), "in_path": "some_string"},
            {"y": 1, "out_path": "another_string"},
            MDODiscipline.JSON_GRAMMAR_TYPE,
            ["x"],
            [],
        ),
    ],
)
def test_add_differentiated_io_non_numeric(
    inputs, outputs, grammar_type, expected_diff_inputs, expected_diff_outputs
):
    """Check that non-numeric i/o are ignored in add_differentiated_inputs/outputs.

    If the discipline grammar type is :attr:`.MDODiscipline.JSON_GRAMMAR_TYPE` and
    an input/output is either a non-numeric array or not an array, it will be ignored.

    If the discipline grammar type is :attr:`.MDODiscipline.SIMPLE_GRAMMAR_TYPE` and
    an input/output is not an array, it will be ignored. Keep in mind that in this case
    the array subtype is not checked.

    Args:
        inputs: The inputs of the discipline.
        outputs: The outputs of the discipline.
        grammar_type: The discipline grammar type.
        expected_diff_inputs: The expected differentiated inputs.
        expected_diff_outputs: The expected differentiated outputs.
    """
    discipline = MDODiscipline(grammar_type=grammar_type)
    discipline.input_grammar.update_from_data(inputs)
    discipline.output_grammar.update_from_data(outputs)
    discipline.add_differentiated_inputs()
    discipline.add_differentiated_outputs()
    assert discipline._differentiated_inputs == expected_diff_inputs
    assert discipline._differentiated_outputs == expected_diff_outputs


def test_hdf5cache_twice(tmp_wd, caplog):
    """Check what happens when the cache policy is set twice at HDF5Cache."""
    discipline = MDODiscipline()
    discipline.set_cache_policy(
        "HDF5Cache", cache_hdf_file="cache.hdf", cache_hdf_node_name="foo"
    )
    cache_id = id(discipline.cache)

    discipline.set_cache_policy(
        "HDF5Cache", cache_hdf_file="cache.hdf", cache_hdf_node_name="foo"
    )
    assert id(discipline.cache) == cache_id
    _, log_level, log_message = caplog.record_tuples[0]
    assert log_level == logging.WARNING
    assert log_message == (
        "The cache policy is already set to HDF5Cache "
        "with the file path 'cache.hdf' and node name 'foo'; "
        "call discipline.cache.clear() to clear the cache."
    )

    discipline.set_cache_policy(
        "HDF5Cache", cache_hdf_file="cache.hdf", cache_hdf_node_name="bar"
    )
    assert id(discipline.cache) != cache_id


class Observer:
    """This class will record the successive statuses a discipline will be in."""

    statuses: list[str]

    def __init__(self) -> None:
        self.statuses = []

    def update_status(self, disc: MDODiscipline) -> None:
        self.statuses.append(disc.status)

    def reset(self) -> None:
        self.statuses.clear()


@pytest.fixture()
def observer() -> Observer:
    return Observer()


def test_statuses(observer):
    """Verify the successive status."""
    disc = Sellar1()
    disc.add_status_observer(observer)

    assert not observer.statuses

    disc.reset_statuses_for_run()
    assert observer.statuses == [MDODiscipline.STATUS_PENDING]
    observer.reset()

    disc.execute()
    assert observer.statuses == [
        MDODiscipline.STATUS_RUNNING,
        MDODiscipline.STATUS_DONE,
    ]
    observer.reset()

    disc.linearize(force_all=True)
    assert observer.statuses == [
        MDODiscipline.STATUS_PENDING,
        MDODiscipline.STATUS_LINEARIZE,
        MDODiscipline.STATUS_DONE,
    ]
    observer.reset()

    disc._run = lambda x: 1 / 0
    try:
        disc.execute({"x_local": disc.local_data["x_local"] + 1.0})
    except Exception:
        pass
    assert observer.statuses == [
        MDODiscipline.STATUS_RUNNING,
        MDODiscipline.STATUS_FAILED,
    ]


def test_statuses_linearize(observer):
    """Verify the successive status for linearize alone."""
    disc = Sellar1()
    disc.add_status_observer(observer)

    disc.linearize(force_all=True)
    assert observer.statuses == [
        MDODiscipline.STATUS_PENDING,
        MDODiscipline.STATUS_RUNNING,
        MDODiscipline.STATUS_DONE,
        MDODiscipline.STATUS_LINEARIZE,
        MDODiscipline.STATUS_DONE,
    ]
    observer.reset()


@pytest.fixture(scope="module")
def self_coupled_disc() -> MDODiscipline:
    """A minimalist self-coupled discipline, where the self-coupled variable is
    multiplied by two."""
    disc = AnalyticDiscipline({"x": "2*x", "y": "x"})
    disc.default_inputs["x"] = array([1])
    return disc


@pytest.mark.parametrize(
    "name, group, value",
    [
        ("x", "inputs", array([1])),
        ("x[out]", "outputs", array([2])),
    ],
)
def test_self_coupled(self_coupled_disc, name, group, value):
    """Check that the value of each variable is equal to the prescribed value, and that
    each variable belongs to the prescribed group."""
    self_coupled_disc.execute()
    d = self_coupled_disc.cache.export_to_dataset()
    assert allclose(d[name], value)
    assert d._groups[name] == group
