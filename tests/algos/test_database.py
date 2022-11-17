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
#    INITIAL AUTHORS - initial API and implementation and/or
#                        initial documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
#        :author: Benoit Pauwels - stacked data ; docstrings
from __future__ import annotations

from pathlib import Path

import h5py
import pytest
from gemseo.algos.database import Database
from gemseo.algos.database import HashableNdarray
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.algos.opt_result import OptimizationResult
from gemseo.problems.analytical.rosenbrock import Rosenbrock
from numpy import arange
from numpy import array
from numpy import ones
from numpy import string_
from numpy.linalg import norm
from numpy.testing import assert_almost_equal
from scipy.optimize import rosen
from scipy.optimize import rosen_der

DIRNAME = Path(__file__).parent
FAIL_HDF = DIRNAME / "fail.hdf5"


@pytest.fixture
def h5_file(tmp_wd):
    return h5py.File("test.h5", "w")


def rel_err(to_test, ref):
    """Returns the relative error.

    :param to_test: value to be tested
    :param ref: referece value
    """
    n_ref = norm(ref)
    err = norm(to_test - ref)
    if n_ref > 1e-14:
        return err / n_ref
    else:
        return err


@pytest.fixture
def problem_and_result() -> tuple[Rosenbrock, OptimizationResult]:
    """The Rosenbrock problem solved with L-BFGS-B and the optimization result."""
    rosenbrock = Rosenbrock()
    result = OptimizersFactory().execute(rosenbrock, "L-BFGS-B")
    return rosenbrock, result


@pytest.fixture
def problem(problem_and_result) -> Rosenbrock:
    """The Rosenbrock problem solved with L-BFGS-B."""
    return problem_and_result[0]


def test_correct_store_unstore(problem):
    """Test the storage of objective function values and gradient values."""
    database = problem.database
    fname = problem.objective.name
    for x_var in database.keys():
        x_var = x_var.unwrap()
        func_value = database.get_f_of_x(fname, x_var)
        gname = database.get_gradient_name(fname)
        func_rel_err = rel_err(func_value, rosen(x_var))
        assert_almost_equal(func_rel_err, 0.0, decimal=14)
        grad_value = database.get_f_of_x(gname, x_var)
        if grad_value is not None:
            grad_rel_err = rel_err(grad_value, rosen_der(x_var))
            assert_almost_equal(grad_rel_err, 0.0, decimal=14)


def test_write_read(tmp_wd, problem):
    """Test the writing of objective function values and gradient values."""
    database = problem.database
    outf = "rosen.hdf"
    database.export_hdf(outf)
    assert Path(outf).exists()
    fname = problem.objective.name

    loaded_db = Database(outf)

    for x_var in database.keys():
        x_var = x_var.unwrap()
        f_ref = database.get_f_of_x(fname, x_var)
        gname = database.get_gradient_name(fname)
        df_ref = database.get_f_of_x(gname, x_var)

        f_loaded = loaded_db.get_f_of_x(fname, x_var)

        f_rel_err = rel_err(f_ref, f_loaded)
        assert_almost_equal(f_rel_err, 0.0, decimal=14)

        df_loaded = loaded_db.get_f_of_x(gname, x_var)
        if df_ref is None:
            assert df_loaded is None
        else:
            df_rel_err = rel_err(df_ref, df_loaded)
            assert_almost_equal(df_rel_err, 0.0, decimal=14)


def test_set_item():
    """Tests setitem."""
    database = Database()
    k = array([1.0])
    #         Non ndarray key error
    with pytest.raises(TypeError):
        database.setdefault("toto", 1)
    hash_k = HashableNdarray(k)
    with pytest.raises(TypeError):
        database.setdefault(hash_k, k)
    database.setdefault(hash_k, {"f": 1})
    assert database.get(1.0) is None
    assert database.get(1.0) is None
    with pytest.raises(KeyError):
        database[1.0]
    database.get(k)
    database[k]


def test_set_wrong_item():
    """Tests setitem with wrong elements."""
    database = Database()
    msg = "Optimization history keys must be design variables numpy arrays"
    with pytest.raises(TypeError, match=msg):
        database["1"] = "toto"

    msg = "dictionary update sequence "
    with pytest.raises(ValueError, match=msg):
        database[array([1.0])] = "toto"


def test_set_hashable_ndarray():
    """Tests setitem with hashable ndarray."""
    database = Database()
    k = HashableNdarray(array([1.0]))
    database[k] = {}


def test_get_all_datanames():
    database = Database()
    fname = "f"
    gname = database.get_gradient_name(fname)
    database.store(array([1.0]), {fname: 1, gname: array([1.0])})

    assert database.get_all_data_names(True, True) == [fname]
    assert database.get_all_data_names(False, True) == [gname, fname]

    assert database.get_all_data_names() == [database.ITER_TAG, fname]
    assert database.get_all_data_names(False) == [
        gname,
        database.ITER_TAG,
        fname,
    ]


def test_get_f_hist():
    """Test the objective history extraction."""
    problem = Rosenbrock()
    database = problem.database
    problem.preprocess_functions()
    for x_vec in (array([0.0, 1.0]), array([1, 2.0])):
        problem.objective(x_vec)
        problem.objective.jac(x_vec)

    hist_x = database.get_x_history()
    fname = problem.objective.name
    hist_f = database.get_func_history(fname)
    gname = Database.get_gradient_name(fname)
    hist_g = database.get_func_history(gname)
    hist_g2 = database.get_func_grad_history(fname)
    assert len(hist_f) == 2
    assert len(hist_g) == 2
    assert (hist_g == hist_g2).all()
    assert len(hist_f) == len(hist_x)


def test_get_f_hist_rosen(problem):
    """Test the objective history extraction."""
    database = problem.database
    hist_x = database.get_x_history()
    fname = problem.objective.name
    hist_f = database.get_func_history(fname)
    gname = Database.get_gradient_name(fname)
    hist_g = database.get_func_history(gname)
    hist_g2 = database.get_func_grad_history(fname)
    assert len(hist_f) > 2
    assert (hist_g == hist_g2).all()
    assert len(hist_f) == len(hist_x)


def test_clean_from_iterate(problem):
    """Tests access to design variables by iteration index."""
    database = problem.database
    # clean after iterate 12
    database.clean_from_iterate(12)
    assert len(database) == 13
    # Add another point that cannot already exists
    x_test = array([10.0, -100.0])
    database.store(x_test, {})
    # Make sure that the iter tag is correct
    iter_id = int(database.get_f_of_x(Database.ITER_TAG, x_test)[0])
    assert iter_id, len(database)


def test_get_x_by_iter():
    """Tests access to design variables by iteration index."""
    problem = Rosenbrock()
    database = problem.database
    with pytest.raises(ValueError):
        database.get_x_by_iter(0)
    OptimizersFactory().execute(problem, "L-BFGS-B")
    hist_g2 = database.get_x_by_iter(20)
    assert database.get_index_of(hist_g2) == 20
    with pytest.raises(KeyError):
        database.get_index_of(array([123.456]))
    assert_almost_equal(hist_g2[0], 0.92396186, decimal=6)
    assert_almost_equal(hist_g2[1], 0.85259476, decimal=6)


def test_scipy_df0_rosenbrock(problem_and_result):
    """Tests the storage of optimization solutions."""
    problem, result = problem_and_result
    database = problem.database
    assert result.f_opt < 6.5e-11
    assert norm(database.get_x_history()[-1] - ones(2)) < 2e-5
    assert database.get_func_history(funcname="rosen")[-1] < 6.2e-11


def test_append_export(tmp_wd):
    database = Database()
    file_path_db = "test_db_append.hdf5"
    # Export empty file
    database.export_hdf(file_path_db)
    val = {"f": arange(2)}
    n_calls = 200
    for i in range(n_calls):
        database.store(array([i]), values_dict=val, add_iter=False)

    # Export again with append mode
    database.export_hdf(file_path_db, append=True)

    assert len(Database(file_path_db)) == n_calls

    i += 1
    database.store(array([i]), values_dict=val, add_iter=False)
    # Export again with append mode and check that it is much faster
    database.export_hdf(file_path_db, append=True)
    assert len(Database(file_path_db)) == n_calls + 1


def test_add_listeners():
    database = Database()
    with pytest.raises(TypeError, match="Listener function is not callable"):
        database.add_store_listener("toto")
    with pytest.raises(TypeError, match="Listener function is not callable"):
        database.add_new_iter_listener("toto")


def test_append_export_after_store(tmp_wd):
    """Test that a database is correctly exported when it is appended after each storage
    call."""
    database = Database()
    file_path_db = "test_db_append.hdf5"
    val1 = {"f": arange(2)}
    val2 = {"g": 10}
    val3 = {"@f": array([[100], [200]])}
    n_calls = 50
    for i in range(n_calls):
        idx = array([i, i + 1])
        database.store(idx, values_dict=val1)
        database.export_hdf(file_path_db, append=True)
        database.store(idx, values_dict=val2)
        database.export_hdf(file_path_db, append=True)
        database.store(idx, values_dict=val3)
        database.export_hdf(file_path_db, append=True)

    new_database = Database(file_path_db)

    n_calls = len(database)

    assert len(new_database) == n_calls

    for i in range(n_calls):
        idx = array([i, i + 1])
        for key, value in database[idx].items():
            assert value == pytest.approx(new_database[idx][key])


def test_create_hdf_input_dataset(h5_file):
    """Test that design variable values are correctly added to the hdf5 group of design
    variables."""
    database = Database()

    design_vars_grp = h5_file.require_group("x")

    input_val_1 = HashableNdarray(array([0, 1, 2]))
    database._add_hdf_input_dataset(0, design_vars_grp, input_val_1)
    assert array(design_vars_grp["0"]) == pytest.approx(input_val_1.unwrap())

    database._add_hdf_input_dataset(1, design_vars_grp, input_val_1)
    assert array(design_vars_grp["1"]) == pytest.approx(input_val_1.unwrap())

    input_val_2 = HashableNdarray(array([3, 4, 5]))
    with pytest.raises(ValueError):
        database._add_hdf_input_dataset(0, design_vars_grp, input_val_2)


def test_add_hdf_name_output(h5_file):
    """Test that output names are correctly added to the hdf5 group of output names."""
    database = Database()

    keys_group = h5_file.require_group("k")

    database._add_hdf_name_output(0, keys_group, ["f1"])
    assert array(keys_group["0"]) == array(["f1"], dtype=string_)

    database._add_hdf_name_output(0, keys_group, ["f2", "f3", "f4"])
    assert (
        array(keys_group["0"]) == array(["f1", "f2", "f3", "f4"], dtype=string_)
    ).all()

    database._add_hdf_name_output(1, keys_group, ["f2", "f3", "f4"])
    assert (array(keys_group["1"]) == array(["f2", "f3", "f4"], dtype=string_)).all()

    database._add_hdf_name_output(1, keys_group, ["@-y_1"])
    assert (
        array(keys_group["1"]) == array(["f2", "f3", "f4", "@-y_1"], dtype=string_)
    ).all()


def test_add_hdf_scalar_output(h5_file):
    """Test that scalar values are correctly added to the group of output values."""
    database = Database()

    values_group = h5_file.require_group("v")

    database._add_hdf_scalar_output(0, values_group, [10])
    assert array(values_group["0"]) == pytest.approx(array([10]))

    database._add_hdf_scalar_output(0, values_group, [20])
    assert array(values_group["0"]) == pytest.approx(array([10, 20]))

    database._add_hdf_scalar_output(0, values_group, [30, 40, 50, 60])
    assert array(values_group["0"]) == pytest.approx(array([10, 20, 30, 40, 50, 60]))

    database._add_hdf_scalar_output(1, values_group, [100, 200])
    assert array(values_group["1"]) == pytest.approx(array([100, 200]))


def test_add_hdf_vector_output(h5_file):
    """Test that a vector (array and/or list) of outputs is correctly added to the group
    of output values."""
    database = Database()

    values_group = h5_file.require_group("v")

    database._add_hdf_vector_output(0, 0, values_group, [10, 20, 30])
    assert array(values_group["arr_0"]["0"]) == pytest.approx(array([10, 20, 30]))

    database._add_hdf_vector_output(0, 1, values_group, array([100, 200]))
    assert array(values_group["arr_0"]["1"]) == pytest.approx(array([100, 200]))

    database._add_hdf_vector_output(1, 2, values_group, array([[0.1, 0.2, 0.3, 0.4]]))
    assert array(values_group["arr_1"]["2"]) == pytest.approx(
        array([[0.1, 0.2, 0.3, 0.4]])
    )

    with pytest.raises(ValueError):
        database._add_hdf_vector_output(1, 2, values_group, [1, 2])


def test_add_hdf_output_dataset(h5_file):
    """Test that output datasets are correctly added to the hdf groups of output."""
    database = Database()

    values_group = h5_file.require_group("v")
    keys_group = h5_file.require_group("k")

    values = {"f": 10, "g": array([1, 2]), "Iter": [3], "@f": array([[1, 2, 3]])}
    database._add_hdf_output_dataset(10, keys_group, values_group, values)
    assert list(keys_group["10"]) == list(array(list(values.keys()), dtype=string_))
    assert array(values_group["10"]) == pytest.approx(array([10]))
    assert array(values_group["arr_10"]["1"]) == pytest.approx(array([1, 2]))
    assert array(values_group["arr_10"]["2"]) == pytest.approx(array([3]))
    assert array(values_group["arr_10"]["3"]) == pytest.approx(array([[1, 2, 3]]))

    values = {
        "i": array([1, 2]),
        "Iter": 1,
        "@j": array([[1, 2, 3]]),
        "k": 99,
        "l": 100,
    }
    database._add_hdf_output_dataset(100, keys_group, values_group, values)
    assert list(keys_group["100"]) == list(array(list(values.keys()), dtype=string_))
    assert array(values_group["100"]) == pytest.approx(array([1, 99, 100]))
    assert array(values_group["arr_100"]["0"]) == pytest.approx(array([1, 2]))
    assert array(values_group["arr_100"]["2"]) == pytest.approx(array([[1, 2, 3]]))


def test_get_missing_hdf_output_dataset(h5_file):
    """Test that missing values in the hdf  output datasets are correctly found."""
    database = Database()

    values_group = h5_file.require_group("v")
    keys_group = h5_file.require_group("k")

    values = {"f": 0.1, "g": array([1, 2])}
    database._add_hdf_output_dataset(10, keys_group, values_group, values)

    with pytest.raises(ValueError):
        database._get_missing_hdf_output_dataset(0, keys_group, values)

    values = {"f": 0.1, "g": array([1, 2]), "h": [10]}
    new_values, idx_mapping = database._get_missing_hdf_output_dataset(
        10, keys_group, values
    )
    assert new_values == {"h": [10]}
    assert idx_mapping == {"h": 2}

    values = {"f": 0.1, "g": array([1, 2])}
    new_values, idx_mapping = database._get_missing_hdf_output_dataset(
        10, keys_group, values
    )
    assert new_values == {}
    assert idx_mapping is None

    values = {"f": 0.1, "g": array([1, 2]), "h": [2, 3], "i": 20}
    new_values, idx_mapping = database._get_missing_hdf_output_dataset(
        10, keys_group, values
    )
    assert new_values == {"h": [2, 3], "i": 20}
    assert idx_mapping == {"h": 2, "i": 3}


def test_get_x_by_iter_except(problem):
    """Tests exception in get_x_by_iter."""
    with pytest.raises(ValueError):
        problem.database.get_x_by_iter(1000)


def test_contains_dataname(problem):
    """Tests data name belonging check."""
    database = problem.database
    assert not database.contains_dataname("toto")
    assert database.contains_dataname("Iter")


def test_get_history_array(problem):
    """Tests history extraction into an array."""
    database = problem.database
    values_array, variables_names, functions = (
        values_array,
        variables_names,
        functions,
    ) = database.get_history_array(design_variables_names=["x_1", "x_2"])
    values_array, _, _ = database.get_history_array(design_variables_names="x_1")
    assert values_array[-1, 1] < 1e-13
    with pytest.raises(TypeError):
        database.get_history_array(design_variables_names={"x_1": 0})
    # Test special case with only one iteration:
    database = Database()
    database.store(array([1.0, 1.0]), {"Rosenbrock": 0.0})
    database.get_history_array()


def test_ggobi_export(tmp_wd, problem):
    """Tests export to GGobi."""
    file_path = "opt_hist.xml"
    problem.database.export_to_ggobi(file_path=file_path)
    assert Path(file_path).exists()


def test_hdf_grad_export(tmp_wd, problem):
    """Tests export into HDF."""
    database = problem.database
    f_database_ref, x_database_ref = database.get_complete_history()
    func_data = "rosen_grad_test.hdf5"
    database.export_hdf(func_data)

    database_read = Database(func_data)
    f_database, x_database = database_read.get_complete_history()
    assert array(f_database) == pytest.approx(array(f_database_ref), rel=1e-16)
    assert array(x_database) == pytest.approx(array(x_database_ref), rel=1e-16)
    assert len(database) == len(database_read)


def test_hdf_import():
    """Tests import from HDF."""
    database = Database(input_hdf_file=DIRNAME / "rosen_grad.hdf5")
    fname = "rosen"
    gname = Database.get_gradient_name(fname)
    hist_x = database.get_x_history()
    hist_f = database.get_func_history(fname)
    hist_g = database.get_func_history(gname)
    hist_g2 = database.get_func_grad_history(fname)
    assert len(hist_f) > 2
    assert len(hist_f) == len(hist_g)
    assert (hist_g == hist_g2).all()
    assert len(hist_f) == len(hist_x)


def test_hdf_import_sob():
    """Tests import from HDF."""
    inf = DIRNAME / "mdf_backup.h5"
    database = Database(input_hdf_file=inf)
    hist_x = database.get_x_history()
    assert len(hist_x) == 5
    for func in ("-y_4", "g_1", "g_2", "g_3"):
        hist_f = database.get_func_history(func)
        assert len(hist_f) == 5
        hist_g = database.get_func_history(Database.get_gradient_name(func))
        assert len(hist_g) == 5


def test_opendace_import(tmp_wd):
    """Tests import from Opendace."""
    database = Database()
    inf = DIRNAME / "rae2822_cl075_085_mach_068_074.xml"
    database.import_from_opendace(inf)
    outfpath = Path("rae2822_cl075_085_mach_068_074_cp.hdf5")
    database.export_hdf(outfpath)
    assert outfpath.exists()


def test_duplicates():
    """Tests the storing of identical entries at different iterations."""
    database = Database()
    assert database.get_max_iteration() == 0
    x_vect = array([1.9, 8.9])
    value = rosen(x_vect)
    gradient = rosen_der(x_vect)
    values_dict = {"Rosenbrock": value, "@Rosenbrock": gradient, "Iter^2": [1]}
    # Insert the point twice with stacked data (e.g. 'Iter_squared'):
    database.store(x_vect, values_dict)
    values_dict["Iter^2"] = [1, 4]
    database.store(x_vect, values_dict)
    assert database.get_value(x_vect)["Iter"] == [1]
    # Check the stacked data names exception:
    with pytest.raises(ValueError):
        database.get_complete_history(stacked_data=["stuff"])
    # Check the extraction of the stacked data:
    database.get_complete_history(stacked_data=["Iter^2"], all_iterations=True)
    # assert f_history == [[value, 1, 1], [value, 4, 2]]
    # And without duplicate::
    database.get_complete_history(stacked_data=["Iter^2"])
    # assert f_history == [[value, 4, 2]]

    database.store(x_vect, {"newval": [1.0]})
    assert database.get_value(x_vect)["Iter"] == [1]


def test_missing_tag():
    """Tests the missing tag insertion."""
    database = Database()
    # Store a point with the associated objective function value and
    # gradient value:
    x_vect = array([1.9, 8.9])
    value = rosen(x_vect)
    gradient = rosen_der(x_vect)
    database.store(x_vect, {"Rosenbrock": value, "@Rosenbrock": gradient})
    # Store another point with the associated objective function value
    # only:
    database.store(array([1.0, 1.0]), {"Rosenbrock": 0.0})
    # Check that a missing tag is added during history extraction:
    functions = ["Rosenbrock", Database.get_gradient_name("Rosenbrock")]
    f_history, _ = database.get_complete_history(functions, add_missing_tag=True)
    assert f_history == [[value, gradient], [0.0, "NA"]]


def test__str__database():
    """Test that the string representation of the database is correct."""
    database = Database()
    x1 = array([1.0, 2.0])
    x2 = array([3.0, 4.5])
    value1 = rosen(x1)
    value2 = rosen(x2)
    database.store(x1, {"Rosenbrock": value1}, add_iter=False)
    database.store(x2, {"Rosenbrock": value2}, add_iter=False)

    ref = "{[1. 2.]: {'Rosenbrock': 100.0}, " "[3.  4.5]: {'Rosenbrock': 2029.0}}"

    assert database.__str__() == ref


def test_filter_database():
    """Test that the database is correctly filtered."""
    database = Database()
    x1 = array([1.0, 2.0])
    x2 = array([3.0, 4.5])
    value1 = rosen(x1)
    der_value1 = rosen_der(x1)
    value2 = rosen(x2)
    der_value2 = rosen_der(x2)
    database.store(x1, {"Rosenbrock": value1, "@Rosenbrock": der_value1})
    database.store(x2, {"Rosenbrock": value2, "@Rosenbrock": der_value2})

    # before filter
    assert database.get_value(x1) == {
        "Rosenbrock": value1,
        "@Rosenbrock": der_value1,
        "Iter": [1],
    }
    assert database.get_value(x2) == {
        "Rosenbrock": value2,
        "@Rosenbrock": der_value2,
        "Iter": [2],
    }

    database.filter(["Rosenbrock"])

    # after filter
    assert database.get_value(x1) == {"Rosenbrock": value1}
    assert database.get_value(x2) == {"Rosenbrock": value2}


def test__str__hashable_ndarray():
    """Tests the string representation."""
    x_array = array([1.0, 1.0])
    x_hash = HashableNdarray(x_array)
    assert str(x_hash) == str(x_array)


def test__repr__():
    """Tests the __repr__ method."""
    x_array = array([1.0, 1.0])
    x_hash = HashableNdarray(x_array)
    assert repr(x_hash) == str(x_array)


def test_unwrap():
    """Tests HashableNdarray unwrapping."""
    x_array = array([1.0, 1.0])
    x_hash = HashableNdarray(x_array)
    assert x_hash.unwrap() is x_hash.wrapped
    x_hash = HashableNdarray(x_array, tight=True)
    assert not x_hash.unwrap() is x_hash.wrapped
    assert (x_hash.unwrap() == x_array).all()


def test_fail_import():
    with pytest.raises(KeyError):
        Database(FAIL_HDF)


def test_remove_empty_entries():
    database = Database()
    database.store(ones(1), {})
    database.remove_empty_entries()
    with pytest.raises(ValueError):
        database.get_x_by_iter(0)


def test_get_last_n_x():
    database = Database()
    database.store(ones(1), {})
    database.store(2 * ones(1), {})
    database.store(3 * ones(1), {})
    assert database.get_last_n_x(3) == [ones(1), 2 * ones(1), 3 * ones(1)]
    assert database.get_last_n_x(2) == [2 * ones(1), 3 * ones(1)]

    with pytest.raises(ValueError):
        database.get_last_n_x(4)


def test_name():
    """Check the name of the database."""

    class NewDatabase(Database):
        pass

    assert NewDatabase().name == "NewDatabase"
    assert Database(name="my_database").name == "my_database"


def test_notify_newiter_store_listeners():
    """Check that notify_newiter_listeners and notify_store_listeners works properly."""
    database = Database()
    database.x_sum = 0

    def add(x):
        database.x_sum += x

    database.store(array([1]), {"y": 0})
    assert database.notify_newiter_listeners() is None
    database.add_new_iter_listener(add)
    database.add_store_listener(add)
    database.notify_newiter_listeners()
    assert database.x_sum == 1
    database.notify_newiter_listeners(HashableNdarray(array([2])))
    assert database.x_sum == 3
    database.notify_store_listeners()
    assert database.x_sum == 4
    database.notify_store_listeners(HashableNdarray(array([2])))
    assert database.x_sum == 6


@pytest.fixture
def simple_database():
    """A database with a single element: ([0.], {'y': 1.})."""
    database = Database()
    database.store(array([0.0]), {"y": 1.0})
    return database


@pytest.mark.parametrize(
    "reset_iteration_counter,max_iteration", [(False, 1), (True, 0)]
)
def test_clear(simple_database, reset_iteration_counter, max_iteration):
    """Check the Database.clear method."""
    simple_database.clear(reset_iteration_counter)
    assert len(simple_database) == 0
    assert simple_database.get_max_iteration() == max_iteration


def test_last_item(simple_database):
    """Check that the property last_item is the last item stored in the database."""
    assert simple_database.last_item["y"] == 1.0
