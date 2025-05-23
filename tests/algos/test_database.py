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

import re
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from numpy import array
from numpy import full
from numpy import ones
from numpy.linalg import norm
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from pandas import NA
from scipy.optimize import rosen
from scipy.optimize import rosen_der

from gemseo.algos.database import Database
from gemseo.algos.database import HashableNdarray
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt.factory import OptimizationLibraryFactory
from gemseo.datasets.dataset import Dataset
from gemseo.problems.optimization.rosenbrock import Rosenbrock

if TYPE_CHECKING:
    from gemseo.algos.optimization_result import OptimizationResult

DIRNAME = Path(__file__).parent
FAIL_HDF = DIRNAME / "fail.hdf5"


def rel_err(to_test, ref):
    """Compute the relative error.

    Args:
        to_test: The value to be tested.
        ref: The reference value.
    """
    n_ref = norm(ref)
    err = norm(to_test - ref)
    if n_ref > 1e-14:
        return err / n_ref
    return err


@pytest.fixture
def problem_and_result() -> tuple[Rosenbrock, OptimizationResult]:
    """The Rosenbrock problem solved with L-BFGS-B and the optimization result."""
    rosenbrock = Rosenbrock()
    result = OptimizationLibraryFactory().execute(rosenbrock, algo_name="L-BFGS-B")
    return rosenbrock, result


@pytest.fixture
def problem(problem_and_result) -> Rosenbrock:
    """The Rosenbrock problem solved with L-BFGS-B."""
    return problem_and_result[0]


@pytest.fixture(scope="module")
def simple_database_with_large_x_vect() -> Database:
    """A database with a single element."""
    design_space = DesignSpace()
    design_space.add_variable("variable_1")
    design_space.add_variable("foo_bar", size=3)
    design_space.add_variable("tata", size=2)
    database = Database(input_space=design_space)
    database.store(
        array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        {"w": 1.0, "x": [2], "y": array([3.0]), "z": array([4.0, 5.0])},
    )
    database.store(
        array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0]),
        {"w": 5.0, "x": [4], "y": array([3.0]), "z": array([2.0, 1.0])},
    )
    database.store(
        array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0]),
        {"w": 2.0, "x": [4], "y": array([6.0]), "z": array([8.0, 10.0])},
    )
    return database


def test_correct_store_unstore(problem) -> None:
    """Test the storage of objective function values and gradient values."""
    database = problem.database
    fname = problem.objective.name
    for x_var in database:
        x_var = x_var.unwrap()
        func_value = database.get_function_value(fname, x_var)
        gname = database.get_gradient_name(fname)
        func_rel_err = rel_err(func_value, rosen(x_var))
        assert_almost_equal(func_rel_err, 0.0, decimal=14)
        grad_value = database.get_function_value(gname, x_var)
        if grad_value is not None:
            grad_rel_err = rel_err(grad_value, rosen_der(x_var))
            assert_almost_equal(grad_rel_err, 0.0, decimal=14)


def test_write_read(tmp_wd, problem) -> None:
    """Test the writing of objective function values and gradient values."""
    database = problem.database
    hdf_file = "rosen.hdf"
    database.to_hdf(hdf_file)
    assert Path(hdf_file).exists()
    fname = problem.objective.name

    loaded_db = Database.from_hdf(hdf_file)

    for x_var in database:
        x_var = x_var.unwrap()
        f_ref = database.get_function_value(fname, x_var)
        gname = database.get_gradient_name(fname)
        df_ref = database.get_function_value(gname, x_var)

        f_loaded = loaded_db.get_function_value(fname, x_var)

        f_rel_err = rel_err(f_ref, f_loaded)
        assert_almost_equal(f_rel_err, 0.0, decimal=14)

        df_loaded = loaded_db.get_function_value(gname, x_var)
        if df_ref is None:
            assert df_loaded is None
        else:
            df_rel_err = rel_err(df_ref, df_loaded)
            assert_almost_equal(df_rel_err, 0.0, decimal=14)


def test_get_output_names() -> None:
    database = Database()
    fname = "f"
    gname = database.get_gradient_name(fname)
    database.store(array([1.0]), {fname: 1, gname: array([1.0])})

    assert database.get_function_names(True) == [fname]
    assert database.get_function_names(False) == [gname, fname]

    assert database.get_function_names() == [fname]
    assert database.get_function_names(False) == [
        gname,
        fname,
    ]


def test_get_f_hist() -> None:
    """Test the objective history extraction."""
    problem = Rosenbrock()
    database = problem.database
    problem.preprocess_functions()
    for x_vec in (array([0.0, 1.0]), array([1, 2.0])):
        problem.objective.evaluate(x_vec)
        problem.objective.jac(x_vec)

    hist_x = database.get_x_vect_history()
    fname = problem.objective.name
    hist_f = database.get_function_history(fname)
    gname = Database.get_gradient_name(fname)
    hist_g = database.get_function_history(gname)
    hist_g2 = database.get_gradient_history(fname)
    assert len(hist_f) == 2
    assert len(hist_g) == 2
    assert (hist_g == hist_g2).all()
    assert len(hist_f) == len(hist_x)


def test_get_f_hist_rosen(problem) -> None:
    """Test the objective history extraction."""
    database = problem.database
    hist_x = database.get_x_vect_history()
    fname = problem.objective.name
    hist_f = database.get_function_history(fname)
    gname = Database.get_gradient_name(fname)
    hist_g = database.get_function_history(gname)
    hist_g2 = database.get_gradient_history(fname)
    assert len(hist_f) > 2
    assert (hist_g == hist_g2).all()
    assert len(hist_f) == len(hist_x)


def test_clean_from_iteration(problem) -> None:
    """Tests access to design variables by iteration index."""
    database = problem.database
    # clean after iteration 13
    database.clear_from_iteration(13)
    assert len(database) == 13
    # Add another point that cannot already exist
    x_test = array([10.0, -100.0])
    database.store(x_test, {})


def test_get_x_at_iteration() -> None:
    """Tests access to design variables by iteration index."""
    problem = Rosenbrock()
    database = problem.database
    with pytest.raises(ValueError):
        database.get_x_vect(1)
    OptimizationLibraryFactory().execute(problem, algo_name="L-BFGS-B")
    hist_g2 = database.get_x_vect(21)
    assert database.get_iteration(hist_g2) - 1 == 20
    with pytest.raises(KeyError):
        database.get_iteration(array([123.456]))
    assert_almost_equal(hist_g2[0], 0.92396186, decimal=6)
    assert_almost_equal(hist_g2[1], 0.85259476, decimal=6)
    assert all(database.get_x_vect(-1) == database.get_x_vect(29))


def test_scipy_df0_rosenbrock(problem_and_result) -> None:
    """Tests the storage of optimization solutions."""
    problem, result = problem_and_result
    database = problem.database
    assert result.f_opt < 6.5e-11
    assert norm(database.get_x_vect_history()[-1] - ones(2)) < 2e-5
    assert database.get_function_history("rosen")[-1] < 6.2e-11


@pytest.mark.parametrize("each_iter", [True, False])
@pytest.mark.parametrize("node", ["test_node", ""])
def test_append_export(tmp_wd, each_iter, node) -> None:
    """Test that a database is correctly exported when it is appended.

    Test either after each storage call, or after each iteration, with or without hdf
    node.
    """

    database = Database()
    file_path_db = "db.hdf5"

    n_calls = 20
    outputs = [
        {"y1": array([10.0, 10])},
        {"y2": 20.0},
        {"@y2": array([30.0, 30])},
        {"y3": array([40.0, 40, 40])},
        {"y4": 50.0},
    ]

    def callback(x):
        return database.to_hdf(file_path_db, append=True, hdf_node_path=node)

    if each_iter:
        database.add_new_iter_listener(callback)
    else:
        database.add_store_listener(callback)

    for i in range(n_calls):
        for output in outputs:
            database.store(full(2, i), output)

    if node:
        with pytest.raises(KeyError):
            Database.from_hdf(file_path_db)
    with pytest.raises(KeyError):
        Database.from_hdf(file_path_db, hdf_node_path="wrong_node")

    new_database = Database.from_hdf(file_path_db, hdf_node_path=node)
    assert len(new_database) == len(database)

    for i in range(n_calls):
        x = full(2, i)
        for key, value in database[x].items():
            assert value == pytest.approx(new_database[x][key])


def test_get_x_at_iteration_except(problem) -> None:
    """Tests exception in get_x_vect."""
    with pytest.raises(ValueError):
        problem.database.get_x_vect(1000)


def test_contains_dataname(problem) -> None:
    """Tests data name belonging check."""
    database = problem.database
    assert "toto" not in database.get_function_names(False)


def test_get_history_array(problem) -> None:
    """Tests history extraction into an array."""
    database = problem.database
    values_array, _, _ = database.get_history_array(input_names=["x_1", "x_2"])
    assert_almost_equal(values_array[-1, 1], 1)
    # Test special case with only one iteration:
    database = Database()
    database.store(array([1.0, 1.0]), {"Rosenbrock": 0.0})
    values_array, values_array_names, function_names = database.get_history_array()
    assert_array_equal(values_array, array([[0.0, 1.0, 1.0]]))
    assert values_array_names == ["Rosenbrock", "x_1", "x_2"]
    assert function_names == ["Rosenbrock"]


@pytest.mark.parametrize("input_names", ["label_1", ["label_1"]])
def test_get_history_array_wrong_dimension(
    simple_database_with_large_x_vect, input_names
) -> None:
    """Check the exception raised when the input names have the wrong dimension."""
    with pytest.raises(ValueError, match="Expected 6 names, got 1\\."):
        simple_database_with_large_x_vect.get_history_array(input_names=input_names)


@pytest.mark.parametrize(
    (
        "function_names",
        "input_names",
        "expected_values_array",
        "expected_values_array_names",
        "expected_function_names",
    ),
    [
        (
            "z",
            (),
            array([
                [4.0, 5.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                [2.0, 1.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
                [8.0, 10.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
            ]),
            ["z[0]", "z[1]", "x_1", "x_2", "x_3", "x_4", "x_5", "x_6"],
            "z",
        ),
        (
            "z",
            (
                "variable_1",
                "foo_bar[0]",
                "foo_bar[1]",
                "foo_bar[2]",
                "tata[0]",
                "tata[1]",
            ),
            array([
                [4.0, 5.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                [2.0, 1.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
                [8.0, 10.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
            ]),
            [
                "z[0]",
                "z[1]",
                "variable_1",
                "foo_bar[0]",
                "foo_bar[1]",
                "foo_bar[2]",
                "tata[0]",
                "tata[1]",
            ],
            "z",
        ),
        (
            ("z", "w"),
            (),
            array([
                [4.0, 5.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                [2.0, 1.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
                [8.0, 10.0, 2.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
            ]),
            ["z[0]", "z[1]", "w", "x_1", "x_2", "x_3", "x_4", "x_5", "x_6"],
            ("z", "w"),
        ),
        (
            ("z", "w"),
            (
                "variable_1",
                "foo_bar[0]",
                "foo_bar[1]",
                "foo_bar[2]",
                "tata[0]",
                "tata[1]",
            ),
            array([
                [4.0, 5.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                [2.0, 1.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
                [8.0, 10.0, 2.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
            ]),
            [
                "z[0]",
                "z[1]",
                "w",
                "variable_1",
                "foo_bar[0]",
                "foo_bar[1]",
                "foo_bar[2]",
                "tata[0]",
                "tata[1]",
            ],
            ("z", "w"),
        ),
    ],
)
def test_get_history_array_names(
    simple_database_with_large_x_vect,
    function_names,
    input_names,
    expected_values_array,
    expected_values_array_names,
    expected_function_names,
) -> None:
    """Check that get_history_array works with input names."""
    values_array, values_array_names, function_names = (
        simple_database_with_large_x_vect.get_history_array(
            function_names=function_names, input_names=input_names
        )
    )
    assert_array_equal(values_array, expected_values_array)
    assert values_array_names == expected_values_array_names
    assert function_names == expected_function_names


def test_get_history_array_wrong_f_name(problem) -> None:
    """Check that get_history_array raises an error with an unknown function name."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "'foo' is not an output name; available ones are '@rosen' and 'rosen'."
        ),
    ):
        problem.database.get_history_array(function_names=["foo"])

    with pytest.raises(
        ValueError,
        match=re.escape(
            "'bar' and 'foo' are not output names; "
            "available ones are '@rosen' and 'rosen'."
        ),
    ):
        problem.database.get_history_array(function_names=["foo", "bar"])


def test_ggobi_export(tmp_wd, problem) -> None:
    """Tests export to GGobi."""
    file_path = "opt_hist.xml"
    problem.database.to_ggobi(file_path=file_path)
    assert Path(file_path).exists()


def test_hdf_grad_export(tmp_wd, problem) -> None:
    """Tests export into HDF."""
    database = problem.database
    f_database_ref, x_database_ref = database.get_history()
    hdf_file = "rosen_grad_test.hdf5"
    database.to_hdf(hdf_file)

    database_read = Database.from_hdf(hdf_file)
    f_database, x_database = database_read.get_history()
    assert array(f_database) == pytest.approx(array(f_database_ref), rel=1e-16)
    assert array(x_database) == pytest.approx(array(x_database_ref), rel=1e-16)
    assert len(database) == len(database_read)


def test_hdf_import() -> None:
    """Tests import from HDF."""
    database = Database.from_hdf(DIRNAME / "rosen_grad.hdf5")
    fname = "rosen"
    gname = Database.get_gradient_name(fname)
    hist_x = database.get_x_vect_history()
    hist_f = database.get_function_history(fname)
    hist_g = database.get_function_history(gname)
    hist_g2 = database.get_gradient_history(fname)
    assert len(hist_f) > 2
    assert len(hist_f) == len(hist_g)
    assert (hist_g == hist_g2).all()
    assert len(hist_f) == len(hist_x)


def test_hdf_import_sob() -> None:
    """Tests import from HDF."""
    database = Database.from_hdf(DIRNAME / "mdf_backup.h5")
    hist_x = database.get_x_vect_history()
    assert len(hist_x) == 5
    for func in ("-y_4", "g_1", "g_2", "g_3"):
        hist_f = database.get_function_history(func)
        assert len(hist_f) == 5
        hist_g = database.get_function_history(Database.get_gradient_name(func))
        assert len(hist_g) == 5


def test_opendace_import(tmp_wd) -> None:
    """Tests import from Opendace."""
    database = Database()
    inf = DIRNAME / "rae2822_cl075_085_mach_068_074.xml"
    database.update_from_opendace(inf)
    outfpath = Path("rae2822_cl075_085_mach_068_074_cp.hdf5")
    database.to_hdf(outfpath)
    assert outfpath.exists()


def test_missing_tag() -> None:
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
    f_history, _ = database.get_history(functions, add_missing_tag=True)
    assert f_history == [[value, gradient], [0.0, "NA"]]


def test__str__database() -> None:
    """Test that the string representation of the database is correct."""
    database = Database()
    x1 = array([1.0, 2.0])
    x2 = array([3.0, 4.5])
    value1 = rosen(x1)
    value2 = rosen(x2)
    database.store(x1, {"Rosenbrock": value1})
    database.store(x2, {"Rosenbrock": value2})

    ref = "{[1. 2.]: {'Rosenbrock': 100.0}, [3.  4.5]: {'Rosenbrock': 2029.0}}"

    assert str(database) == ref


def test_filter_database() -> None:
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
    assert database[x1] == {"Rosenbrock": value1, "@Rosenbrock": der_value1}
    assert database[x2] == {"Rosenbrock": value2, "@Rosenbrock": der_value2}

    database.filter(["Rosenbrock"])

    # after filter
    assert database[x1] == {"Rosenbrock": value1}
    assert database[x2] == {"Rosenbrock": value2}


def test__str__hashable_ndarray() -> None:
    """Tests the string representation."""
    x_array = array([1.0, 1.0])
    x_hash = HashableNdarray(x_array)
    assert str(x_hash) == str(x_array)


def test__repr__() -> None:
    """Tests the __repr__ method."""
    x_array = array([1.0, 1.0])
    x_hash = HashableNdarray(x_array)
    assert repr(x_hash) == str(x_array)


def test_unwrap() -> None:
    """Tests HashableNdarray unwrapping."""
    x_array = array([1.0, 1.0])
    x_hash = HashableNdarray(x_array)
    assert x_hash.unwrap() is x_hash.wrapped_array
    x_hash = HashableNdarray(x_array, copy=True)
    assert x_hash.unwrap() is not x_hash.wrapped_array
    assert (x_hash.unwrap() == x_array).all()


def test_fail_import() -> None:
    with pytest.raises(KeyError):
        Database.from_hdf(FAIL_HDF)


def test_remove_empty_entries() -> None:
    database = Database()
    database.store(array([1.0]), {})
    database.store(array([2.0]), {"f": array([1.0])})
    database.remove_empty_entries()
    assert len(database) == 1
    assert array([2.0]) in database


def test_get_last_n_x() -> None:
    database = Database()
    database.store(ones(1), {})
    database.store(2 * ones(1), {})
    database.store(3 * ones(1), {})
    assert database.get_last_n_x_vect(3) == [ones(1), 2 * ones(1), 3 * ones(1)]
    assert database.get_last_n_x_vect(2) == [2 * ones(1), 3 * ones(1)]

    with pytest.raises(ValueError):
        database.get_last_n_x_vect(4)


def test_name() -> None:
    """Check the name of the database."""

    class NewDatabase(Database):
        pass

    assert NewDatabase().name == "NewDatabase"
    assert Database(name="my_database").name == "my_database"


def test_notify_newiter_store_listeners() -> None:
    """Check that notify_new_iter_listeners and notify_store_listeners works
    properly."""
    database = Database()
    database.x_sum = 0

    def add(x) -> None:
        database.x_sum += x

    database.store(array([1]), {"y": 0})
    assert database.notify_new_iter_listeners() is None
    database.add_new_iter_listener(add)
    database.add_store_listener(add)
    database.notify_new_iter_listeners()
    assert database.x_sum == 1
    database.notify_new_iter_listeners(HashableNdarray(array([2])))
    assert database.x_sum == 3
    database.notify_store_listeners()
    assert database.x_sum == 4
    database.notify_store_listeners(HashableNdarray(array([2])))
    assert database.x_sum == 6


@pytest.fixture
def simple_database():
    """A database with a single element."""
    database = Database()
    database.store(
        array([0.0]), {"w": 1.0, "x": [2], "y": array([3.0]), "z": array([4.0, 5.0])}
    )
    return database


def test_clear(simple_database) -> None:
    """Check the Database.clear method."""
    simple_database.clear()
    assert len(simple_database) == 0


def test_last_item(simple_database) -> None:
    """Check that the property last_item is the last item stored in the database."""
    assert simple_database.last_item["y"] == 3.0


def test_get_history_array_with_simple_database(simple_database) -> None:
    """Check get_history_array with a simple database."""
    values_array, variable_names, functions = simple_database.get_history_array()
    assert_almost_equal(values_array, array([[1.0, 2.0, 3.0, 4.0, 5.0, 0.0]]))
    assert variable_names == ["w", "x", "y", "z[0]", "z[1]", "x_1"]
    assert functions == ["w", "x", "y", "z"]


def test_get_output_value() -> None:
    """Check get_output()."""
    database = Database()
    input_value = array([1.0])
    output_value = {"y": array([2.0])}
    database.store(input_value, output_value)

    other_input_value = input_value + 0.01
    assert database._Database__get_output(other_input_value) is None
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The iteration must be within {-N, ..., -1, 1, ..., N} "
            "where N=1 is the number of iterations."
        ),
    ):
        database._Database__get_output(2)

    assert database._Database__get_output(1) == output_value
    assert database._Database__get_output(input_value) == output_value
    assert database._Database__get_output(HashableNdarray(input_value)) == output_value

    assert (
        database._Database__get_output(other_input_value, tolerance=0.1) == output_value
    )
    assert (
        database._Database__get_output(
            HashableNdarray(other_input_value), tolerance=0.1
        )
        == output_value
    )


def test_check_output_history_is_empty() -> None:
    """Check check_output_history_is_empty."""
    database = Database()
    database.store(array([1.0]), {"f": array([2.0])})
    assert not database.check_output_history_is_empty("f")
    assert database.check_output_history_is_empty("g")


def test_get_hashable_ndarray() -> None:
    """Check get_hashable_ndarray."""
    with pytest.raises(
        KeyError,
        match=re.escape(
            "A database key must be either a NumPy array of a HashableNdarray; "
            "got <class 'int'> instead."
        ),
    ):
        Database.get_hashable_ndarray(12)


def test_delitem() -> None:
    """Check __delitem__."""
    database = Database()
    database.store(array([1.0]), {"f": array([2.0])})
    database.store(array([2.0]), {"f": array([3.0])})
    del database[array([1.0])]
    assert len(database) == 1
    assert not database.get(array([1.0]))
    assert database.get(array([2.0]))


@pytest.mark.parametrize("name", ["store_listener", "new_iter_listener"])
def test_add_listener_twice(name):
    """Check that a listener cannot be added twice."""
    database = Database()
    listeners = getattr(database, f"_Database__{name}s")
    assert not listeners

    method = getattr(database, f"add_{name}")
    assert method(sum)
    assert listeners == [sum]
    assert not method(sum)
    assert listeners == [sum]


def test_clear_listeners():
    """Check default clear_listeners."""
    database = Database()
    database.add_new_iter_listener(sum)
    database.add_store_listener(sorted)
    assert database._Database__new_iter_listeners == [sum]
    assert database._Database__store_listeners == [sorted]

    new_iter_listeners, store_listeners = database.clear_listeners()
    assert database._Database__new_iter_listeners == []
    assert database._Database__store_listeners == []
    assert new_iter_listeners == {sum}
    assert store_listeners == {sorted}


@pytest.mark.parametrize(
    ("store_listeners", "new_iter_listeners", "expected"),
    [
        (None, None, (set(), set())),
        ((), None, (set(), {sum, sorted})),
        (None, (), ({sum, sorted}, set())),
        ((), (), ({sum, sorted}, {sum, sorted})),
        ([sum], [sorted], ({sorted}, {sum})),
    ],
)
def test_clear_listeners_arguments(store_listeners, new_iter_listeners, expected):
    """Check the arguments of clear_listeners."""
    database = Database()
    for func in [sum, sorted]:
        database.add_new_iter_listener(func)
        database.add_store_listener(func)
    assert (
        database.clear_listeners(
            store_listeners=store_listeners, new_iter_listeners=new_iter_listeners
        )
        == expected
    )


def test_get_function_history():
    """Check that get_function_history raises a KeyError when the name is missing."""
    with pytest.raises(
        KeyError, match=re.escape("The database 'foo' contains no value of 'g'.")
    ):
        Database("foo").get_function_history("g")


@pytest.mark.parametrize(
    "store_orders", [[0, 1, 2, 3], [1, 3, 2, 0], [3, 1, 0, 2], [3, 2, 1, 0]]
)
def test_store_append(tmp_wd, store_orders):
    """Verify the export in append mode with successive store.

    A particular case is treated, store empty values, then store values one at a time,
    versus storing a complete dict.
    """
    db1 = Database()
    db2 = Database()
    bk_file_1 = Path("out_1.h5")
    bk_file_2 = Path("out_2.h5")
    x0 = array([1.0, 2.0])
    x1 = array([2.0, 2.0])

    def _store(x, in_db1, store_z=True, store_t=True, store_s=True):
        values = {}
        if in_db1:
            db = db1
            out_file = bk_file_1
        else:
            db = db2
            out_file = bk_file_2
        if store_z:
            values["z"] = array([sum(x0)])
        if store_t:
            values["t"] = 2 * x0 + 3
        if store_s:
            values["s"] = x0[0]
        db.store(x, values)
        db.to_hdf(out_file, append=True)

    db1.store(x0, {"z": array([sum(x0)])})
    db1.to_hdf(bk_file_1, append=True)
    _store(x0, True, store_z=True, store_t=False)
    _store(x0, True, store_z=False, store_t=True)
    _store(x1, True, store_z=True, store_t=True)

    # Test various orders for storing the data z, t, s
    args_store = [
        [False, False, False],
        [True, False, False],
        [False, True, False],
        [False, False, True],
    ]

    for args in [args_store[i] for i in store_orders]:
        _store(x0, False, *args)
    _store(x1, store_z=True, store_t=True, store_s=True, in_db1=False)

    db_read_1 = Database.from_hdf(bk_file_1)
    db_read_2 = Database.from_hdf(bk_file_2)
    for func in ["z", "t", "s"]:
        f_s = db_read_1.get_function_history(func, with_x_vect=False)
        f_p = db_read_2.get_function_history(func, with_x_vect=False)
        assert_array_equal(f_s, f_p, strict=True)


def test_store_twice(tmp_wd) -> None:
    """Check that store twice the same data in hdf only stores once."""
    database = Database()
    x_vect = array([1.0, 1.0])
    item = {"foo": 0.0}
    database.store(x_vect, item)
    database.store(x_vect, item)
    path = Path("foo.hdf5")
    database.to_hdf(path)
    assert len(Database.from_hdf(path)) == 1


def test_to_dataset_with_missing_integer() -> None:
    """Test the dataset export when there is an integer missing."""
    database = Database()
    database.store(array([1.0]), {"foo": 0, "toto": 1.0})
    database.store(array([0.0]), {"toto": 3.0})
    dataset = database.to_dataset()

    expected_dataset = Dataset()
    expected_dataset.add_variable("input", array([1.0, 0.0]))
    expected_dataset.add_variable(
        "foo",
        array([0, NA]),
    )
    expected_dataset.add_variable("toto", array([1.0, 3.0]))
    expected_dataset["parameters", "foo", 0] = expected_dataset[
        "parameters", "foo", 0
    ].astype("Int64")

    assert dataset.equals(expected_dataset)
