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
#    INITIAL AUTHORS - initial API and implementation and/or
#                        initial documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
#        :author: Benoit Pauwels - stacked data ; docstrings

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import unittest
from os.path import dirname, exists, join

import pytest
from numpy import arange, array, ones
from numpy.linalg import norm
from scipy.optimize import rosen, rosen_der

from gemseo.algos.database import Database, HashableNdarray
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.problems.analytical.rosenbrock import Rosenbrock

DIRNAME = dirname(os.path.realpath(__file__))
FAIL_HDF = join(DIRNAME, "fail.hdf5")


@pytest.mark.usefixtures("tmp_wd")
class TestDatabase(unittest.TestCase):
    """Tests for the Database class."""

    @staticmethod
    def test_init():
        """Tests Database initializer."""
        Database()

    @staticmethod
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

    def test_correct_store_unstore(self):
        """Tests the storage of objective function values and gradient values."""
        problem = Rosenbrock()
        OptimizersFactory().execute(problem, "L-BFGS-B")
        database = problem.database
        fname = problem.objective.name
        for x_var in database.keys():
            x_var = x_var.unwrap()
            func_value = database.get_f_of_x(fname, x_var)
            grad_value = database.get_f_of_x(database.GRAD_TAG + fname, x_var)
            func_rel_err = self.rel_err(func_value, rosen(x_var))
            grad_rel_err = self.rel_err(grad_value, rosen_der(x_var))
            self.assertAlmostEqual(func_rel_err, 0.0, places=14)
            self.assertAlmostEqual(grad_rel_err, 0.0, places=14)

    def test_write_read(self):
        """Tests the writing of objective function values and gradient values."""
        problem = Rosenbrock()
        OptimizersFactory().execute(problem, "L-BFGS-B")
        database = problem.database
        outf = "rosen.hdf"
        database.export_hdf(outf)
        assert exists(outf)
        fname = problem.objective.name

        loaded_db = Database(outf)

        for x_var in database.keys():
            x_var = x_var.unwrap()
            f_ref = database.get_f_of_x(fname, x_var)
            df_ref = database.get_f_of_x(database.GRAD_TAG + fname, x_var)

            f_loaded = loaded_db.get_f_of_x(fname, x_var)
            df_loaded = loaded_db.get_f_of_x(database.GRAD_TAG + fname, x_var)

            f_rel_err = self.rel_err(f_ref, f_loaded)
            df_rel_err = self.rel_err(df_ref, df_loaded)

            self.assertAlmostEqual(f_rel_err, 0.0, places=14)
            self.assertAlmostEqual(df_rel_err, 0.0, places=14)

    def test_set_item(self):
        """Tests setitem."""
        database = Database()
        k = array([1.0])
        #         Non ndarray key error
        self.assertRaises(Exception, database.setdefault, "toto", 1)
        hash_k = HashableNdarray(k)
        self.assertRaises(Exception, database.setdefault, hash_k, k)
        database.setdefault(hash_k, {"f": 1})
        self.assertRaises(TypeError, database.get, 1.0)
        self.assertRaises(TypeError, lambda x: database[x], 1.0)
        database.get(k)
        database[k]

    def test_get_all_datanames(self):
        database = Database()
        database.store(array([1.0]), {"f": 1, database.GRAD_TAG + "f": array([1.0])})

        assert database.get_all_data_names(True, True) == ["f"]
        assert database.get_all_data_names(False, True) == [
            database.GRAD_TAG + "f",
            "f",
        ]

        assert database.get_all_data_names(True, False) == [database.ITER_TAG, "f"]
        assert database.get_all_data_names(False, False) == [
            database.GRAD_TAG + "f",
            database.ITER_TAG,
            "f",
        ]

    @staticmethod
    def test_get_f_hist():
        """Tests objective history extraction."""
        problem = Rosenbrock()
        database = problem.database
        OptimizersFactory().execute(problem, "L-BFGS-B")
        hist_x = database.get_x_history()
        hist_f = database.get_func_history(problem.objective.name)
        hist_g = database.get_func_history(Database.GRAD_TAG + problem.objective.name)
        hist_g2 = database.get_func_grad_history(problem.objective.name)
        assert len(hist_f) > 2
        assert len(hist_f) == len(hist_g)
        assert (hist_g == hist_g2).all()
        assert len(hist_f) == len(hist_x)

    def test_clean_from_iterate(self):
        """Tests access to design variables by iteration index."""
        problem = Rosenbrock()
        database = problem.database
        OptimizersFactory().execute(problem, "L-BFGS-B")

        # clean after iterate 12
        database.clean_from_iterate(12)
        self.assertEqual(len(database), 13)
        # Add another point that cannot already exists
        x_test = array([10.0, -100.0])
        database.store(x_test, {}, add_iter=True)
        # Make sure that the iter tag is correct
        iter_id = int(database.get_f_of_x(Database.ITER_TAG, x_test)[0])
        self.assertEqual(iter_id, len(database))

    def test_get_x_by_iter(self):
        """Tests access to design variables by iteration index."""
        problem = Rosenbrock()
        database = problem.database
        self.assertRaises(ValueError, database.get_x_by_iter, 0)
        OptimizersFactory().execute(problem, "L-BFGS-B")
        hist_g2 = database.get_x_by_iter(20)
        assert database.get_index_of(hist_g2) == 20
        self.assertRaises(KeyError, database.get_index_of, array([123.456]))
        self.assertAlmostEqual(hist_g2[0], 0.92396186, 6)
        self.assertAlmostEqual(hist_g2[1], 0.85259476, 6)

    @staticmethod
    def test_scipy_df0_rosenbrock():
        """Tests the storage of optimization solutions."""
        problem = Rosenbrock()
        database = problem.database
        result = OptimizersFactory().execute(problem, "L-BFGS-B")
        assert result.f_opt < 6.5e-11
        assert norm(database.get_x_history()[-1] - ones(2)) < 2e-5
        assert database.get_func_history(funcname="rosen", x_hist=False)[-1] < 6.2e-11

    def test_append_export(self):
        database = Database()
        file_path_db = "test_db_append.hdf5"
        # Export empty file
        database.export_hdf(file_path_db, append=False)
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

    def test_get_x_by_iter_except(self):
        """Tests exception in get_x_by_iter."""
        problem = Rosenbrock()
        database = problem.database
        OptimizersFactory().execute(problem, "L-BFGS-B")
        self.assertRaises(ValueError, database.get_x_by_iter, 1000)

    def test_contains_dataname(self):
        """Tests data name belonging check."""
        problem = Rosenbrock()
        database = problem.database
        OptimizersFactory().execute(problem, "L-BFGS-B")
        self.assertFalse(database.contains_dataname("toto"))
        self.assertTrue(database.contains_dataname("Iter"))

    def test_get_history_array(self):
        """Tests history extraction into an array."""
        problem = Rosenbrock()
        database = problem.database
        OptimizersFactory().execute(problem, "L-BFGS-B")
        values_array, variables_names, functions = (
            values_array,
            variables_names,
            functions,
        ) = database.get_history_array(design_variables_names=["x_1", "x_2"])
        values_array, _, _ = database.get_history_array(design_variables_names="x_1")
        self.assertLess(values_array[-1, 1], 1e-13)
        self.assertRaises(
            TypeError, database.get_history_array, design_variables_names={"x_1": 0}
        )
        # Test special case with only one iteration:
        database = Database()
        database.store(array([1.0, 1.0]), {"Rosenbrock": 0.0})
        database.get_history_array()

    @staticmethod
    def test_ggobi_export():
        """Tests export to GGobi."""
        problem = Rosenbrock()
        database = problem.database
        OptimizersFactory().execute(problem, "L-BFGS-B")
        file_path = "opt_hist.xml"
        database.export_to_ggobi(file_path=file_path)
        database.export_to_ggobi(file_path=file_path)
        path_exists = os.path.exists(file_path)
        assert path_exists

    @staticmethod
    def test_hdf_grad_export():
        """Tests export into HDF."""
        problem = Rosenbrock()
        database = problem.database
        OptimizersFactory().execute(problem, "L-BFGS-B")
        f_database_ref, x_database_ref = database.get_complete_history()
        func_data = "rosen_grad_test.hdf5"
        database.export_hdf(func_data)
        database_read = Database(func_data)
        f_database, x_database = database_read.get_complete_history()
        assert (norm(array(f_database) - array(f_database_ref)) < 1e-16).all()
        assert (norm(array(x_database) - array(x_database_ref)) < 1e-16).all()
        assert len(database) == len(database_read)

    @staticmethod
    def test_hdf_import():
        """Tests import from HDF."""
        inf = join(DIRNAME, "rosen_grad.hdf5")
        database = Database(input_hdf_file=inf)
        hist_x = database.get_x_history()
        hist_f = database.get_func_history("rosen")
        hist_g = database.get_func_history(Database.GRAD_TAG + "rosen")
        hist_g2 = database.get_func_grad_history("rosen")
        assert len(hist_f) > 2
        assert len(hist_f) == len(hist_g)
        assert (hist_g == hist_g2).all()
        assert len(hist_f) == len(hist_x)

    def test_hdf_import_sob(self):
        """Tests import from HDF."""
        inf = join(DIRNAME, "mdf_backup.h5")
        database = Database(input_hdf_file=inf)
        hist_x = database.get_x_history()
        assert len(hist_x) == 5
        for func in ("-y_4", "g_1", "g_2", "g_3"):
            hist_f = database.get_func_history(func)
            assert len(hist_f) == 5
            hist_g = database.get_func_history(Database.GRAD_TAG + func)
            assert len(hist_g) == 5

    @staticmethod
    def test_opendace_import():
        """Tests import from Opendace."""
        database = Database()
        inf = join(DIRNAME, "rae2822_cl075_085_mach_068_074.xml")
        database.import_from_opendace(inf)
        outfpath = "rae2822_cl075_085_mach_068_074_cp.hdf5"
        database.export_hdf(outfpath)
        assert os.path.exists(outfpath)

    def test_duplicates(self):
        """Tests the storing of identical entries at different iterations."""
        database = Database()
        assert database.get_max_iteration() == 0
        x_vect = array([1.9, 8.9])
        value = rosen(x_vect)
        gradient = rosen_der(x_vect)
        values_dict = {"Rosenbrock": value, "@Rosenbrock": gradient}
        # Insert the point twice with stacked data (e.g. 'Iter_squared'):
        values_dict["Iter^2"] = [1]
        database.store(x_vect, values_dict)
        values_dict["Iter^2"] = [1, 4]
        database.store(x_vect, values_dict)
        assert database.get_value(x_vect)["Iter"] == [1]
        # Check the stacked data names exception:
        self.assertRaises(
            Exception, database.get_complete_history, stacked_data=["stuff"]
        )
        # Check the extraction of the stacked data:
        database.get_complete_history(stacked_data=["Iter^2"], all_iterations=True)
        # assert f_history == [[value, 1, 1], [value, 4, 2]]
        # And without duplicate::
        database.get_complete_history(stacked_data=["Iter^2"])
        # assert f_history == [[value, 4, 2]]

        database.store(x_vect, {"newval": [1.0]})
        assert database.get_value(x_vect)["Iter"] == [1]

    @staticmethod
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
        functions = ["Rosenbrock", Database.GRAD_TAG + "Rosenbrock"]
        f_history, _ = database.get_complete_history(
            functions, add_missing_tag=True, missing_tag="NA"
        )
        assert f_history == [[value, gradient], [0.0, "NA"]]

    @staticmethod
    def test__str__():
        """Tests the string representation."""
        x_array = array([1.0, 1.0])
        x_hash = HashableNdarray(x_array)
        assert str(x_hash) == str(x_array)

    @staticmethod
    def test__repr__():
        """Tests the __repr__ method."""
        x_array = array([1.0, 1.0])
        x_hash = HashableNdarray(x_array)
        assert repr(x_hash) == str(x_array)

    @staticmethod
    def test_unwrap():
        """Tests HashableNdarray unwrapping."""
        x_array = array([1.0, 1.0])
        x_hash = HashableNdarray(x_array)
        assert x_hash.unwrap() is x_hash.wrapped
        x_hash = HashableNdarray(x_array, tight=True)
        assert not x_hash.unwrap() is x_hash.wrapped
        assert (x_hash.unwrap() == x_array).all()

    def test_fail_import(self):
        self.assertRaises(KeyError, Database, FAIL_HDF)

    def test_remove_empty_entries(self):
        database = Database()
        database.store(ones(1), {})
        database.remove_empty_entries()
        self.assertRaises(ValueError, database.get_x_by_iter, 0)

    def test_get_last_n_x(self):
        database = Database()
        database.store(ones(1), {})
        database.store(2 * ones(1), {})
        database.store(3 * ones(1), {})
        assert database.get_last_n_x(3) == [ones(1), 2 * ones(1), 3 * ones(1)]
        assert database.get_last_n_x(2) == [2 * ones(1), 3 * ones(1)]

        with pytest.raises(ValueError):
            database.get_last_n_x(4)
