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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import absolute_import, division, unicode_literals

import unittest
from builtins import map, object, range, str, zip
from timeit import default_timer as timer

from future import standard_library
from numpy import array, complex128, equal, ones
from scipy.optimize import rosen

from gemseo import SOFTWARE_NAME
from gemseo.api import configure_logger, create_discipline
from gemseo.core.function import MDOFunctionGenerator
from gemseo.core.parallel_execution import (
    DiscParallelExecution,
    DiscParallelLinearization,
    ParallelExecution,
)
from gemseo.problems.sellar.sellar import Sellar1, Sellar2, SellarSystem, get_inputs
from gemseo.utils.testing_utils import skip_under_windows

standard_library.install_aliases()

LOGGER = configure_logger(SOFTWARE_NAME)


class CallableWorker(object):
    """Callable worker"""

    def __init__(self):
        """Constructor"""
        pass

    def __call__(self, counter):
        """Callable"""
        return 2 * counter


def function_raising_exception(counter):
    """Raises an Exception"""
    raise Exception("This is an Exception")


class TestParallelExecution(unittest.TestCase):
    """Test the parallel execution"""

    @skip_under_windows
    def test_functional(self):
        n = 10
        function_list = [rosen] * n
        parallel_execution = ParallelExecution(function_list)
        output_list = parallel_execution.execute([[0.5] * i for i in range(1, n + 1)])
        assert output_list == [rosen([0.5] * i) for i in range(1, n + 1)]
        self.assertRaises(
            ValueError,
            parallel_execution.execute,
            [[0.5] * i for i in range(1, n + 10)],
        )

        self.assertRaises(
            TypeError,
            parallel_execution.execute,
            [[0.5] * i for i in range(1, n + 1)],
            exec_callback="toto",
        )

        function_list = [rosen] * n
        parallel_execution = ParallelExecution(function_list)
        self.assertRaises(
            TypeError,
            parallel_execution.execute,
            [[0.5] * i for i in range(1, n + 1)],
            task_submitted_callback="not_callable",
        )

    def test_callable(self):
        """Test ParallelExecution with a Callable worker"""
        n = 2
        function_list = [CallableWorker(), CallableWorker()]
        parallel_execution = ParallelExecution(function_list, use_threading=True)
        output_list = parallel_execution.execute([1] * n)
        assert output_list == [2] * n

    def test_callable_exception(self):
        """Test ParallelExecution with a Callable worker"""
        n = 2
        function_list = [function_raising_exception, CallableWorker()]
        parallel_execution = ParallelExecution(function_list, use_threading=True)
        parallel_execution.execute([1] * n)

    @skip_under_windows
    def test_disc_parallel_doe(self):
        s_1 = Sellar1()
        n = 10
        disciplines = [s_1] * n
        parallel_execution = DiscParallelExecution(
            disciplines, n_processes=2, wait_time_between_fork=0.1
        )
        input_list = []
        for i in range(n):
            inpts = get_inputs()
            inpts["x_shared"][0] = i
            input_list.append(inpts)

        t_0 = timer()
        outs = parallel_execution.execute(input_list)
        t_f = timer()

        elapsed_time = t_f - t_0
        assert elapsed_time > 0.1 * (n - 1)

        assert s_1.n_calls == n

        func_gen = MDOFunctionGenerator(s_1)
        y_0_func = func_gen.get_function(["x_shared"], ["y_0"])

        parallel_execution = ParallelExecution([y_0_func] * n)
        input_list = [array([i, 0.0], dtype=complex128) for i in range(n)]
        output_list = parallel_execution.execute(input_list)

        for i in range(n):
            inpts = get_inputs()
            inpts["x_shared"][0] = i
            s_1.execute(inpts)
            assert s_1.local_data["y_0"] == outs[i]["y_0"]
            assert s_1.local_data["y_0"] == output_list[i]

    @skip_under_windows
    def test_parallel_lin(self):
        disciplines = [Sellar1(), Sellar2(), SellarSystem()]
        parallel_execution = DiscParallelLinearization(disciplines)

        input_list = []
        for i in range(3):
            inpts = get_inputs()
            inpts["x_shared"][0] = i + 1
            input_list.append(inpts)
        outs = parallel_execution.execute(input_list)

        disciplines2 = [Sellar1(), Sellar2(), SellarSystem()]

        for i, disc in enumerate(disciplines):
            inpts = get_inputs()
            inpts["x_shared"][0] = i + 1

            j_ref = disciplines2[i].linearize(inpts)

            for f, jac_loc in disc.jac.items():
                for x, dfdx in jac_loc.items():
                    assert (dfdx == j_ref[f][x]).all()
                    assert (dfdx == outs[i][f][x]).all()

    @skip_under_windows
    def test_disc_parallel_threading_proc(self):
        disciplines = [Sellar1(), Sellar2(), SellarSystem()]
        parallel_execution = DiscParallelExecution(
            disciplines, n_processes=2, use_threading=True
        )
        outs1 = parallel_execution.execute([None] * 3)

        disciplines = [Sellar1(), Sellar2(), SellarSystem()]
        parallel_execution = DiscParallelExecution(
            disciplines, n_processes=2, use_threading=False
        )
        outs2 = parallel_execution.execute([None] * 3)

        for out_d1, out_d2 in zip(outs1, outs2):
            for name, val in out_d2.items():
                assert equal(out_d1[name], val).all()

        disciplines = [Sellar1()] * 2
        self.assertRaises(
            ValueError,
            DiscParallelExecution,
            disciplines,
            n_processes=2,
            use_threading=True,
        )

    @skip_under_windows
    def test_async_call(self):

        disc = create_discipline("SobieskiMission")
        func = MDOFunctionGenerator(disc).get_function(["x_shared"], ["y_4"])

        def func2(x):
            LOGGER.info("computing light func2")
            return x.sum()

        x_list = [i * ones(6) for i in range(4)]

        def do_work():
            return list(map(func, x_list))

        par = ParallelExecution([func] * 2, n_processes=2, use_threading=False)
        LOGGER.info("Submitting heavy calculation")
        res = par.execute(
            [i * ones(6) + 1 for i in range(2)], task_submitted_callback=do_work
        )
        LOGGER.info("Got result heavy async exec " + str(res))
