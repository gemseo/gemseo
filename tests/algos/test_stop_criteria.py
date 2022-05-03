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
from gemseo.algos.stop_criteria import is_f_tol_reached
from gemseo.algos.stop_criteria import is_x_tol_reached
from gemseo.problems.analytical.rosenbrock import Rosenbrock
from numpy import ones


def test_is_x_tol_reached():
    pb = Rosenbrock(l_b=0, u_b=1.0)
    pb.preprocess_functions()
    pb.objective(0 * ones(2))
    pb.objective(ones(2))

    assert not is_x_tol_reached(pb, x_tol_rel=0, x_tol_abs=0.1, n_x=2)
    pb.objective(1.01 * ones(2))
    assert is_x_tol_reached(pb, x_tol_rel=0, x_tol_abs=0.1 + 1e-13, n_x=2)
    assert not is_x_tol_reached(pb, x_tol_rel=0, x_tol_abs=0.001, n_x=2)

    assert not is_x_tol_reached(pb, x_tol_rel=0, x_tol_abs=0.2, n_x=3)
    assert is_x_tol_reached(pb, x_tol_rel=0.1, x_tol_abs=0.0, n_x=2)


def test_is_f_tol_reached():
    pb = Rosenbrock(l_b=0, u_b=1.0)
    pb.preprocess_functions()

    pb.objective(0 * ones(2))
    pb.objective(ones(2))

    # rosen(0,0)=1
    # rosen(1,1)=0
    #     assert not is_f_tol_reached(pb, f_tol_rel=0, f_tol_abs=0.4, n_x=2)
    # abs(1-0.5)<=1.*0.5
    assert is_f_tol_reached(pb, f_tol_rel=0, f_tol_abs=0.5, n_x=2)
    pb.objective(1.05 * ones(2))  # 0.278

    assert is_f_tol_reached(pb, f_tol_rel=0, f_tol_abs=0.2, n_x=2)
    assert not is_f_tol_reached(pb, f_tol_rel=0, f_tol_abs=0.001, n_x=2)

    assert not is_f_tol_reached(pb, f_tol_rel=0, f_tol_abs=0.2, n_x=3)
