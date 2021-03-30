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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES


from __future__ import absolute_import, division, print_function, unicode_literals

from os.path import exists, join

import pytest
from future import standard_library
from matplotlib import pyplot as plt
from numpy import array
from numpy.random import rand, seed

from gemseo import SOFTWARE_NAME
from gemseo.algos.pareto_front import generate_pareto_plots, select_pareto_optimal
from gemseo.api import configure_logger

standard_library.install_aliases()


LOGGER = configure_logger(SOFTWARE_NAME)


def test_pareto_front(tmp_path):
    objs = array([[1, 2], [1.4, 1.7], [1.6, 1.6], [2, 1], [2, 2], [1.5, 1.5], [2, 0.5]])

    inds = select_pareto_optimal(objs)
    assert (inds == array([True, True, False, False, False, True, True])).all()

    generate_pareto_plots(objs, range(2))
    outfile = join(str(tmp_path), "Pareto_2d.png")
    plt.savefig(outfile)
    plt.close()
    assert exists(outfile)
    with pytest.raises(ValueError) as e_info:
        generate_pareto_plots(objs, range(3))


def test_5d(tmp_path):
    seed(1)
    n_obj = 5
    objs = rand(100, n_obj)
    inds = select_pareto_optimal(objs)
    assert sum(inds) > 0
    generate_pareto_plots(objs, range(n_obj))
    outfile = join(str(tmp_path), "Pareto_5d.png")
    plt.savefig(outfile)
    plt.close()
    assert exists(outfile)
