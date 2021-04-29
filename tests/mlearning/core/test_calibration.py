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
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Test machine learning algorithm calibration."""
from __future__ import absolute_import, division, unicode_literals

from numpy import allclose, array, array_equal

from gemseo.algos.design_space import DesignSpace
from gemseo.mlearning.core.calibration import MLAlgoAssessor, MLAlgoCalibration
from gemseo.mlearning.qual_measure.mse_measure import MSEMeasure
from gemseo.problems.dataset.rosenbrock import RosenbrockDataset


def test_discipline():
    """Test discipline."""
    dataset = RosenbrockDataset(opt_naming=False)
    measure_options = {"method": "loo"}
    disc = MLAlgoAssessor(
        "PolynomialRegression", dataset, ["degree"], MSEMeasure, measure_options
    )
    result = disc.execute({"degree": array([3])})
    assert "degree" in result
    assert "criterion" in result
    assert "learning" in result
    assert allclose(result["criterion"], array([32107.67868617]), atol=1e0)
    assert array_equal(result["degree"], array([3]))


def test_calibration():
    """Test calibration."""
    dataset = RosenbrockDataset(opt_naming=False)
    calibration_space = DesignSpace()
    calibration_space.add_variable("degree", 1, "integer", 1, 10, 1)
    measure_options = {"method": "loo"}
    calibration = MLAlgoCalibration(
        "PolynomialRegression",
        dataset,
        ["degree"],
        calibration_space,
        MSEMeasure,
        measure_options,
    )
    calibration.execute({"algo": "fullfact", "n_samples": 10})
    x_opt = calibration.optimal_parameters
    f_opt = calibration.optimal_criterion
    algo_opt = calibration.optimal_algorithm
    assert allclose(f_opt, array([0.0]))
    assert not array_equal(x_opt["degree"], array([1]))
    assert algo_opt.parameters["degree"] == x_opt["degree"]
    degree = calibration.get_history("degree")
    criterion = calibration.get_history("criterion")
    train = calibration.get_history("learning")
    assert degree.shape == (10, 1)
    assert criterion.shape == (10, 1)
    assert train.shape == (10, 1)
    assert len(calibration.algos) == 10
