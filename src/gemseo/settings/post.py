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
"""Settings for the post-processing algorithms."""

from __future__ import annotations

from gemseo.post.animation_settings import Animation_Settings
from gemseo.post.basic_history_settings import BasicHistory_Settings
from gemseo.post.constraints_history_settings import ConstraintsHistory_Settings
from gemseo.post.correlations_settings import Correlations_Settings
from gemseo.post.gradient_sensitivity_settings import GradientSensitivity_Settings
from gemseo.post.hessian_history_settings import HessianHistory_Settings
from gemseo.post.obj_constr_hist_settings import ObjConstrHist_Settings
from gemseo.post.opt_history_view_settings import OptHistoryView_Settings
from gemseo.post.parallel_coordinates_settings import ParallelCoordinates_Settings
from gemseo.post.pareto_front_settings import ParetoFront_Settings
from gemseo.post.quad_approx_settings import QuadApprox_Settings
from gemseo.post.radar_chart_settings import RadarChart_Settings
from gemseo.post.robustness_settings import Robustness_Settings
from gemseo.post.scatter_plot_matrix_settings import ScatterPlotMatrix_Settings
from gemseo.post.som_settings import SOM_Settings
from gemseo.post.topology_view_settings import TopologyView_Settings
from gemseo.post.variable_influence_settings import VariableInfluence_Settings

__all__ = [
    "Animation_Settings",
    "BasicHistory_Settings",
    "ConstraintsHistory_Settings",
    "Correlations_Settings",
    "GradientSensitivity_Settings",
    "HessianHistory_Settings",
    "ObjConstrHist_Settings",
    "OptHistoryView_Settings",
    "ParallelCoordinates_Settings",
    "ParetoFront_Settings",
    "QuadApprox_Settings",
    "RadarChart_Settings",
    "Robustness_Settings",
    "SOM_Settings",
    "ScatterPlotMatrix_Settings",
    "TopologyView_Settings",
    "VariableInfluence_Settings",
]
