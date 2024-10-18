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

from gemseo.post.animation_settings import AnimationSettings  # noqa:F401
from gemseo.post.basic_history_settings import BasicHistorySettings  # noqa:F401
from gemseo.post.constraints_history_settings import (  # noqa:F401
    ConstraintsHistorySettings,
)
from gemseo.post.correlations_settings import CorrelationsSettings  # noqa:F401
from gemseo.post.gradient_sensitivity_settings import (  # noqa:F401
    GradientSensitivitySettings,
)
from gemseo.post.hessian_history_settings import HessianHistorySettings  # noqa:F401
from gemseo.post.obj_constr_hist_settings import ObjConstrHistSettings  # noqa:F401
from gemseo.post.opt_history_view_settings import OptHistoryViewSettings  # noqa:F401
from gemseo.post.parallel_coordinates_settings import (  # noqa:F401
    ParallelCoordinatesSettings,
)
from gemseo.post.pareto_front_settings import ParetoFrontSettings  # noqa:F401
from gemseo.post.quad_approx_settings import QuadApproxSettings  # noqa:F401
from gemseo.post.radar_chart_settings import RadarChartSettings  # noqa:F401
from gemseo.post.robustness_settings import RobustnessSettings  # noqa:F401
from gemseo.post.scatter_plot_matrix_settings import (  # noqa:F401
    ScatterPlotMatrixSettings,
)
from gemseo.post.som_settings import SOMSettings  # noqa:F401
from gemseo.post.topology_view_settings import TopologyViewSettings  # noqa:F401
from gemseo.post.variable_influence_settings import (  # noqa:F401
    VariableInfluenceSettings,
)
