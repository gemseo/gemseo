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
"""Optimization and DOE history post-processing and analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Final

from gemseo.util.package_import import install_lazy_reexport

if TYPE_CHECKING:
    # static visibility for mypy / IDEs
    from gemseo.post.animation import Animation  # noqa: F401
    from gemseo.post.animation_settings import Animation_Settings  # noqa: F401
    from gemseo.post.basic_history import BasicHistory  # noqa: F401
    from gemseo.post.basic_history_settings import BasicHistory_Settings  # noqa: F401
    from gemseo.post.constraint_radar import ConstraintRadar  # noqa: F401
    from gemseo.post.constraint_radar_settings import (
        ConstraintRadar_Settings,  # noqa: F401
    )
    from gemseo.post.constraints_history import ConstraintsHistory  # noqa: F401
    from gemseo.post.constraints_history_settings import (
        ConstraintsHistory_Settings,  # noqa: F401
    )
    from gemseo.post.correlations import Correlations  # noqa: F401
    from gemseo.post.correlations_settings import Correlations_Settings  # noqa: F401
    from gemseo.post.factory import POST_FACTORY  # noqa: F401
    from gemseo.post.gradient_sensitivity import GradientSensitivity  # noqa: F401
    from gemseo.post.gradient_sensitivity_settings import (
        GradientSensitivity_Settings,  # noqa: F401
    )
    from gemseo.post.hessian_history import HessianHistory  # noqa: F401
    from gemseo.post.hessian_history_settings import (
        HessianHistory_Settings,  # noqa: F401
    )
    from gemseo.post.obj_constr_hist import ObjConstrHist  # noqa: F401
    from gemseo.post.obj_constr_hist_settings import (
        ObjConstrHist_Settings,  # noqa: F401
    )
    from gemseo.post.opt_history_view import OptHistoryView  # noqa: F401
    from gemseo.post.opt_history_view_settings import (
        OptHistoryView_Settings,  # noqa: F401
    )
    from gemseo.post.parallel_coordinates import ParallelCoordinates  # noqa: F401
    from gemseo.post.parallel_coordinates_settings import (
        ParallelCoordinates_Settings,  # noqa: F401
    )
    from gemseo.post.pareto_front import ParetoFront  # noqa: F401
    from gemseo.post.pareto_front_settings import ParetoFront_Settings  # noqa: F401
    from gemseo.post.quad_approx import QuadApprox  # noqa: F401
    from gemseo.post.quad_approx_settings import QuadApprox_Settings  # noqa: F401
    from gemseo.post.robustness import Robustness  # noqa: F401
    from gemseo.post.robustness_settings import Robustness_Settings  # noqa: F401
    from gemseo.post.scatter_plot_matrix import ScatterPlotMatrix  # noqa: F401
    from gemseo.post.scatter_plot_matrix_settings import (
        ScatterPlotMatrix_Settings,  # noqa: F401
    )
    from gemseo.post.som import SOM  # noqa: F401
    from gemseo.post.som_settings import SOM_Settings  # noqa: F401
    from gemseo.post.topology_view import TopologyView  # noqa: F401
    from gemseo.post.topology_view_settings import TopologyView_Settings  # noqa: F401
    from gemseo.post.variable_influence import VariableInfluence  # noqa: F401
    from gemseo.post.variable_influence_settings import (
        VariableInfluence_Settings,  # noqa: F401
    )

# Class name -> defining submodule (lazy-loaded on attribute access).
_NAME_TO_LOCATION: Final[dict[str, str]] = {
    "Animation": "animation",
    "Animation_Settings": "animation_settings",
    "BasicHistory": "basic_history",
    "BasicHistory_Settings": "basic_history_settings",
    "ConstraintRadar": "constraint_radar",
    "ConstraintRadar_Settings": "constraint_radar_settings",
    "ConstraintsHistory": "constraints_history",
    "ConstraintsHistory_Settings": "constraints_history_settings",
    "Correlations": "correlations",
    "Correlations_Settings": "correlations_settings",
    "GradientSensitivity": "gradient_sensitivity",
    "GradientSensitivity_Settings": "gradient_sensitivity_settings",
    "HessianHistory": "hessian_history",
    "HessianHistory_Settings": "hessian_history_settings",
    "ObjConstrHist": "obj_constr_hist",
    "ObjConstrHist_Settings": "obj_constr_hist_settings",
    "OptHistoryView": "opt_history_view",
    "OptHistoryView_Settings": "opt_history_view_settings",
    "ParallelCoordinates": "parallel_coordinates",
    "ParallelCoordinates_Settings": "parallel_coordinates_settings",
    "ParetoFront": "pareto_front",
    "ParetoFront_Settings": "pareto_front_settings",
    "POST_FACTORY": "factory",
    "QuadApprox": "quad_approx",
    "QuadApprox_Settings": "quad_approx_settings",
    "Robustness": "robustness",
    "Robustness_Settings": "robustness_settings",
    "ScatterPlotMatrix": "scatter_plot_matrix",
    "ScatterPlotMatrix_Settings": "scatter_plot_matrix_settings",
    "SOM": "som",
    "SOM_Settings": "som_settings",
    "TopologyView": "topology_view",
    "TopologyView_Settings": "topology_view_settings",
    "VariableInfluence": "variable_influence",
    "VariableInfluence_Settings": "variable_influence_settings",
}

install_lazy_reexport(globals(), _NAME_TO_LOCATION)
