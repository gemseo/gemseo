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
"""Settings of the machine learning algorithms."""

from __future__ import annotations

from gemseo.mlearning.classification.algos.knn_settings import (  # noqa: F401
    KNNClassifierSettings,
)
from gemseo.mlearning.classification.algos.random_forest_settings import (  # noqa: F401
    RandomForestClassifierSettings,
)
from gemseo.mlearning.classification.algos.svm_settings import (  # noqa: F401
    SVMClassifierSettings,
)
from gemseo.mlearning.clustering.algos.gaussian_mixture_settings import (  # noqa: F401
    GaussianMixtureSettings,
)
from gemseo.mlearning.clustering.algos.kmeans_settings import (  # noqa: F401
    KMeansSettings,
)
from gemseo.mlearning.regression.algos.gpr_settings import (  # noqa: F401
    GaussianProcessRegressorSettings,
)
from gemseo.mlearning.regression.algos.gradient_boosting_settings import (  # noqa: F401
    GradientBoostingRegressorSettings,
)
from gemseo.mlearning.regression.algos.linreg_settings import (  # noqa: F401
    LinearRegressorSettings,
)
from gemseo.mlearning.regression.algos.mlp_settings import (  # noqa: F401
    MLPRegressorSettings,
)
from gemseo.mlearning.regression.algos.moe_settings import MOESettings  # noqa: F401
from gemseo.mlearning.regression.algos.ot_gpr_settings import (  # noqa: F401
    OTGaussianProcessRegressorSettings,
)
from gemseo.mlearning.regression.algos.pce_settings import (  # noqa: F401
    PCERegressorSettings,
)
from gemseo.mlearning.regression.algos.polyreg_settings import (  # noqa: F401
    PolynomialRegressorSettings,
)
from gemseo.mlearning.regression.algos.random_forest_settings import (  # noqa: F401
    RandomForestRegressorSettings,
)
from gemseo.mlearning.regression.algos.rbf_settings import (  # noqa: F401
    RBFRegressorSettings,
)
from gemseo.mlearning.regression.algos.regressor_chain_settings import (  # noqa: F401
    RegressorChainSettings,
)
from gemseo.mlearning.regression.algos.svm_settings import (  # noqa: F401
    SVMRegressorSettings,
)
from gemseo.mlearning.regression.algos.thin_plate_spline_settings import (  # noqa: F401
    TPSRegressorSettings,
)
