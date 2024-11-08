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
"""The DOE used by the Morris sensitivity analysis."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Optional
from typing import TextIO
from typing import Union

from numpy import vstack

from gemseo.algos.doe.base_doe_library import BaseDOELibrary
from gemseo.algos.doe.base_doe_library import DOEAlgorithmDescription
from gemseo.algos.doe.factory import DOELibraryFactory
from gemseo.algos.doe.morris_doe.settings.morris_doe_settings import MorrisDOE_Settings
from gemseo.typing import MutableStrKeyMapping
from gemseo.typing import RealArray

if TYPE_CHECKING:
    from gemseo.algos.design_space import DesignSpace

OptionType = Optional[Union[str, int, float, bool, list[str], Path, TextIO, RealArray]]


class MorrisDOE(BaseDOELibrary):
    """The DOE used by the Morris sensitivity analysis.

    This DOE algorithm applies the :class:`.OATDOE` algorithm at :math:`r` points.
    The number of samples is equal to :math:`r(1+d)`
    where :math:`d` is the space dimension.
    """

    ALGORITHM_INFOS: ClassVar[dict[str, DOEAlgorithmDescription]] = {
        "MorrisDOE": DOEAlgorithmDescription(
            algorithm_name="MorrisDOE",
            description="The DOE used by the Morris sensitivity analysis.",
            internal_algorithm_name="MorrisDOE",
            library_name="MorrisDOE",
            Settings=MorrisDOE_Settings,
        )
    }

    def __init__(self, algo_name: str = "MorrisDOE") -> None:  # noqa:D107
        super().__init__(algo_name)

    def _generate_unit_samples(
        self,
        design_space: DesignSpace,
        n_samples: int,
        doe_algo_name: str,
        doe_algo_settings: MutableStrKeyMapping,
        step: float,
    ) -> RealArray:
        """
        Args:
            n_samples: The maximum number of samples required by the user.
                If 0,
                deduce it from the design space dimension and ``doe_algo_settings``.
            doe_algo_name: The name of the DOE algorithm to repeat the OAT DOE.
            doe_algo_settings: The settings of the DOE algorithm to repeat the OAT DOE.
            step: The relative step of the OAT DOE.

        Raises:
            ValueError: When the number of samples is lower than
                the dimension of the input space plus one.
        """  # noqa: D205, D212
        factory = DOELibraryFactory()
        doe_algo = factory.create(doe_algo_name)
        oat_algo = factory.create("OATDOE")
        dimension = design_space.dimension
        n_replicates = doe_algo_settings.get("n_samples", 5)
        if n_samples > 0:
            n_replicates = n_samples // (dimension + 1)
            if n_replicates == 0:
                msg = (
                    f"The number of samples ({n_samples}) must be "
                    "at least equal to the dimension of the input space plus one "
                    f"({dimension}+1={dimension + 1})."
                )
                raise ValueError(msg)

        # If possible, set the number of samples of the DOE algorithm
        n_samples_available = set(
            doe_algo.ALGORITHM_INFOS[doe_algo_name].Settings.model_fields
        ).intersection(["n_samples", "samples"])
        if n_samples_available and doe_algo_name != "CustomDOE":
            doe_algo_settings[n_samples_available.pop()] = n_replicates

        base_options = {"variables_space": dimension, "unit_sampling": True}
        initial_points = doe_algo.compute_doe(**base_options, **doe_algo_settings)
        return vstack([
            oat_algo.compute_doe(**base_options, step=step, initial_point=initial_point)
            for initial_point in initial_points
        ])
