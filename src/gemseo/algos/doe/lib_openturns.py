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
#        :author: Damien Guenot
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""OpenTURNS DOE algorithms."""
from __future__ import division, unicode_literals

import logging
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union

import openturns
from numpy import array, full
from numpy import max as np_max
from numpy import min as np_min
from numpy import ndarray
from packaging import version

from gemseo.algos.doe.doe_lib import DOELibrary
from gemseo.utils.string_tools import MultiLineString

OptionType = Optional[Union[str, int, float, bool, Sequence[int], ndarray]]

LOGGER = logging.getLogger(__name__)


class OpenTURNS(DOELibrary):
    """Library of OpenTURNS DOE algorithms."""

    __OT_WEBPAGE = (
        "http://openturns.github.io/openturns/latest/user_manual/"
        "_generated/openturns.{}.html"
    )

    # Available algorithm for DOE design
    OT_SOBOL = "OT_SOBOL"
    OT_RANDOM = "OT_RANDOM"
    OT_HASEL = "OT_HASELGROVE"
    OT_REVERSE_HALTON = "OT_REVERSE_HALTON"
    OT_HALTON = "OT_HALTON"
    OT_FAURE = "OT_FAURE"
    OT_MC = "OT_MONTE_CARLO"
    OT_FACTORIAL = "OT_FACTORIAL"
    OT_COMPOSITE = "OT_COMPOSITE"
    OT_AXIAL = "OT_AXIAL"
    OT_LHSO = "OT_OPT_LHS"
    OT_LHS = "OT_LHS"
    OT_LHSC = "OT_LHSC"
    OT_FULLFACT = "OT_FULLFACT"  # Box in openturns
    OT_SOBOL_INDICES = "OT_SOBOL_INDICES"
    __OT_METADATA = {
        OT_SOBOL: ("Sobol sequence", "SobolSequence"),
        OT_RANDOM: ("Random sampling", "RandomGenerator"),
        OT_HASEL: ("Haselgrove sequence", "HaselgroveSequence"),
        OT_REVERSE_HALTON: ("Reverse Halton", "ReverseHaltonSequence"),
        OT_HALTON: ("Halton sequence", "HaltonSequence"),
        OT_FAURE: ("Faure sequence", "FaureSequence"),
        OT_MC: ("Monte Carlo sequence", "RandomGenerator.html"),
        OT_FACTORIAL: ("Factorial design", "Factorial"),
        OT_COMPOSITE: ("Composite design", "Composite"),
        OT_AXIAL: ("Axial design", "Axial"),
        OT_LHSO: ("Optimal Latin Hypercube Sampling", "SimulatedAnnealingLHS"),
        OT_LHS: ("Latin Hypercube Sampling", "LHS"),
        OT_LHSC: ("Centered Latin Hypercube Sampling", "LHS"),
        OT_FULLFACT: ("Full factorial design", "Box"),
        OT_SOBOL_INDICES: ("DOE for Sobol 'indices", "SobolIndicesAlgorithm"),
    }

    # Optional parameters
    CENTER_KEYWORD = "centers"
    DOE_SETTINGS_OPTIONS = [
        DOELibrary.LEVEL_KEYWORD,
        CENTER_KEYWORD,
    ]
    CRITERION = "criterion"
    CRITERIA = {
        "C2": openturns.SpaceFillingC2,
        "PhiP": openturns.SpaceFillingPhiP,
        "MinDist": openturns.SpaceFillingMinDist,
    }
    TEMPERATURE = "temperature"
    TEMPERATURES = {
        "Geometric": openturns.GeometricProfile,
        "Linear": openturns.LinearProfile,
    }
    N_REPLICATES = "n_replicates"
    ANNEALING = "annealing"

    __DISCREPANCY_SEQUENCES = {
        OT_FAURE: openturns.FaureSequence,
        OT_HALTON: openturns.HaltonSequence,
        OT_HASEL: openturns.HaselgroveSequence,
        OT_SOBOL: openturns.SobolSequence,
        OT_REVERSE_HALTON: openturns.ReverseHaltonSequence,
    }

    __STRATIFIED_DOES = {
        OT_COMPOSITE: openturns.Composite,
        OT_AXIAL: openturns.Axial,
        OT_FACTORIAL: openturns.Factorial,
    }

    def __init__(self):
        super(OpenTURNS, self).__init__()
        self.__sequence = None
        for algo_name, algo_value in self.__OT_METADATA.items():
            self.lib_dict[algo_name] = {
                DOELibrary.LIB: self.__class__.__name__,
                DOELibrary.INTERNAL_NAME: algo_name,
                DOELibrary.DESCRIPTION: algo_value[0],
                DOELibrary.WEBSITE: self.__OT_WEBPAGE.format(algo_value[1]),
            }

    def _get_options(
        self,
        levels=None,  # type: Optional[int,Sequence[int]]
        centers=None,  # type: Optional[Sequence[int]]
        eval_jac=False,  # type: bool
        n_samples=None,  # type: Optional[int]
        n_processes=1,  # type: int
        wait_time_between_samples=0.0,  # type: float
        criterion="C2",  # type: str
        temperature="Geometric",  # type: str
        annealing=True,  # type: bool
        n_replicates=1000,  # type: int
        seed=1,  # type: int
        max_time=0,  # type: float
        **kwargs  # type: OptionType
    ):  # type: (...) -> Dict[str,OptionType]
        r"""Set the options.

        Args:
            levels: The levels for axial, full-factorial (box), factorial
                and composite designs. If None, the number of samples is
                used in order to deduce the levels.
            centers: The centers for axial, factorial and composite designs.
                If None, centers = 0.5.
            eval_jac: Whether to evaluate the jacobian.
            n_samples: The number of samples. If None, the algorithm uses
                the number of levels per input dimension provided by the
                argument ``levels``.
            n_processes: The number of processes.
            wait_time_between_samples: The waiting time between two samples.
            criterion: The space-filling criterion, either "C2", "PhiP" or "MinDist".
            temperature: The temperature profile for simulated annealing,
                either "Geometric" or "Linear".
            annealing: If True, use simulated annealing to optimize LHS. Otherwise,
                use crude Monte Carlo.
            n_replicates: The number of Monte Carlo replicates to optimize LHS.
            seed: The seed value.
            max_time: The maximum runtime in seconds,
                disabled if 0.
            **kwargs: The additional arguments.

        Returns:
            The processed options.
        """
        if centers is None:
            centers = [0.5]

        options = self._process_options(
            levels=levels,
            centers=centers,
            eval_jac=eval_jac,
            n_samples=n_samples,
            n_processes=n_processes,
            wait_time_between_samples=wait_time_between_samples,
            criterion=criterion,
            temperature=temperature,
            annealing=annealing,
            n_replicates=n_replicates,
            seed=seed,
            max_time=max_time,
            **kwargs
        )

        return options

    def __check_and_cast_levels(
        self,
        options,  # type: Mapping[str,Any]
    ):  # type: (...) -> None
        """Check that the options ``levels`` is properly defined and cast it to array.

        Args:
            options: The DOE options.

        Raises:
            ValueError: When a level does not belong to [0, 1].
            TypeError: When the levels are neither a list nor a tuple.
        """
        levels = options[self.LEVEL_KEYWORD]
        if isinstance(levels, (list, tuple)):
            levels = array(levels)
            lower_bound = np_min(levels)
            upper_bound = np_max(levels)
            if lower_bound < 0.0 or upper_bound > 1.0:
                raise ValueError(
                    "Levels must belong to [0, 1]; "
                    "got [{}, {}].".format(lower_bound, upper_bound)
                )
            options[self.LEVEL_KEYWORD] = levels
        else:
            raise TypeError(
                "The argument 'levels' must be either a list or a tuple; "
                "got a '{}'.".format(levels.__class__.__name__)
            )

    def __check_and_cast_centers(
        self,
        dimension,  # type: int
        options,  # type: Mapping[str,Any]
    ):  # type: (...) -> None
        """Check that the options ``centers`` is properly defined and cast it to array.

        Args:
            dimension: The parameter space dimension.
            options: The DOE options.

        Raises:
            ValueError: When the centers dimension has a wrong dimension.
            TypeError: When the centers are neither a list nor a tuple.
        """
        center = options[self.CENTER_KEYWORD]
        if isinstance(center, (list, tuple)):
            if len(center) != dimension:
                raise ValueError(
                    "Inconsistent length of 'centers' list argument "
                    "compared to design vector size: {} vs {}.".format(
                        dimension, len(center)
                    )
                )
            options[self.CENTER_KEYWORD] = array(center)
        else:
            raise TypeError(
                "Error for 'centers' definition in DOE design; "
                "a tuple or a list is expected whereas {} is provided.".format(
                    type(center)
                )
            )

    def _generate_samples(
        self, **options  # type: OptionType
    ):  # type: (...) -> ndarray
        """Generate the samples.

        Args:
            **options: The options for the algorithm, see associated JSON file.

        Returns:
            The samples for the DOE.
        """
        self.seed += 1
        openturns.RandomGenerator.SetSeed(options.get(self.SEED, self.seed))
        dimension = options.pop(self.DIMENSION)
        n_samples = options.pop(self.N_SAMPLES, None)

        LOGGER.info("Generation of %s DOE with OpenTurns", self.algo_name)

        if self.algo_name in (self.OT_LHS, self.OT_LHSC, self.OT_LHSO):
            return self.__generate_lhs(n_samples, dimension, **options)

        if self.algo_name in [self.OT_MC, self.OT_RANDOM]:
            return self.__generate_random(n_samples, dimension)

        if self.algo_name == self.OT_FULLFACT:
            levels = options.pop(self.LEVEL_KEYWORD, None)
            return self._generate_fullfact(dimension, n_samples, levels)

        if self.algo_name in self.__STRATIFIED_DOES:
            return self.__generate_stratified(dimension, options)

        if self.algo_name == self.OT_SOBOL_INDICES:
            return self.__generate_sobol(n_samples, dimension)

        if self.algo_name in self.__DISCREPANCY_SEQUENCES:
            algo = self.__DISCREPANCY_SEQUENCES[self.algo_name](dimension)
            return array(algo.generate(n_samples))

    def __check_stratified_options(
        self,
        dimension,  # type: int
        options,  # type: Mapping[str,Any]
    ):  # type: (...) -> None
        """Check that the mandatory inputs for the composite design are set.

        Args:
            dimension: The parameter space dimension.
            options: The options of the DOE.

        Raises:
            KeyError: If the key `levels` is not in `options`.
        """
        if self.LEVEL_KEYWORD not in options:
            raise KeyError(
                "Missing parameter 'levels', "
                "tuple of normalized levels in [0,1] you need in your design."
            )
        self.__check_and_cast_levels(options)
        if self.CENTER_KEYWORD in options:
            self.__check_and_cast_centers(dimension, options)
        else:
            options[self.CENTER_KEYWORD] = [0.5] * dimension

    def __generate_stratified(
        self,
        dimension,  # type: int
        options,  # type: Mapping[str,Any]
    ):  # type: (...) -> ndarray
        """Generate a DOE using the composite DOE algorithm.

        Args:
            dimension: The dimension of the parameter space.
            options: The DOE options.

        Returns:
            The samples.
        """
        self.__check_stratified_options(dimension, options)
        levels = options[self.LEVEL_KEYWORD]
        centers = options[self.CENTER_KEYWORD]
        msg = MultiLineString()
        msg.add("Composite design:")
        msg.indent()
        msg.add("centers: {}", centers)
        msg.add("levels: {}", levels)
        LOGGER.debug("%s", msg)
        algo = self.__STRATIFIED_DOES[self.algo_name](centers, levels)
        samples = self._rescale_samples(array(algo.generate()))
        return samples

    def __generate_lhs(
        self,
        n_samples,  # type: int
        dimension,  # type: int
        **options  # type: OptionType
    ):  # type: (...) -> ndarray
        """Generate a DOE using the LHS algorithm, possibly centered or optimized.

        Args:
            n_samples: The number of samples.
            dimension: The parameter space dimension.
            options: The DOE options.

        Returns:
            The samples.
        """

        lhs = openturns.LHSExperiment(
            self.__get_uniform_distribution(dimension), n_samples
        )
        if self.algo_name == self.OT_LHSO:
            lhs.setAlwaysShuffle(True)
            if options[self.ANNEALING]:
                if version.parse(openturns.__version__) < version.parse("1.17.0"):
                    algo = openturns.SimulatedAnnealingLHS(
                        lhs,
                        self.TEMPERATURES[options[self.TEMPERATURE]](),
                        self.CRITERIA[options[self.CRITERION]](),
                    )
                else:
                    algo = openturns.SimulatedAnnealingLHS(
                        lhs,
                        self.CRITERIA[options[self.CRITERION]](),
                        self.TEMPERATURES[options[self.TEMPERATURE]](),
                    )
                design = algo.generate()
            else:
                algo = openturns.MonteCarloLHS(lhs, options[self.N_REPLICATES])
                design = algo.generate()
        else:
            design = lhs.generate()

        samples = array(design)
        if self.algo_name == self.OT_LHSC:
            samples = self.__compute_centered_lhs(samples)
        return samples

    @staticmethod
    def __compute_centered_lhs(
        samples,  # type:ndarray
    ):  # type:(...) -> ndarray
        """Center the samples resulting from a Latin hypercube sampling.

        Args:
            samples: The samples resulting from a Latin hypercube sampling.

        Returns:
            The centered version of the initial samples.
        """
        n_samples = len(samples)
        centered_samples = (samples // (1.0 / n_samples) + 0.5) / n_samples
        return centered_samples

    @staticmethod
    def __get_uniform_distribution(
        dimension,  # type: int
    ):  # type: (...) -> openturns.ComposedDistribution
        return openturns.ComposedDistribution([openturns.Uniform(0.0, 1.0)] * dimension)

    def __generate_sobol(
        self,
        n_samples,  # type: int
        dimension,  # type: int
    ):  # type: (...) -> ndarray
        """Generate a DOE using a Sobol' sampling.

        Args:
            n_samples: The number of samples.
            dimension: The parameter space dimension.

        Returns:
            The samples.
        """
        n_samples = int(n_samples / (dimension + 2))
        algo = openturns.SobolIndicesExperiment(
            self.__get_uniform_distribution(dimension), n_samples
        )
        return array(algo.generate())

    def _generate_fullfact_from_levels(
        self,
        levels,  # type: Iterable[int]
    ):  # type: (...) -> ndarray
        # This method relies on openturns.Box.
        # This latter assumes that the levels provided correspond to the intermediate
        # levels between lower and upper bounds, while GEMSEO includes these bounds
        # in the definition of the levels, so we substract 2 in order to get
        # only intermediate levels.
        levels = [level - 2 for level in levels]

        # If any level is negative, we take them out, generate the DOE,
        # then append the DOE with 0.5 for the missing levels.
        ot_indices = []
        ot_levels = []
        for ot_index, ot_level in enumerate(levels):
            if ot_level >= 0:
                ot_levels.append(ot_level)
                ot_indices.append(ot_index)

        if not ot_levels:
            return full([1, len(levels)], 0.5)

        ot_doe = array(openturns.Box(ot_levels).generate())

        if len(ot_levels) == len(levels):
            return ot_doe

        doe = full([ot_doe.shape[0], len(levels)], 0.5)
        doe[:, ot_indices] = ot_doe
        return doe

    def __generate_random(
        self,
        n_samples,  # type: int
        dimension,  # type: int
    ):  # type: (...) -> ndarray
        """Generate a DOE using the random generator.

        Args:
            n_samples: The number of samples.
            dimension: The parameter space dimension.

        Returns:
            The samples.
        """
        samples = array(openturns.RandomGenerator.Generate(dimension * n_samples))
        return samples.reshape((n_samples, dimension), order="F")
