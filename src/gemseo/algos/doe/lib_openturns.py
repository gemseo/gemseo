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
"""OpenTUNRS DOE algorithms wrapper."""
from __future__ import division, unicode_literals

import logging
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import openturns
from matplotlib import pyplot as plt
from numpy import array, full
from numpy import max as np_max
from numpy import min as np_min
from numpy import ndarray
from openturns.viewer import View
from packaging import version

from gemseo.algos.doe.doe_lib import DOELibrary
from gemseo.utils.string_tools import MultiLineString

OptionType = Optional[Union[str, int, float, bool, Sequence[int], ndarray]]

LOGGER = logging.getLogger(__name__)


class OpenTURNS(DOELibrary):
    """OpenTURNS library of DOE algorithms wrapping."""

    OT_DOC = "http://openturns.github.io/openturns/master/user_manual/"
    # Available algorithm for DOE design
    OT_SOBOL = "OT_SOBOL"
    OT_SOBOL_DESC = "Sobol sequence implemented in openTURNS library"
    OT_SOBOL_WEB = OT_DOC + "_generated/openturns.SobolSequence.html"
    OT_RANDOM = "OT_RANDOM"
    OT_RANDOM_DESC = "Random sampling implemented in openTURNS library"
    OT_RANDOM_WEB = OT_DOC + "_generated/openturns.RandomGenerator.html"
    OT_HASEL = "OT_HASELGROVE"
    OT_HASEL_DESC = "Haselgrove sequence implemented in openTURNS library"
    OT_HASEL_WEB = OT_DOC + "_generated/openturns.HaselgroveSequence.html"
    OT_REVERSE_HALTON = "OT_REVERSE_HALTON"
    OT_REVERSE_HALTON_DESC = (
        "Reverse Halton sequence implemented" " in openTURNS library"
    )
    OT_REVERSE_HALTON_WEB = OT_DOC + "_generated/openturns.ReverseHaltonSequence.html"
    OT_HALTON = "OT_HALTON"
    OT_HALTON_DESC = "Halton sequence implemented in openTURNS library"
    OT_HALTON_WEB = OT_DOC + "_generated/openturns.HaltonSequence.html"
    OT_FAURE = "OT_FAURE"
    OT_FAURE_DESC = "Faure sequence implemented in openTURNS library"
    OT_FAURE_WEB = OT_DOC + "_generated/openturns.FaureSequence.html"
    OT_MC = "OT_MONTE_CARLO"
    OT_MC_DESC = "Monte Carlo sequence implemented in openTURNS library"
    OT_MC_WEB = OT_DOC + "_generated/openturns.RandomGenerator.html"
    OT_FACTORIAL = "OT_FACTORIAL"
    OT_FACTORIAL_DESC = "Factorial design implemented in openTURNS library"
    OT_FACTORIAL_WEB = OT_DOC + "_generated/openturns.Factorial.html"
    OT_COMPOSITE = "OT_COMPOSITE"
    OT_COMPOSITE_DESC = "Composite design implemented in openTURNS library"
    OT_COMPOSITE_WEB = OT_DOC + "_generated/openturns.Composite.html"
    OT_AXIAL = "OT_AXIAL"
    OT_AXIAL_DESC = "Axial design implemented in openTURNS library"
    OT_AXIAL_WEB = OT_DOC + "_generated/openturns.Axial.html"
    OT_LHSO = "OT_OPT_LHS"
    OT_LHSO_DESC = "Optimal Latin Hypercube Sampling implemented in openTURNS library"
    OT_LHSO_WEB = (
        "https://openturns.github.io/openturns/master/examples/optimal_lhs.html"
    )
    OT_LHS = "OT_LHS"
    OT_LHS_DESC = "Latin Hypercube Sampling implemented in openTURNS library"
    OT_LHS_WEB = OT_DOC + "_generated/openturns.LHS.html"
    OT_LHSC = "OT_LHSC"
    OT_LHSC_DESC = (
        "Centered Latin Hypercube Sampling implemented" " in openTURNS library"
    )
    OT_LHSC_WEB = OT_DOC + "_generated/openturns.LHS.html"
    OT_FULLFACT = "OT_FULLFACT"  # Box in openturns
    OT_FULLFACT_DESC = "Full factorial design implemented" "in openTURNS library"
    OT_FULLFACT_WEB = OT_DOC + "_generated/openturns.Box.html"
    OT_SOBOL_INDICES = "OT_SOBOL_INDICES"
    OT_SOBOL_INDICES_DESC = "Sobol indices"
    OT_SOBOL_INDICES_WEB = OT_DOC + "_generated/openturns.SobolIndicesAlgorithm.html"

    ALGO_LIST = [
        OT_SOBOL,
        OT_HASEL,
        OT_REVERSE_HALTON,
        OT_HALTON,
        OT_FAURE,
        OT_AXIAL,
        OT_FACTORIAL,
        OT_MC,
        OT_LHS,
        OT_LHSC,
        OT_LHSO,
        OT_RANDOM,
        OT_FULLFACT,
        OT_COMPOSITE,
        OT_SOBOL_INDICES,
    ]
    DESC_LIST = [
        OT_SOBOL_DESC,
        OT_HASEL_DESC,
        OT_REVERSE_HALTON_DESC,
        OT_HALTON_DESC,
        OT_FAURE_DESC,
        OT_AXIAL_DESC,
        OT_FACTORIAL_DESC,
        OT_MC_DESC,
        OT_LHS_DESC,
        OT_LHSC_DESC,
        OT_LHSO_DESC,
        OT_RANDOM_DESC,
        OT_FULLFACT_DESC,
        OT_COMPOSITE_DESC,
        OT_SOBOL_INDICES_DESC,
    ]
    WEB_LIST = [
        OT_SOBOL_WEB,
        OT_HASEL_WEB,
        OT_REVERSE_HALTON_WEB,
        OT_HALTON_WEB,
        OT_FAURE_WEB,
        OT_AXIAL_WEB,
        OT_FACTORIAL_WEB,
        OT_MC_WEB,
        OT_LHS_WEB,
        OT_LHSC_WEB,
        OT_LHSO_WEB,
        OT_RANDOM_WEB,
        OT_FULLFACT_WEB,
        OT_COMPOSITE_WEB,
        OT_SOBOL_INDICES_WEB,
    ]

    # Available distribution (only a part of what is available in openturns
    OT_ARCSINE = "Arcsine"
    OT_BETA = "Beta"
    OT_DIRICHLET = "Dirichlet"
    OT_NORMAL = "Normal"
    OT_TRUNCNORMAL = "TruncatedNormal"
    OT_TRIANGULAR = "Triangular"
    OT_TRAPEZOIDAL = "Trapezoidal"
    OT_UNIFORM = "Uniform"

    DISTRIBUTION_LIST = [
        OT_ARCSINE,
        OT_BETA,
        OT_DIRICHLET,
        OT_NORMAL,
        OT_TRUNCNORMAL,
        OT_TRIANGULAR,
        OT_TRAPEZOIDAL,
        OT_UNIFORM,
    ]

    # Optional parameters
    CENTER_KEYWORD = "centers"
    DISTRIBUTION_KEYWORD = "distribution_name"
    MEAN_KEYWORD = "mu"
    STD_KEYWORD = "sigma"
    START_KEYWORD = "start"
    END_KEYWORD = "end"
    DOE_SETTINGS_OPTIONS = [
        DOELibrary.LEVEL_KEYWORD,
        DISTRIBUTION_KEYWORD,
        MEAN_KEYWORD,
        STD_KEYWORD,
        START_KEYWORD,
        END_KEYWORD,
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

    # Default parameters
    DISTRIBUTION_DEFAULT = OT_UNIFORM

    def __init__(self):
        """Constructor Unless mentioned, DOE are normalized between [0,1]"""
        super(OpenTURNS, self).__init__()
        self.__distr_list = []
        self.__comp_dist = None
        self.__sequence = None
        for idx, algo in enumerate(self.ALGO_LIST):
            self.lib_dict[algo] = {
                DOELibrary.LIB: self.__class__.__name__,
                DOELibrary.INTERNAL_NAME: algo,
                DOELibrary.DESCRIPTION: self.DESC_LIST[idx],
                DOELibrary.WEBSITE: self.WEB_LIST[idx],
            }

    def _get_options(
        self,
        distribution_name="Uniform",  # type: str
        levels=None,  # type: Optional[int,Sequence[int]]
        centers=None,  # type: Optional[Sequence[int]]
        eval_jac=False,  # type: bool
        n_samples=None,  # type: Optional[int]
        mu=0.5,  # type: float
        sigma=None,  # type: Optional[float]
        start=0.25,  # type: float
        end=0.75,  # type: float
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
            distribution_name: The distribution name.
            levels: The levels for axial, full-factorial (box), factorial
                and composite designs. If None, the number of samples is
                used in order to deduce the levels.
            centers: The centers for axial, factorial and composite designs.
                If None, centers = 0.5.
            eval_jac: Whether to evaluate the jacobian.
            n_samples: The number of samples. If None, the algorithm uses
                the number of levels per input dimension provided by the
                argument ``levels``.
            mu: The mean of a random variable for beta, normal and
                truncated normal distributions.
            sigma: The standard deviation for beta, normal and
                truncated normal distributions. If None,
                :math:`\sigma = 0.447214 * 0.5`.
            start: The start level for the trapezoidal distribution.
            end: The end level for the trapezoidal distribution.
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
        if sigma is None:
            sigma = 0.447214 * 0.5
        wtbs = wait_time_between_samples
        popts = self._process_options(
            distribution_name=distribution_name,
            levels=levels,
            centers=centers,
            eval_jac=eval_jac,
            n_samples=n_samples,
            mu=mu,
            sigma=sigma,
            start=start,
            end=end,
            n_processes=n_processes,
            wait_time_between_samples=wtbs,
            criterion=criterion,
            temperature=temperature,
            annealing=annealing,
            n_replicates=n_replicates,
            seed=seed,
            max_time=max_time,
            **kwargs
        )

        return popts

    def __set_level_option(
        self, options  # type: Mapping[str,OptionType]
    ):  # type: (...) -> Dict[str,OptionType]
        """Check the `levels` option definition for a stratified DOE.

        Args:
            options: The options for the DOE.

        Returns:
            The options for the DOE.

        Raises:
            ValueError: If the upper bound of levels is greater than 1.
                If the lower bound of levels is lower than 0.
            TypeError: If `levels` is given with a wrong type.
        """
        option = options[self.LEVEL_KEYWORD]
        if isinstance(option, (list, tuple)):
            option = array(option)
            if np_max(option) > 1.0:
                raise ValueError(
                    "Upper bound of levels must be lower than or equal to 1; "
                    "{} given.".format(np_max(option))
                )
            if np_min(option) < 0.0:
                raise ValueError(
                    "Lower bound of levels must be greater than or equal to 0; "
                    "{} given.".format(np_min(option))
                )
            options[self.LEVEL_KEYWORD] = option
        else:
            raise TypeError(
                "Error for levels definition in DOE design; "
                "a tuple or a list is expected whereas {} is provided.".format(
                    type(option)
                )
            )
        return options

    def __set_center_option(
        self,
        dimension,  # type: int
        options,  # type: Mapping[str,OptionType]
    ):  # type: (...) -> Dict[str,OptionType]
        """Check the `centers` option definition for a stratified DOE.

        Args:
            dimension: The parameter space dimension.
            options: The options for the DOE.

        Returns:
            The options for the DOE.

        Raises:
            ValueError: If the length of `centers` is inconsistent with the
                design vector size.
            TypeError: If `centers` is given with the wrong type.
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
        return options

    def __get_distribution(
        self, options  # type: Mapping[str,OptionType]
    ):  # type: (...) -> Tuple[str, Dict[str,OptionType]]
        """Set the distribution to be used.

        If no distribution is provided (a name or a list of composed distributions)
        then the `DISTRIBUTION_DEFAULT` is used.

        Args:
            options: The options for the DOE.

        Returns:
            The distribution name and the options for the DOE.
        """
        if self.DISTRIBUTION_KEYWORD in options:
            distribution_name = options[self.DISTRIBUTION_KEYWORD]
            del options[self.DISTRIBUTION_KEYWORD]
        else:
            distribution_name = self.DISTRIBUTION_DEFAULT
        return distribution_name, options

    def _generate_samples(
        self, **options  # type: OptionType
    ):  # type: (...) -> ndarray
        """Generate the samples.

        Args:
            **options: The options for the algorithm,
                see associated JSON file.

        Returns:
            The samples for the DOE.
        """
        self.seed += 1
        dimension = options.pop(self.DIMENSION)
        n_samples = options.pop(self.N_SAMPLES, None)

        LOGGER.info("Generation of %s DOE with OpenTurns", self.algo_name)

        if self.algo_name in (self.OT_LHS, self.OT_LHSC, self.OT_LHSO):
            distribution_name, options = self.__get_distribution(options)
            samples = self.__generate_lhs(
                n_samples, dimension, distribution_name=distribution_name, **options
            )
        elif self.algo_name == self.OT_RANDOM:
            samples = self.__generate_random(n_samples, dimension, **options)
        elif self.algo_name == self.OT_MC:
            distribution_name, options = self.__get_distribution(options)
            samples = self.__generate_mc(
                n_samples, dimension, distribution_name=distribution_name, **options
            )
        elif self.algo_name == self.OT_FULLFACT:
            levels = options.pop(self.LEVEL_KEYWORD, None)
            samples = self._generate_fullfact(dimension, n_samples, levels)
        elif self.algo_name in (self.OT_COMPOSITE, self.OT_AXIAL, self.OT_FACTORIAL):
            options = self.__check_stratified_options(dimension, options)
            samples = self.__generate_stratified(options)
        elif self.algo_name == self.OT_SOBOL_INDICES:
            samples = self.__generate_sobol(n_samples, dimension, **options)
        else:
            samples = self.__generate_seq(n_samples, dimension)
        return samples

    @staticmethod
    def __check_float(
        options,  # type: Mapping[str,OptionType]
        keyword,  # type: str
        default=0,  # type: float
        u_b=None,  # type: Optional[float]
        l_b=None,  # type: Optional[float]
    ):  # type: (...) -> Dict[str,OptionType]
        """Check if `keyword` exists in `options` and set a default value.

        Args:
            options: The optional parameters.
            keyword: The name of the optional keyword.
            default: The default value to be set.
            u_b: The upper bound. If None, the value is not verified
                against this criterion.
            l_b: The lower bound. If None, the value is not verified
                against this criterion.

        Returns:
            The updated options.

        Raises:
            ValueError: If the value of the keyword is greater than the given
                upper bound.
                If the value of the keyword is lower than the given lower bound.
            TypeError: If the value associated to the keyword is not a float.
        """
        if keyword in options:
            opt = options[keyword]
            if not isinstance(opt, float):
                raise TypeError(
                    "{} value must be a float: {} given.".format(keyword, type(opt))
                )
            if u_b is not None:
                if opt > u_b:
                    raise ValueError(
                        "{} value must be lower than {}; "
                        "{} given.".format(keyword, u_b, opt)
                    )
            if l_b is not None:
                if opt < l_b:
                    raise ValueError(
                        "{} value must be greater than {}; "
                        "{} given.".format(keyword, l_b, opt)
                    )
            options[keyword] = opt
        else:
            options[keyword] = default
        return options

    def __check_stratified_options(
        self,
        dimension,  # type: int
        options,  # type: Mapping[str,OptionType]
    ):  # type: (...) -> Dict[str,OptionType]
        """Check that the mandatory inputs for the composite design are set.

        Args:
            dimension: The parameter space dimension.
            options: The options of the DOE.

        Returns:
            The updated options.

        Raises:
            KeyError: If the key `levels` is not in `options`.
        """
        if self.LEVEL_KEYWORD not in options:
            raise KeyError(
                "Missing  parameter 'levels', "
                "tuple of normalized levels in [0,1] you need in your design."
            )
        options = self.__set_level_option(options)
        if self.CENTER_KEYWORD not in options:
            options[self.CENTER_KEYWORD] = [0.5 for _ in range(dimension)]
        else:
            options = self.__set_center_option(dimension, options)
        return options

    def __generate_stratified(
        self, options  # type: Mapping[str,OptionType]
    ):  # type: (...) -> ndarray
        """Generate the samples of a DOE using the composite algo of openturns.

        Args:
            options: The options of the DOE.

        Returns:
            The samples for the DOE.
        """
        levels = options[self.LEVEL_KEYWORD]
        centers = options[self.CENTER_KEYWORD]
        msg = MultiLineString()
        msg.add("Composite design:")
        msg.indent()
        msg.add("centers: {}", centers)
        msg.add("levels: {}", levels)
        LOGGER.debug("%s", msg)
        if self.algo_name == self.OT_COMPOSITE:
            experiment = openturns.Composite(centers, levels)
        elif self.algo_name == self.OT_AXIAL:
            experiment = openturns.Axial(centers, levels)
        elif self.algo_name == self.OT_FACTORIAL:
            experiment = openturns.Factorial(centers, levels)

        samples = array(experiment.generate())
        samples = self._rescale_samples(samples)
        return samples

    def __generate_seq(
        self,
        n_samples,  # type: int
        dimension,  # type: int
    ):  # type: (...) -> ndarray
        """Generate the samples of a DOE using the LHS algo of openturns.

        Args:
            n_samples: The number of samples for the DOE.
            dimension: The parameter space dimension.

        Rreturns:
            The samples for the DOE.
        """
        if self.algo_name == self.OT_FAURE:
            self.__sequence = openturns.FaureSequence
        elif self.algo_name == self.OT_HALTON:
            self.__sequence = openturns.HaltonSequence
        elif self.algo_name == self.OT_REVERSE_HALTON:
            self.__sequence = openturns.ReverseHaltonSequence
        elif self.algo_name == self.OT_HASEL:
            self.__sequence = openturns.HaselgroveSequence
        elif self.algo_name == self.OT_SOBOL:
            self.__sequence = openturns.SobolSequence
        seq = self.__sequence(dimension).generate(n_samples)
        return array(seq)

    def create_composed_distributions(self):  # type: (...) -> None
        """Create a composed distribution from a list of distributions."""
        self.__comp_dist = openturns.ComposedDistribution(self.__distr_list)

    def get_composed_distributions(
        self,
    ):  # type: (...) -> openturns.ComposedDistribution
        """Return the composed distributions.

        Returns:
            The composed distribution.
        """
        return self.__comp_dist

    def __check_composed_distribution(
        self,
        distribution_name,  # type: str
        dimension,  # type: int
    ):  # type: (...) -> None
        """Check the composed distribution.

        Args:
            distribution_name: The name of the distribution.
            dimension: The parameter space dimension.

        Raises:
            ValueError: If the given dimension is different from the
                the one of the problem.
                If the ComposedDistribution does not match the given
                dimension.
        """
        if self.__comp_dist is None:
            n_distrib = len(self.__distr_list)
            if n_distrib == 0:
                LOGGER.debug(
                    "Creating default composed distribution based on %s.",
                    distribution_name,
                )
                self.create_distribution(distribution_name)
                self.__comp_dist = openturns.ComposedDistribution(
                    [self.__distr_list[0] for _ in range(dimension)]
                )
            elif n_distrib == 1:
                # Only one distribution was defined ==> duplicating it in all
                # dimensions
                self.__comp_dist = openturns.ComposedDistribution(
                    [self.__distr_list[0] for _ in range(dimension)]
                )
            elif n_distrib != dimension:
                raise ValueError(
                    "Size mismatch between number of distribution and problem: "
                    "{} vs {}.".format(dimension, n_distrib)
                )
        elif self.__comp_dist.getDimension() != dimension:
            raise ValueError(
                "Size mismatch between ComposedDistribution and problem: "
                "{} vs. {}".format(dimension, self.__comp_dist.getDimension())
            )
        else:
            LOGGER.debug(
                "Using composed distribution previously created: %s.",
                self.__comp_dist.getDistributionCollection(),
            )

    def check_distribution_name(
        self, distribution_name  # type: str
    ):  # type: (...) -> None
        """Check that the distribution is available.

        Args:
            distribution_name: The name of the distribution.

        Raises:
            ValueError: If the distribution is not available.
        """
        if distribution_name not in self.DISTRIBUTION_LIST:
            raise ValueError(
                "Distribution '{}' is not available; "
                "please switch to one of the followings: {}.".format(
                    distribution_name, self.DISTRIBUTION_LIST
                )
            )

    def create_distribution(
        self,
        distribution_name="Uniform",  # type: str
        **options  # type: OptionType
    ):  # type: (...) -> None
        """Create a distribution for all the design vectors.

        Also add it to the list of distributions.

        Args:
            distribution_name: The name of the distribution.
            **options: The openturns distribution options.
        """
        self.check_distribution_name(distribution_name)
        options = self.__check_float(options, self.MEAN_KEYWORD, 0.5, u_b=1, l_b=0)
        options = self.__check_float(options, self.CENTER_KEYWORD, 0.5, u_b=1, l_b=0)
        options = self.__check_float(options, self.START_KEYWORD, 0.25, u_b=1, l_b=0)
        options = self.__check_float(options, self.END_KEYWORD, 0.75, u_b=1, l_b=0)

        if distribution_name == self.OT_UNIFORM:
            LOGGER.debug("Creation of a uniform distribution.")
            self.__distr_list.append(openturns.Uniform(0, 1))
        elif distribution_name == self.OT_TRIANGULAR:
            dist_center = options[self.CENTER_KEYWORD]
            LOGGER.debug(
                "Creation of a triangular distribution with center %s.", dist_center
            )
            self.__distr_list.append(openturns.Triangular(0.0, dist_center, 1.0))
        elif distribution_name == self.OT_TRAPEZOIDAL:
            lower_bound = options[self.START_KEYWORD]
            upper_bound = options[self.END_KEYWORD]
            LOGGER.debug(
                "Creation of a trapezoidal distribution "
                "with lower bound %s and upper bound %s.",
                lower_bound,
                upper_bound,
            )
            self.__distr_list.append(
                openturns.Trapezoidal(0.0, lower_bound, upper_bound, 1.0)
            )
        elif distribution_name == self.OT_BETA:
            mean = options[self.MEAN_KEYWORD]
            options = self.__check_float(
                options, self.STD_KEYWORD, 0.447214 * 0.5, u_b=1, l_b=0
            )
            std = options[self.STD_KEYWORD]
            LOGGER.debug(
                "Creation of a %s distribution: mu (mean) %g and sigma (std) %g",
                distribution_name,
                mean,
                std,
            )
            beta = openturns.Beta()
            beta.setParameter(openturns.BetaMuSigma()([mean, std, 0.0, 1.0]))
            self.__distr_list.append(beta)
        elif distribution_name == self.OT_ARCSINE:
            LOGGER.debug("Creation of a %s.", distribution_name)
            arcs = openturns.Arcsine(0.0, 1.0)
            self.__distr_list.append(arcs)
        elif distribution_name == self.OT_TRUNCNORMAL:
            mean = options[self.MEAN_KEYWORD]
            options = self.__check_float(
                options, self.STD_KEYWORD, 0.5 / 3.75, u_b=1, l_b=0
            )
            std = options[self.STD_KEYWORD]
            LOGGER.debug(
                "Creation of a %s distribution: mu (mean) %g and" "sigma (std) %g.",
                distribution_name,
                mean,
                std,
            )
            nrmal = openturns.TruncatedNormal(mean, std, 0.0, 1.0)
            self.__distr_list.append(nrmal)
        elif distribution_name == self.OT_NORMAL:
            mean = options[self.MEAN_KEYWORD]
            options = self.__check_float(
                options, self.STD_KEYWORD, 0.5 / 3.75, u_b=1, l_b=0
            )
            std = options[self.STD_KEYWORD]
            LOGGER.debug(
                "Creation of a %s distribution: mu (mean) %g and" "sigma (std) %g.",
                distribution_name,
                mean,
                std,
            )
            self.__distr_list.append(openturns.Normal(mean, std))

    def display_distributions_list(self):  # type: (...) -> None
        """Display the distributions in use for DOE's based on LHS or Monte-Carlo."""
        msg = MultiLineString()
        msg.add("List of distributions:")
        msg.indent()
        for distribution in self.__distr_list:
            msg.add(str(distribution))
        LOGGER.info("%s", msg)

    def get_distributions_list(self):  # type: (...) -> List[str]
        """Accessor for the distributions list.

        Returns:
            The distribution list.
        """
        return self.__distr_list

    def __generate_lhs(
        self,
        n_samples,  # type: int
        dimension,  # type: int
        distribution_name="Uniform",  # type: str
        **options  # type: OptionType
    ):  # type: (...) -> ndarray
        """Generate the samples for a DOE using the LHS algo of openturns.

        Args:
            n_samples: The number of samples for the DOE.
            dimension: The parameter space dimension.
            distribution_name: The name of the distribution.
            **options: The options of the DOE.

        Returns:
            The samples for the DOE.

        Raises:
            ValueError: If the given `criterion` is not available.
                If the given `temperature` is not available.
        """
        self.__check_composed_distribution(
            distribution_name=distribution_name, dimension=dimension
        )

        seed = options.get(self.SEED, self.seed)
        openturns.RandomGenerator.SetSeed(seed)
        lhs = openturns.LHSExperiment(self.__comp_dist, n_samples)
        if self.algo_name == self.OT_LHSO:
            lhs.setAlwaysShuffle(True)
            criterion = options.get(self.CRITERION, "C2")
            try:
                criterion = self.CRITERIA[criterion]()
            except KeyError:
                raise ValueError(
                    "{} is not an available criterion; "
                    "available ones are: {}.".format(criterion, self.CRITERIA)
                )
            annealing = options.get(self.ANNEALING, True)
            if annealing:
                temperature = options.get(self.TEMPERATURE, "Geometric")
                try:
                    temperature = self.TEMPERATURES[temperature]()
                except KeyError:
                    raise ValueError(
                        "{} is not an available temperature profile; "
                        "available ones are: {}.".format(temperature, self.TEMPERATURES)
                    )
                if version.parse(openturns.__version__) < version.parse("1.17.0"):
                    algo = openturns.SimulatedAnnealingLHS(lhs, temperature, criterion)
                else:
                    algo = openturns.SimulatedAnnealingLHS(lhs, criterion, temperature)
                design = algo.generate()
            else:
                n_replicates = options.get(self.N_REPLICATES, 1000)
                algo = openturns.MonteCarloLHS(lhs, n_replicates)
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

    def __generate_mc(
        self,
        n_samples,  # type: int
        dimension,  # type: int
        distribution_name="Uniform",  # type: str
        **options  # type: OptionType
    ):  # type: (...) -> ndarray
        """Generate the samples of a DOE using the Monte-Carlo algo of openturns.

        Args:
            n_samples: The number of samples for the DOE.
            dimension: The parameter space dimension.
            distribution_name: The name of the distribution.
            **options: The options of the DOE.

        Returns:
            The samples of the DOE.
        """
        self.__check_composed_distribution(
            distribution_name=distribution_name, dimension=dimension
        )

        seed = options.get(self.SEED, self.seed)
        openturns.RandomGenerator.SetSeed(seed)
        experiment = openturns.MonteCarloExperiment(self.__comp_dist, n_samples)
        return array(experiment.generate())

    def __generate_sobol(
        self,
        n_samples,  # type: int
        dimension,  # type: int
        **options  # type: OptionType
    ):  # type: (...) -> ndarray
        """Generate the samples of a DOE using Sobol sampling.

        Args:
            n_samples: The number of samples for the DOE.
            dimension: The parameter space dimension.
            **options: The options of the DOE.

        Returns:
            The samples for the DOE.
        """
        seed = options.get(self.SEED, self.seed)
        openturns.RandomGenerator.SetSeed(seed)
        self.__check_composed_distribution(
            distribution_name="Uniform", dimension=dimension
        )
        n_samples = int(n_samples / (dimension + 2))
        experiment = openturns.SobolIndicesExperiment(self.__comp_dist, n_samples)
        data = array(experiment.generate())
        return data

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
            doe = full([1, len(levels)], 0.5)
            return doe

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
        **options  # type: OptionType
    ):  # type: (...) -> ndarray
        """Generate the samples of a DOE using the random algo of openturns.

        Args:
            n_samples: The number of samples for the DOE.
            dimension: The parameter space dimension.
            **options: The options of the DOE.

        Returns:
            The samples for the DOE.
        """
        seed = options.get(self.SEED, self.seed)
        openturns.RandomGenerator.SetSeed(seed)
        samples_list = []
        for _ in range(n_samples):
            samples_list.append(openturns.RandomGenerator.Generate(dimension))
        return array(samples_list)

    @staticmethod
    def plot_distribution(
        distribution,  # type: openturns.Distribution
        show=False,  # type: bool
    ):  # type: (...) -> None
        """Plot the density PDF & the CDF (cumulative) of a given distribution.

        Args:
            distribution: The distribution to plot.
            show: Whether to show the plot.
        """
        distribution.setDescription(["x"])
        pdf_graph = distribution.drawPDF()
        cdf_graph = distribution.drawCDF()
        fig = plt.figure(figsize=(12, 5))
        plt.suptitle(str(distribution))
        pdf_axis = fig.add_subplot(121)
        cdf_axis = fig.add_subplot(122)
        mean = array(distribution.getMean())[0]
        std = array(distribution.getStandardDeviation())[0]
        View(pdf_graph, figure=fig, axes=[pdf_axis], add_legend=False)
        pdf_axis.axvline(x=mean, ymin=0.0, linewidth=1, linestyle="-.", color="k")
        pdf_axis.axvline(
            x=mean - std, linestyle="--", ymin=0.0, linewidth=0.5, color="k"
        )
        pdf_axis.axvline(
            x=mean + std, linestyle="--", ymin=0.0, linewidth=0.5, color="k"
        )
        pdf_axis.annotate(
            r"$\mu$",
            xy=(0.5, 1.05),
            xycoords="axes fraction",
            horizontalalignment="center",
            verticalalignment="center",
            size=15,
        )
        pdf_axis.annotate(
            r"$\mu+\sigma$",
            xy=(mean + std, 1.05),
            xycoords="axes fraction",
            horizontalalignment="center",
            verticalalignment="center",
            size=15,
        )
        pdf_axis.annotate(
            r"$\mu-\sigma$",
            xy=(mean - std, 1.05),
            xycoords="axes fraction",
            horizontalalignment="center",
            verticalalignment="center",
            size=15,
        )
        View(cdf_graph, figure=fig, axes=[cdf_axis], add_legend=False)
        if show:
            plt.show()
        plt.close()
