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
"""
OpenTUNRS DOE algorithms wrapper
********************************
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import openturns
from future import standard_library
from matplotlib import pyplot as plt
from numpy import array, empty_like
from numpy import max as np_max
from numpy import min as np_min
from openturns.viewer import View

from gemseo.algos.doe.doe_lib import DOELibrary

standard_library.install_aliases()


from gemseo import LOGGER


class OpenTURNS(DOELibrary):
    """OpenTURNS library of DOE algorithms wrapping"""

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
    LEVEL_KEYWORD = "levels"
    CENTER_KEYWORD = "centers"
    DISTRIBUTION_KEYWORD = "distribution_name"
    MEAN_KEYWORD = "mu"
    STD_KEYWORD = "sigma"
    START_KEYWORD = "start"
    END_KEYWORD = "end"
    DOE_SETTINGS_OPTIONS = [
        LEVEL_KEYWORD,
        DISTRIBUTION_KEYWORD,
        MEAN_KEYWORD,
        STD_KEYWORD,
        START_KEYWORD,
        END_KEYWORD,
        CENTER_KEYWORD,
    ]

    # Default parameters
    DISTRIBUTION_DEFAULT = OT_UNIFORM

    def __init__(self):
        """
        Constructor
        Unless mentioned, DOE are normalized between [0,1]
        """
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
        distribution_name="Uniform",  # pylint: disable=W0221
        levels=None,
        centers=None,
        eval_jac=False,
        n_samples=1,
        mu=0.5,
        sigma=None,
        start=0.25,
        end=0.75,
        n_processes=1,
        wait_time_between_samples=0.0,
        **kwargs
    ):
        """Sets the options

        :param distribution_name: distribution name
        :type distribution_name: str
        :param levels: levels for axial, factorial and composite designs
        :type levels: array
        :param centers: centers for axial, factorial and composite designs
        :type centers: array
        :param eval_jac: evaluate jacobian
        :type eval_jac: bool
        :param n_samples: number of samples
        :type n_samples: int
        :param mu: mean of a random variable for beta, normal and
            truncated normal distributions
        :type mu: float
        :param sigma: standard deviation for beta, normal and
            truncated normal distributions
        :type sigma: float
        :param start: level start for trapezoidal distribution
        :type start: float
        :param end: level end for trapezoidal distribution
        :type end: float
        :param n_processes: number of processes
        :type n_processes: int
        :param wait_time_between_samples: waiting time between two samples
        :type wait_time_between_samples: float
        :param kwargs: additional arguments
        """
        if levels is None:
            levels = [0.0, 0.25, 0.5]
        if centers is None:
            centers = [
                0.5,
            ]
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
            **kwargs
        )

        return popts

    def __set_level_option(self, options):
        """
        Check that level options is properly defined for stratified DOE

        :param options: the options dict for the DOE
        """
        option = options[self.LEVEL_KEYWORD]
        if isinstance(option, (list, tuple)):
            option = array(option)
            if np_max(option) > 1.0:
                raise ValueError(
                    "Upper bound of levels must be <=1: %s given" % np_max(option)
                )
            if np_min(option) < 0.0:
                raise ValueError(
                    "Lower bound of levels must be >=1: %s given" % np_min(option)
                )
            options[self.LEVEL_KEYWORD] = option
        else:
            raise TypeError(
                "Error for levels definition in DOE design:"
                + "a tuple or a list is expected whereas ",
                type(option),
                " is provided",
            )
        return options

    def __set_center_option(self, options):
        """
        Check that center level options is properly defined for stratified DOE

        :param options: the options dict for the DOE
        """
        center = options[self.CENTER_KEYWORD]
        dim = self.problem.dimension
        if isinstance(center, (list, tuple)):
            if len(center) != dim:
                raise ValueError(
                    "Inconsistent length of 'centers' list argument "
                    + "compared to design vector size: %s vs %s" % (dim, len(center))
                )
            options[self.CENTER_KEYWORD] = array(center)
        else:
            raise TypeError(
                "Error for 'centers' definition in DOE design:"
                + "a tuple or a list is expected whereas ",
                type(center),
                " is provided",
            )
        return options

    def __get_distribution(self, options):
        """
        If no distribution is provided (a name or a list of composed
        distributions) then a default setting is done

        :param options: the options dict for the distribution
        """
        if self.DISTRIBUTION_KEYWORD in options:
            distribution_name = options[self.DISTRIBUTION_KEYWORD]
            del options[self.DISTRIBUTION_KEYWORD]
        else:
            distribution_name = self.DISTRIBUTION_DEFAULT
        return distribution_name, options

    def _generate_samples(self, **options):
        """
        Generates the list of x samples

        :param options: the options dict for the algorithm,
            see associated JSON file
        """

        LOGGER.info("Generation of %s DOE with OpenTurns", self.algo_name)
        if self.algo_name in (self.OT_LHS, self.OT_LHSC):
            distribution_name, options = self.__get_distribution(options)
            samples = self.__generate_lhs(
                distribution_name=distribution_name, **options
            )
        elif self.algo_name == self.OT_RANDOM:
            samples = self.__generate_random(options[self.N_SAMPLES])
        elif self.algo_name == self.OT_MC:
            distribution_name, options = self.__get_distribution(options)
            samples = self.__generate_mc(distribution_name=distribution_name, **options)
        elif self.algo_name == self.OT_FULLFACT:
            samples = self.__generate_fullfact(options[self.N_SAMPLES])
        elif self.algo_name in (self.OT_COMPOSITE, self.OT_AXIAL, self.OT_FACTORIAL):
            options = self.__check_stratified_options(options)
            samples = self.__generate_stratified(options)
        elif self.algo_name == self.OT_SOBOL_INDICES:
            samples = self.__generate_sobol(options[self.N_SAMPLES])
        else:
            samples = self.__generate_seq(options[self.N_SAMPLES])
        return samples

    @staticmethod
    def __check_float(options, keyword, default=0, u_b=None, l_b=None):
        """
        Base function to check if the keyword exist in dictionary
        and set a default value

        :param options: dictionary of optional parameters
        :type  options: dictionary
        :param keyword: name of the optional keyword
        :type  keyword: string
        :param default: default value
        :type  default: float
        :param u_b: upper bound
        :type  u_b: float
        :param l_b: lower bound
        :type  l_b: float
        """
        if keyword in options:
            opt = options[keyword]
            if not isinstance(opt, float):
                raise TypeError(
                    keyword + " value must be a float : ", type(opt), " given"
                )
            if u_b is not None:
                if opt > u_b:
                    raise ValueError(
                        keyword
                        + " value must be < "
                        + str(u_b)
                        + " : "
                        + str(opt)
                        + " given"
                    )
            if l_b is not None:
                if opt < l_b:
                    raise ValueError(
                        keyword
                        + " value must be > "
                        + str(l_b)
                        + " : "
                        + str(opt)
                        + " given"
                    )
            options[keyword] = opt
        else:
            options[keyword] = default
        return options

    def __check_stratified_options(self, options):
        """
        Check that mandatory inputs for composite design are set

        :param options: the options
        """
        dim = self.problem.dimension
        if self.LEVEL_KEYWORD not in options:
            raise KeyError(
                "Missing  parameter 'levels', "
                + "tuple of normalized levels  "
                + "in [0,1] you need in your design"
            )
        options = self.__set_level_option(options)
        if self.CENTER_KEYWORD not in options:
            options[self.CENTER_KEYWORD] = [0.5 for _ in range(dim)]
        else:
            options = self.__set_center_option(options)
        return options

    def __generate_stratified(self, options):
        """
        Generate a DOE using composite algo of openturns

        :param options: the options
        :returns: samples
        :rtype: numpy array
        """
        levels = options[self.LEVEL_KEYWORD]
        centers = options[self.CENTER_KEYWORD]
        LOGGER.info("Composite design:")
        LOGGER.info("    - centers: %s", str(centers))
        LOGGER.info("    - levels: %s", str(levels))
        if self.algo_name == self.OT_COMPOSITE:
            experiment = openturns.Composite(centers, levels)
        elif self.algo_name == self.OT_AXIAL:
            experiment = openturns.Axial(centers, levels)
        elif self.algo_name == self.OT_FACTORIAL:
            experiment = openturns.Factorial(centers, levels)

        samples = array(experiment.generate())
        samples = self._rescale_samples(samples)
        return samples

    def __generate_seq(self, n_samples):
        """
        Generate a DOE using LHS algo of openturns

        :param n_samples: number of samples in DOE
        :type  n_samples: integer
        :returns: samples
        :rtype: numpy array
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
        dim = self.problem.dimension
        seq = self.__sequence(dim).generate(n_samples)
        return array(seq)

    def create_composed_distributions(self):
        """Create a composed distribution from a list of distributions"""
        self.__comp_dist = openturns.ComposedDistribution(self.__distr_list)

    def get_composed_distributions(self):
        """Returns the composed distributions

        :returns: composed distributions
        :rtype: openturns.ComposedDistribution
        """
        return self.__comp_dist

    def __check_composed_distribution(self, distribution_name):
        """
        Checks the composed distribution

        :param distribution_name: name of the distribution
        """
        dim = self.problem.dimension
        if self.__comp_dist is None:
            n_distrib = len(self.__distr_list)
            if n_distrib == 0:
                LOGGER.info(
                    "Creating default composed distribution based on %s",
                    distribution_name,
                )
                self.create_distribution(distribution_name)
                self.__comp_dist = openturns.ComposedDistribution(
                    [self.__distr_list[0] for _ in range(dim)]
                )
            elif n_distrib == 1:
                # Only one distribution was defined ==> duplicating it in all
                # dimensions
                self.__comp_dist = openturns.ComposedDistribution(
                    [self.__distr_list[0] for _ in range(dim)]
                )
            elif n_distrib != dim:
                raise ValueError(
                    "Size mismatch between number"
                    " of distribution and problem: "
                    "{} vs. {}".format(str(dim), str(n_distrib))
                )
        elif self.__comp_dist.getDimension() != dim:
            raise ValueError(
                "Size mismatch between ComposedDistribution and "
                "problem: {} vs. {}".format(
                    str(dim), str(self.__comp_dist.getDimension())
                )
            )
        else:
            LOGGER.info(
                "Using composed distribution previously created: %s",
                str(self.__comp_dist.getDistributionCollection()),
            )

    def check_distribution_name(self, distribution_name):
        """Check that distribution is available

        :param distribution_name: name of the distribution
        :type distribution_name: string
        """
        if distribution_name not in self.DISTRIBUTION_LIST:
            raise ValueError(
                "Distribution '"
                + distribution_name
                + "' is not available. "
                + "Please switch to one of the followings: "
                + str(self.DISTRIBUTION_LIST)
            )

    def create_distribution(self, distribution_name="Uniform", **options):
        """Create a distribution for all design vectors
        and add it to the list of distributions

        :param distribution_name: name of the distribution
           (Default value = "Uniform")
        :type distribution_name: str
        :param options: optional parameters
        :type options: dict
        :param options: OT distributions options
        """
        self.check_distribution_name(distribution_name)
        options = self.__check_float(options, self.MEAN_KEYWORD, 0.5, u_b=1, l_b=0)
        options = self.__check_float(options, self.CENTER_KEYWORD, 0.5, u_b=1, l_b=0)
        options = self.__check_float(options, self.START_KEYWORD, 0.25, u_b=1, l_b=0)
        options = self.__check_float(options, self.END_KEYWORD, 0.75, u_b=1, l_b=0)

        if distribution_name == self.OT_UNIFORM:
            LOGGER.info("Creation of a uniform distribution")
            self.__distr_list.append(openturns.Uniform(0, 1))
        elif distribution_name == self.OT_TRIANGULAR:
            dist_center = options[self.CENTER_KEYWORD]
            LOGGER.info(
                "Creation of a triangular distribution" " with center: %s",
                str(dist_center),
            )
            self.__distr_list.append(openturns.Triangular(0.0, dist_center, 1.0))
        elif distribution_name == self.OT_TRAPEZOIDAL:
            lower_bound = options[self.START_KEYWORD]
            upper_bound = options[self.END_KEYWORD]
            LOGGER.info(
                "Creation of a trapezoidal distribution "
                "with lower/upper bounds: %s %s",
                str(lower_bound),
                str(upper_bound),
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
            LOGGER.info(
                "Creation of a %s distribution" ": mu (mean) %g and sigma (std) %g",
                distribution_name,
                mean,
                std,
            )
            beta = openturns.Beta()
            beta.setParameter(openturns.BetaMuSigma()([mean, std, 0.0, 1.0]))
            self.__distr_list.append(beta)
        elif distribution_name == self.OT_ARCSINE:
            LOGGER.info("Creation of a %s", str(distribution_name))
            arcs = openturns.Arcsine(0.0, 1.0)
            self.__distr_list.append(arcs)
        elif distribution_name == self.OT_TRUNCNORMAL:
            mean = options[self.MEAN_KEYWORD]
            options = self.__check_float(
                options, self.STD_KEYWORD, 0.5 / 3.75, u_b=1, l_b=0
            )
            std = options[self.STD_KEYWORD]
            LOGGER.info(
                "Creation of a %s distribution: mu (mean) %g and" "sigma (std) %g",
                distribution_name,
                mean,
                str,
            )
            nrmal = openturns.TruncatedNormal(mean, std, 0.0, 1.0)
            self.__distr_list.append(nrmal)
        elif distribution_name == self.OT_NORMAL:
            mean = options[self.MEAN_KEYWORD]
            options = self.__check_float(
                options, self.STD_KEYWORD, 0.5 / 3.75, u_b=1, l_b=0
            )
            std = options[self.STD_KEYWORD]
            LOGGER.info(
                "Creation of a %s distribution: mu (mean) %g and" "sigma (std) %g",
                distribution_name,
                mean,
                std,
            )
            self.__distr_list.append(openturns.Normal(mean, std))

    def display_distributions_list(self):
        """Display list of distributions use or that will be used for DOE
        design based on LHS or Monte-Carlo methods
        """
        LOGGER.info("List of distributions:")
        for distrib in self.__distr_list:
            LOGGER.info(distrib)

    def get_distributions_list(self):
        """Accessor for distributions list

        :returns: distribution list
        :rtype: list
        """
        return self.__distr_list

    def __generate_lhs(self, distribution_name="Uniform", **options):
        """
        Generate a DOE using LHS algo of openturns

        :param distribution_name: name of the distribution
        :type  distribution_name: string
        :param options: the options
        :returns: samples
        :rtype: numpy array
        """
        n_samples = options[self.N_SAMPLES]
        self.__check_composed_distribution(distribution_name=distribution_name)

        openturns.RandomGenerator.SetSeed(2)
        experiment = openturns.LHSExperiment(self.__comp_dist, n_samples)
        samples = array(experiment.generate())
        if self.algo_name == self.OT_LHSC:
            centered_samples = empty_like(samples)
            for i, sample in enumerate(samples):
                spl = 0.5 * ((sample[:] // (1.0 / n_samples) + 1.0) - 0.5) / n_samples
                centered_samples[i, :] = spl
            samples = centered_samples
        return samples

    def __generate_mc(self, distribution_name="Uniform", **options):
        """
        Generate a DOE using Monte-Carlo algo of openturns

        :param distribution_name: name of the distribution
        :type  distribution_name: string
        :param options: the options
        :returns: samples
        :rtype: numpy array
        """
        n_samples = options[self.N_SAMPLES]
        self.__check_composed_distribution(distribution_name=distribution_name)

        openturns.RandomGenerator.SetSeed(2)
        experiment = openturns.MonteCarloExperiment(self.__comp_dist, n_samples)
        return array(experiment.generate())

    def __generate_sobol(self, n_samples):
        """
        Generate a DOE using Sobol' sampling.
        :returns: samples
        :rtype: numpy array
        """
        openturns.RandomGenerator.SetSeed(2)
        self.__check_composed_distribution(distribution_name="Uniform")
        dim = self.problem.dimension
        n_samples = int(n_samples / (dim + 2))
        experiment = openturns.SobolIndicesExperiment(self.__comp_dist, n_samples)
        data = array(experiment.generate())
        return data

    def __generate_fullfact(self, n_samples):
        """
        Generate a DOE using Monte-Carlo algo of openturns

        :param n_samples: number of samples in DOE
        :type  n_samples: integer
        :returns: samples
        :rtype: numpy array
        """
        self._display_fullfact_warning(n_samples)
        dim = self.problem.dimension
        level = int(n_samples ** (1.0 / dim) - 2)
        if level < 1:
            level = 0
        levels = [level] * dim
        experiment = openturns.Box(levels)
        return array(experiment.generate())

    def __generate_random(self, n_samples):
        """
        Generate a DOE using random algo of openturns

        :param n_samples: number of samples in DOE
        :type  n_samples: integer
        :returns: samples
        :rtype: numpy array
        """
        openturns.RandomGenerator.SetSeed(2)
        samples_list = []
        dim = self.problem.dimension
        for _ in range(n_samples):
            samples_list.append(openturns.RandomGenerator.Generate(dim))
        return array(samples_list)

    @staticmethod
    def plot_distribution(distribution, show=False):
        """Plot the density PDF & the CDF (cumulative) of a given distribution

        :param distribution: the distribution to plot
        :type distribution: openturns.Distribution
        :param show: show plot (Default value = False)
        :type show: bool
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
        mean = array(distribution.getMean())[0]
        if show:
            plt.show()
        plt.close()
