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
PyDOE algorithms wrapper
************************
"""
from __future__ import division, unicode_literals

import logging

from numpy import array, ndarray
from numpy.random import RandomState
from numpy.random import seed as set_seed

from gemseo.algos.doe.doe_lib import DOELibrary
from gemseo.utils.py23_compat import PY3

if PY3:
    import pyDOE2 as pyDOE
else:
    import pyDOE


LOGGER = logging.getLogger(__name__)


class PyDOE(DOELibrary):
    """PyDOE optimization library interface See DOELibrary."""

    # Available designs
    PYDOE_DOC = "https://pythonhosted.org/pyDOE/"
    PYDOE_LHS = "lhs"
    PYDOE_LHS_DESC = "Latin Hypercube Sampling implemented in pyDOE"
    PYDOE_LHS_WEB = PYDOE_DOC + "randomized.html#latin-hypercube"
    PYDOE_2LEVELFACT = "ff2n"
    PYDOE_2LEVELFACT_DESC = "2-Level Full-Factorial implemented in pyDOE"
    PYDOE_2LEVELFACT_WEB = PYDOE_DOC + "factorial.html#level-full-factorial"
    PYDOE_FULLFACT = "fullfact"
    PYDOE_FULLFACT_DESC = "Full-Factorial implemented in pyDOE"
    PYDOE_FULLFACT_WEB = PYDOE_DOC + "factorial.html#general-full-factorial"
    PYDOE_PBDESIGN = "pbdesign"
    PYDOE_PBDESIGN_DESC = "Plackett-Burman design implemented in pyDOE"
    PYDOE_PBDESIGN_WEB = PYDOE_DOC + "factorial.html#plackett-burman"
    PYDOE_BBDESIGN = "bbdesign"
    PYDOE_BBDESIGN_DESC = "Box-Behnken design implemented in pyDOE"
    PYDOE_BBDESIGN_WEB = PYDOE_DOC + "rsm.html#box-behnken"
    PYDOE_CCDESIGN = "ccdesign"
    PYDOE_CCDESIGN_DESC = "Central Composite implemented in pyDOE"
    PYDOE_CCDESIGN_WEB = PYDOE_DOC + "rsm.html#central-composite"
    ALGO_LIST = [
        PYDOE_FULLFACT,
        PYDOE_2LEVELFACT,
        PYDOE_PBDESIGN,
        PYDOE_BBDESIGN,
        PYDOE_CCDESIGN,
        PYDOE_LHS,
    ]
    DESC_LIST = [
        PYDOE_FULLFACT_DESC,
        PYDOE_2LEVELFACT_DESC,
        PYDOE_PBDESIGN_DESC,
        PYDOE_BBDESIGN_DESC,
        PYDOE_CCDESIGN_DESC,
        PYDOE_LHS_DESC,
    ]
    WEB_LIST = [
        PYDOE_FULLFACT_WEB,
        PYDOE_2LEVELFACT_WEB,
        PYDOE_PBDESIGN_WEB,
        PYDOE_BBDESIGN_WEB,
        PYDOE_CCDESIGN_WEB,
        PYDOE_LHS_WEB,
    ]
    CRITERION_KEYWORD = "criterion"
    ITERATION_KEYWORD = "iterations"
    ALPHA_KEYWORD = "alpha"
    FACE_KEYWORD = "face"
    CENTER_BB_KEYWORD = "center_bb"
    CENTER_CC_KEYWORD = "center_cc"

    def __init__(self):
        """Constructor."""
        super(PyDOE, self).__init__()
        for idx, algo in enumerate(self.ALGO_LIST):
            self.lib_dict[algo] = {
                DOELibrary.LIB: self.__class__.__name__,
                DOELibrary.INTERNAL_NAME: algo,
                DOELibrary.DESCRIPTION: self.DESC_LIST[idx],
                DOELibrary.WEBSITE: self.WEB_LIST[idx],
            }

        self.lib_dict["bbdesign"][DOELibrary.MIN_DIMS] = 3
        self.lib_dict["ccdesign"][DOELibrary.MIN_DIMS] = 2

    def _get_options(
        self,
        alpha="orthogonal",
        face="faced",
        criterion=None,
        iterations=5,
        eval_jac=False,
        center_bb=None,
        center_cc=None,
        n_samples=None,
        levels=None,
        n_processes=1,
        wait_time_between_samples=0.0,
        seed=1,
        max_time=0,
        **kwargs
    ):  # pylint: disable=W0221
        """Sets the options.

        :param alpha: effect the variance, either "orthogonal" or "rotatable"
        :type alpha: str
        :param face: The relation between the start points and the corner
            (factorial) points, either "circumscribed", "inscribed" or "faced"
        :type face: str
        :param criterion: Default value = None)
        :type criterion:
        :param iterations: Default value = 5)
        :type iterations:
        :param eval_jac: evaluate jacobian
        :type eval_jac: bool
        :param center_bb: number of center points for Box-Behnken design
        :type center_bb: int
        :param center_cc: 2-tuple of center points for the central
            composite design
        :type center_cc: tuple
        :param n_samples: number of samples
        :type n_samples: int
        :param levels: level in each direction for the full-factorial design
        :type levels: array
        :param n_processes: number of processes
        :type n_processes: int
        :param wait_time_between_samples: waiting time between two samples
        :type wait_time_between_samples: float
        :param seed: seed value.
        :type seed: int
        :param max_time: maximum runtime in seconds,
            disabled if 0 (Default value = 0)
        :type max_time: float
        :param kwargs: additional arguments
        """
        if center_cc is None:
            center_cc = [4, 4]
        wtbs = wait_time_between_samples
        popts = self._process_options(
            alpha=alpha,
            face=face,
            criterion=criterion,
            iterations=iterations,
            center_cc=center_cc,
            center_bb=center_bb,
            eval_jac=eval_jac,
            n_samples=n_samples,
            n_processes=n_processes,
            levels=levels,
            wait_time_between_samples=wtbs,
            seed=seed,
            max_time=max_time,
            **kwargs
        )

        return popts

    @staticmethod
    def __translate(result):
        """Translate DOE design variables in [0,1]

        :param result: the samples
        """
        return (result + 1.0) * 0.5

    def _generate_samples(self, **options):
        """Generates the list of x samples.

        :param options: the options dict for the algorithm,
            see associated JSON file
        """
        self.seed += 1
        if self.algo_name == self.PYDOE_LHS:
            seed = options.get(self.SEED, self.seed)
            lhs_kwargs = {
                "samples": options["n_samples"],
                "criterion": options.get(self.CRITERION_KEYWORD),
                "iterations": options.get(self.ITERATION_KEYWORD),
            }
            if PY3:
                lhs_kwargs["random_state"] = RandomState(seed)
            else:
                set_seed(seed)
            return pyDOE.lhs(options[self.DIMENSION], **lhs_kwargs)

        if self.algo_name == self.PYDOE_CCDESIGN:
            return self.__translate(
                pyDOE.ccdesign(
                    options[self.DIMENSION],
                    center=options[self.CENTER_CC_KEYWORD],
                    alpha=options[self.ALPHA_KEYWORD],
                    face=options[self.FACE_KEYWORD],
                )
            )

        if self.algo_name == self.PYDOE_BBDESIGN:
            # Initialy designed for quadratic model fitting
            # center point is can be run several times to allow for a more
            # uniform estimate of the prediction variance over the
            # entire design space. Default value of center depends on dv_size
            return self.__translate(
                pyDOE.bbdesign(
                    options[self.DIMENSION], center=options.get(self.CENTER_BB_KEYWORD)
                )
            )

        if self.algo_name == self.PYDOE_FULLFACT:
            return self._generate_fullfact(
                options[self.DIMENSION],
                levels=options.get(self.LEVEL_KEYWORD),
                n_samples=options.get(self.N_SAMPLES),
            )

        if self.algo_name == self.PYDOE_2LEVELFACT:
            return self.__translate(pyDOE.ff2n(options[self.DIMENSION]))

        if self.algo_name == self.PYDOE_PBDESIGN:
            return self.__translate(pyDOE.pbdesign(options[self.DIMENSION]))

    def _generate_fullfact_from_levels(
        self, levels  # Iterable[int]
    ):  # type: (...) -> ndarray
        doe = pyDOE.fullfact(levels)

        # Because pyDOE return the DOE where the values of levels are integers from 0 to
        # the maximum level number,
        # we need to divide by levels - 1.
        # To not divide by zero,
        # we first find the null denominators,
        # we replace them by one,
        # then we change the final values of the DOE by 0.5.
        divide_factor = array(levels) - 1
        null_indices = divide_factor == 0
        divide_factor[null_indices] = 1
        doe /= divide_factor
        doe[:, null_indices] = 0.5
        return doe

    @staticmethod
    def is_algorithm_suited(algo_dict, problem):
        """Checks if the algorithm is suited to the problem according to its algo dict.

        :param algo_dict: the algorithm characteristics
        :param problem: the opt_problem to be solved
        """
        if DOELibrary.MIN_DIMS in algo_dict:
            if problem.dimension < algo_dict[DOELibrary.MIN_DIMS]:
                return False
        return True
