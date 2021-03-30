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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import absolute_import, division, unicode_literals

import logging
from os.path import exists, join

import pytest
from future import standard_library

from gemseo import SOFTWARE_NAME
from gemseo.problems.scalable.scalable_tm.core import TMParamSS, TMParamSSPost

from ....utils import test_study_analysis

pytestmark = test_study_analysis.pytestmark

standard_library.install_aliases()

LOGGER = logging.getLogger(SOFTWARE_NAME)


@pytest.mark.skipif(**test_study_analysis.has_no_pdflatex)
def test_tm_study_param(tmpdir):
    out_directory = str(tmpdir)
    studies_p = [
        (2, 1, 1, True),
        (2, 1, 1, False),
        ([1, 2], 1, 1, True),
        (2, [1, 2], 1, True),
        (2, 1, [1, 2], True),
        (2, 1, [1, 2], False),
    ]

    max_iter = 50
    for n_shared, n_local, n_coupling, fully_coupled in studies_p:
        tmpss = TMParamSS(
            2,
            n_shared,
            n_local,
            n_coupling,
            fully_coupled=fully_coupled,
            directory=out_directory,
        )
        assert len(str(tmpss)) > 10
        tmpss.run_formulation(
            "IDF",
            max_iter=max_iter,
            post_coupling=False,
            post_optim=False,
            post_coeff=False,
        )
        file_path = join(out_directory, "study.pckl")
        tmpss.save(file_path)
        assert exists(file_path)

        post = TMParamSSPost(file_path)
        post_path = join(out_directory, "plot.pdf")
        post.plot(save=True, show=False, file_path=post_path)

        for study in tmpss.studies:
            assert len(str(study)) > 10
            assert len(str(study.problem)) > 10
            post_path = join(out_directory, "exec_time.pdf")
            # study.plot_exec_time(save=True, show=False, file_path=post_path)
            # assert exists(post_path)

    tmpss = TMParamSS(2, 2, 1, [1, 2])
    assert len(str(tmpss)) > 10
    tmpss.run_formulation(
        "MDF", max_iter=max_iter, post_coupling=True, post_optim=True, post_coeff=True
    )
    with pytest.raises(ValueError):
        TMParamSS(2, [1, 2], [1, 2], 1)
