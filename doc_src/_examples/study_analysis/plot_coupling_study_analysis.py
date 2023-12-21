# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This work is licensed under a BSD 0-Clause License.
#
# Permission to use, copy, modify, and/or distribute this software
# for any purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL
# WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL
# THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT,
# OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING
# FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
# NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
# WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
# Contributors:
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Generate an N2 from an Excel description of the coupling problem
================================================================
"""

from __future__ import annotations

from gemseo import configure_logger
from gemseo.utils.study_analyses.coupling_study_analysis import CouplingStudyAnalysis

configure_logger()


# %%
# Describe your coupling problem in an Excel file
# -----------------------------------------------
#
# .. image:: /_images/study_analysis_example/coupling_study.png
#

# %%
# Visualize this study
# --------------------
study = CouplingStudyAnalysis("coupling_study.xlsx")
study.generate_n2(save=False, show=True)

# %%
# Visualize this study from the command line
# ------------------------------------------
#
# We can create the same figures using command line inputs:
#
# .. code::
#
#    gemseo-study coupling_study.xlsx -t coupling -o outputs --height 5 --width 5
#
# where ``gemseo-study`` is an executable provided by |g|
# and the Excel file path ``coupling_study.xlsx`` is the specification
# of the coupling study.
# Here, we set some options of ``gemseo-study``:
#
# - ``-t coupling`` is the type of study (default: ``mdo``),
# - ``-o outputs`` is the output directory,
# - ``--height 5`` is the height of the N2 chart in inches,
# - ``--width 5`` is the width of the N2 chart in inches.
