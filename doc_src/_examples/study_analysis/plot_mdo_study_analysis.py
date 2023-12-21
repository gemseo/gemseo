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
Generate N2 and XDSM diagrams from an Excel description of the MDO problem
==========================================================================
"""

from __future__ import annotations

from gemseo import configure_logger
from gemseo.utils.study_analyses.mdo_study_analysis import MDOStudyAnalysis

configure_logger()


# %%
# Describe your MDO problem in an Excel file
# ------------------------------------------
#
# .. image:: /_images/study_analysis_example/mdo_study.png
#

# %%
# Visualize this study
# --------------------
study = MDOStudyAnalysis("mdo_study.xlsx")

# %%
# Generate the N2 chart
# ^^^^^^^^^^^^^^^^^^^^^
study.generate_n2(save=False, show=True)

# %%
# Generate the XDSM
# ^^^^^^^^^^^^^^^^^
study.generate_xdsm(".")
# %%
# .. image:: /_images/study_analysis_example/xdsm.png

# %%
# Visualize this study from command line
# --------------------------------------
#
# We can create the same figures using command line inputs:
#
# .. code::
#
#    gemseo-study mdo_study.xlsx -o outputs -h 5 -w 5 -x -l
#
# where ``gemseo-study`` is an executable provided by |g|
# and the Excel file path ``mdo_study.xlsx`` is the specification of the MDO study.
# Here, we set some options of ``gemseo-study``:
#
# - ``-o outputs`` is the output directory,
# - ``-h 5`` is the height of the N2 chart in inches,
# - ``-w 5`` is the width of the N2 chart in inches,
# - ``-x`` is an option to create of the XDSM
#   (compatible only with the study type 'mdo'),
# - ``-l`` is an option to create a PDF file with the creation of the XDSM.
