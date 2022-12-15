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
Generate an N2 and XDSM from an Excel description of the MDO problem
====================================================================
"""
from __future__ import annotations

from os import mkdir
from os.path import exists
from os.path import join

from gemseo.api import configure_logger
from gemseo.utils.study_analysis import StudyAnalysis

configure_logger()


#############################################################################
# Describe your MDO problem in an Excel file
# ------------------------------------------
#
# .. image:: /_images/study_analysis_example/disciplines_spec.png
#

#############################################################################
# Visualize this study
# --------------------
study = StudyAnalysis("disciplines_spec.xlsx")
if not exists("outputs"):
    mkdir("outputs")

#############################################################################
# Generate N2 chrt
# ^^^^^^^^^^^^^^^^^
study.generate_n2(file_path=join("outputs", "n2.png"), save=False, show=True)

#############################################################################
# Generate XDSM
# ^^^^^^^^^^^^^
study.generate_xdsm("outputs")
#############################################################################
# .. image:: /_images/study_analysis_example/xdsm.png

#############################################################################
# Visualize this study from command line
# --------------------------------------
#
# We can create the same figures using command line inputs:
#
# .. code::
#
#    gemseo-study disciplines_spec.xlsx -o outputs -s '(5,5)' -x -l
#
# where:
#
# - :code:`gemseo-study` is an executable provided by |g|,
# - :code:`disciplines_spec.xlsx` is the Excel file path,
# - :code:`-o outputs` is the output directory,
# - :code:`-s '(5,5)'` is the size of the N2 chart,
# - :code:`-x` is an option to create of the XDSM,
# - :code:`-l` is an option to create a PDF file with the creation of the XDSM.
