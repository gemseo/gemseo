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
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
A factory to execute post processings from their class name
***********************************************************
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from future import standard_library

from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.core.factory import Factory
from gemseo.post.opt_post_processor import OptPostProcessor
from gemseo.utils.py23_compat import string_types

standard_library.install_aliases()


from gemseo import LOGGER


class PostFactory(object):
    """Post processing factory to run optimization post processings
    Lists available post processings on the current configuration,
    executes them on demand.

    Works both from memory, from a ran optimization problem,
    and from disk, from a serialized optimization problem.
    """

    def __init__(self):
        """
        Initializes the factory: scans the directories to search for
        subclasses of OptPostProcessor.
        Searches in "GEMSEO_PATH" and gemseo.post
        """
        self.factory = Factory(OptPostProcessor, ("gemseo.post",))
        self.executed_post = []

    @property
    def posts(self):
        """Lists the available post processings

        :returns: the list of methods
        """
        return self.factory.classes

    def is_available(self, name):
        """Checks the availability of a post processing name

        :param name: the name of the post processing
        :returns: True if the post step is installed
        """
        return self.factory.is_available(name)

    def create(self, opt_problem, post_name):
        """Factory method to create a post processing subclass from post_name
        which is a class name

        :param opt_problem: the optimization problem on which to run
            the post processing
        :param post_name: the post processing name

        """
        return self.factory.create(post_name, opt_problem=opt_problem)

    def execute(self, opt_problem, post_name, **options):
        """Finds the appropriate library and executes
        the post processing on the problem

        :param opt_problem: the optimization problem on which to run
            the post procesing
        :param post_name: the post processing name
        """
        if isinstance(opt_problem, string_types):
            opt_problem = OptimizationProblem.import_hdf(opt_problem)
        post = self.create(opt_problem, post_name)
        post.execute(**options)
        self.executed_post.append(post)
        return post

    def list_generated_plots(self):
        """Lists the generated plot files"""
        plots = []
        for post in self.executed_post:
            plots.extend(post.output_files)
        return set(plots)
