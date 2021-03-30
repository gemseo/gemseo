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
#    INITIAL AUTHORS - initial API and implementation and/or
#                      initial documentation
#        :author:  Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Parse source code to extract information
****************************************
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import re
from builtins import range, str
from inspect import getargspec, getdoc

from future import standard_library

standard_library.install_aliases()


class SourceParsing(object):
    """
    Parse source code to extract information
    """

    @staticmethod
    def get_options_doc(method):
        """
        Get the documentation of a method

        :param method: the method to retreive the doc from
        :returns: the dictionary of options meaning
        """
        doc = getdoc(method)
        if doc is None:
            raise ValueError("Empty doc for " + str(method))
        pattern = ":param ([\*\w]+): (.*?)"  # pylint: disable=W1401
        pattern += "(?:(?=:param)|(?=:return)|\Z)"  # pylint: disable=W1401
        param_re = re.compile(pattern, re.S)
        doc_list = param_re.findall(doc)
        return {txt[0]: txt[1].replace(" " * 4, "") for txt in doc_list}

    @staticmethod
    def get_default_options_values(klass):
        """
        Get the options default values for the given class
        Only addresses kwargs

        :param name : name of the class
        :returns: the dict option name: option default value
        """
        args, _, _, defaults = getargspec(klass.__init__)
        if "self" in args:
            args.remove("self")
        n_def = len(defaults)

        args_dict = {args[-n_def:][i]: defaults[i] for i in range(n_def)}

        return args_dict
