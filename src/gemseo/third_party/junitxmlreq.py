# -*- coding: utf-8 -*-
# Based on unittest2/plugins/junitxml.py,
# which is itself based on the junitxml plugin from py.test distributed
# under the MIT license:
# The MIT License (MIT)
#
# Copyright (c) 2004-2016 Holger Krekel and others
#
# Permission is hereby granted, free of charge,
# to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons
# to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice
# shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Output test reports in junit-xml format.
# This plugin implements:func:`startTest`,:func:`testOutcome` and
# :func:`stopTestRun` to compile and then output a test report in
# junit-xml format. By default, the report is written to a file called
# ``nose2-junit.xml`` in the current working directory.
# You can configure the output filename by setting ``path``
# in a ``[junit-xml]``
# section in a config file.  Unicode characters which are invalid in XML 1.0
# are replaced with the ``U+FFFD`` replacement character.
# In the case that your
# software throws an error with an invalid byte string.
# By default, the ranges of discouraged characters are replaced as well.
#  This can be changed by setting the ``keep_restricted``
#   configuration variable to ``True``.

"""
Output test reports in junit-xml format
***************************************
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import functools

from future import standard_library

standard_library.install_aliases()


from gemseo import LOGGER
__unittest = True  # pylint: disable=C0103


def link_to(*dec_args):
    """Prints the requirements covered by the test.

    :param dec_args: decorator arguments
    """
    def decorator(function):
        """Decorates the function to print requirement

        :param function: returns: the decorated function
        :returns: the decorated function
        """
        @functools.wraps(function)
        def call_and_log_req(*args, **kwargs):
            """Logs the covered requirement
            then calls the function

            :param args: function args
            :param kwargs: function kwargs
            :param args:
            :param kwargs:
            :returns: function output
            """
            LOGGER.info('[Covers requirements: %s]', ', '.join(dec_args))
            return function(*args, **kwargs)
        # Add attribute to returned function
        call_and_log_req.requirements = dec_args
        return call_and_log_req
    return decorator
