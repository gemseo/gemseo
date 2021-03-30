# -*- coding: utf-8 -*-
# https://github.com/noamraph/tqdm
# the MIT License (MIT)
# Copyright (c) 2013 noamraph
# permission is hereby granted, free of charge,
# to any person obtaining a copy of
# this software and associated documentation
# files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# the, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of
# the Software, and to permit persons to whom the
# Software is furnished to do so,
# object to the following conditions:
# the above copyright notice and this permission
# notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
A progress bar implementation
*****************************
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from builtins import chr, int
from multiprocessing import Value
from numbers import Number
from numpy.linalg import norm
from timeit import default_timer as timer

from future import standard_library

from gemseo.api import configure_logger


standard_library.install_aliases()


from gemseo import LOGGER

N_BARS = 10


class StatusPrinter(object):
    """
    Prints the status, recors the last printed status
    This class is multiprocessing-safe : the counters values are shared among all processes
    """

    def __init__(self):
        self.last_printed_len = 0

    def print_status(self, status):
        """
        Prints the status

        :param status: the status
        """
        LOGGER.info(status + " " * max(self.last_printed_len - len(status), 0))
        self.last_printed_len = len(status)


class Tqdm(object):
    """TQDM iterator progress bar"""

    def __init__(self, max_iter, desc="", mininterval=0.5, miniters=1):
        """
        Prints a progress meter and updates it every time
        the next() ethod is   called.
        'desc' can contain a short string,
        describing the progress, that is added
        in the beginning of the line.
        If less than mininterval seconds or  miniters iterations have passed
        since the last progress meter update, it is not updated again.
        """
        self.max_iter = max_iter
        self.desc = desc
        self.mininterval = mininterval
        self.miniters = miniters
        self.prefix = self.desc + ": " if self.desc else ""
        self.stat_printer = StatusPrinter()
        self._last_print_t = Value("d", -1.0)
        self._start_t = Value("d", -1.0)
        self._last_print_n = Value("i", -1)
        # The counter is shared among all processes when run in parallel
        self._current_n = Value("i", 0)
        self._closed = Value("b", False)

    def _print_current(self, cur_t, obj_val=None):
        """
        Prints the current status

        :param cur_t: current time (Default value = None)
        :param obj_val: curent objective value (Default value = None)
        """
        formatted = format_meter(
            self._current_n.value, self.max_iter, cur_t - self._start_t.value, obj_val
        )
        self.stat_printer.print_status(self.prefix + formatted)

    def close(self, obj_val=None):
        """
        Closes the progress bar
        """
        with self._current_n.get_lock() and self._last_print_n.get_lock() and self._closed.get_lock():
            if self._last_print_n.value != self._current_n.value:
                if self._current_n.value > self.max_iter:
                    self._current_n.value = self.max_iter
                self._print_current(timer(), obj_val)
                self._last_print_n.value = self._current_n.value
            self._closed.value = True

    def next(self, obj_val=None):
        """
        Iterates the progress bar

        :param obj_val: objective value (Default value = None)
        """

        if self._closed.value:
            raise ValueError("Progress bar is closed while calling next()")
        # Now the object was created and processed, so we can print the
        # meter.
        cur_t = timer()
        with self._current_n.get_lock() and self._last_print_n.get_lock() and self._last_print_t.get_lock() and self._start_t.get_lock():
            if self._current_n.value == self.max_iter:
                raise StopIteration
            if self._current_n.value >= self.miniters + self._last_print_n.value:
                # We check the counter first, to reduce the overhead of

                if cur_t >= self.mininterval + self._last_print_t.value:
                    if self._last_print_t.value < 0.0:
                        # First call, we need to initialize the variables
                        self._last_print_t.value = cur_t
                        self._start_t.value = cur_t
                    self._print_current(cur_t, obj_val)
                    self._last_print_n.value = self._current_n.value
                    self._last_print_t.value = cur_t

            self._current_n.value += 1


def format_interval(time_inter):
    """
    Formats a time interval

    :param time_inter: the time interval to format
    """
    mins, sec = divmod(int(time_inter), 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return "%d:%02d:%02d" % (hours, mins, sec)
    else:
        return "%02d:%02d" % (mins, sec)


def format_meter(n_iter, max_iter, elapsed, obj_val=None):
    """
    Format a message

    :param n_iter: number of finished iterations
    :param obj_val:  objective value (Default value = None)
    :param max_iter: total number of iterations, or None
    :param elapsed: number of seconds passed since start
    """
    if n_iter > max_iter:
        max_iter = None

    elapsed_str = format_interval(elapsed)
    if not elapsed:
        rate_str, unit = "?", "iters/sec"
    else:
        rate, unit = get_printable_rate(n_iter, elapsed)
        rate_str = "%5.2f" % rate
    if max_iter:
        frac = float(n_iter) / max_iter

        bar_length, frac_bar_length = divmod(int(frac * N_BARS * 8), 8)
        bar_l = chr(0x2588) * bar_length
        frac_bar = chr(0x2590 - frac_bar_length) if frac_bar_length else " "

        percentage = "%3d%%" % (frac * 100)
        if bar_length < N_BARS:
            full_bar = bar_l + frac_bar + " " * max(N_BARS - bar_length - 1, 0)
        else:
            full_bar = bar_l + " " * max(N_BARS - bar_length, 0)
        left_str = (
            format_interval(elapsed / n_iter * (max_iter - n_iter)) if n_iter else "?"
        )

        if obj_val is None:
            format_str = "|%s| %d/%d %s [elapsed: %s left: %s, %s " + unit + "]"
            return format_str % (
                full_bar,
                n_iter,
                max_iter,
                percentage,
                elapsed_str,
                left_str,
                rate_str,
            )
        else:
            if obj_val is None or isinstance(obj_val, Number) or obj_val.size == 1:
                tmp = obj_val
            else:
                tmp = norm(obj_val)
            obj_s = "%5.2f" % tmp
            format_str = "|%s| %d/%d %s "
            format_str += "[elapsed: %s left: %s, %s " + unit + " obj: %s ]"
            return format_str % (
                full_bar,
                n_iter,
                max_iter,
                percentage,
                elapsed_str,
                left_str,
                rate_str,
                obj_s,
            )
    else:
        return "%d [elapsed: %s, %s " + unit + "]" % (n_iter, elapsed_str, rate_str)


def get_printable_rate(n_iter, elapsed):
    rps = float(n_iter) / elapsed
    if rps >= 0:
        rate = rps
        unit = "iters/sec"

    rpm = rps * 60.0
    if rpm < 60.0:
        rate = rpm
        unit = "iters/min"

    rph = rpm * 60.0
    if rph < 60:
        rate = rph
        unit = "iters/hour"

    rpd = rph * 24.0
    if rpd < 24:
        rate = rpd
        unit = "iters/day"

    return rate, unit
