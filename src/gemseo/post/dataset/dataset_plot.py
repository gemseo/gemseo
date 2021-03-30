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
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Abstract dataset plot
=====================

The :mod:`~gemseo.post.dataset.dataset_plot` module implements the abstract
:class:`.DatasetPlot` class
whose purpose is to build a graphical representation
of a :class:`.Dataset` and to display it on screen or save it to a file.
This abstract class has to be overloaded by concrete ones implementing
at least method :meth:`!DatasetPlot._run`.
"""
from __future__ import absolute_import, division, unicode_literals

from numbers import Number

import matplotlib
import pylab
from future import standard_library
from six import string_types

standard_library.install_aliases()


class DatasetPlot(object):
    """ Abstract plot class for dataset. """

    def __init__(self, dataset):
        """Constructor.

        :param Dataset dataset: dataset
        """
        if dataset.is_empty():
            raise ValueError("Dataset is empty.")
        self.dataset = dataset
        self.output_files = []

    def execute(
        self, save=True, show=False, file_path=None, extension="pdf", **plot_options
    ):
        """
        Executes the post processing

        :param show: if True, displays the plot windows. Default: False.
        :type show: bool
        :param save: if True, exports plot to pdf. Default: False.
        :type save: bool
        :param file_path: the base paths of the files to export. Default: None.
        :type file_path: str
        :param extension: file extension. Default: 'pdf'.
        :type extension: str
        :param plot_options: options passed to the _plot() method.
        """
        self._run(save, show, file_path, extension, **plot_options)

    def _run(self, save, show, file_path, extension, **plot_options):
        """Define run of the post processing, calling _plot() and saving
        or displaying figure.

        :param plot_options: options passed to the _plot() method.
        :param show: if True, displays the plot windows. Default: False.
        :type show: bool
        :param save: if True, exports plot to pdf. Default: False.
        :type save: bool
        :param file_path: the base paths of the files to export. Default: None.
        :type file_path: str
        :param extension: file extension. Default: 'pdf'.
        :type extension: str
        :param plot_options: options passed to the _plot() method.
        """
        fig = self._plot(**plot_options)
        if isinstance(fig, list):
            for index, subfig in enumerate(fig):
                subfig.tight_layout()
                filename = self.__class__.__name__ + str(index)
                self._save_and_show(
                    subfig,
                    filename=filename,
                    save=save,
                    show=show,
                    file_path=file_path,
                    extension=extension,
                )
        else:
            fig.tight_layout()
            self._save_and_show(
                fig,
                filename=self.__class__.__name__,
                save=save,
                show=show,
                file_path=file_path,
                extension=extension,
            )

    def _plot(self, **options):
        """Define plot of the post processing,
        to be implemented in subclasses

        :param options: plotting options according to associated json file
        """
        raise NotImplementedError

    def _save_and_show(
        self, fig, filename, save=True, show=False, file_path=None, extension="pdf"
    ):
        """
        Saves figures and or shows it depending on options

        :param fig: matplotlib figure
        :param str filename: name of the file to save the plot
        :param bool save: save the figure (default: True)
        :param bool show: show the figure (default: False)
        :param str file_path: file path to save the plot (default: None)
        :param str extension: file extension (default: 'pdf')
        """
        matplotlib.rcParams.update({"font.size": 10})
        if save:
            file_path = file_path or self.dataset.name
            fpath = file_path + "_" + filename
            fpath += "." + extension
            fig.savefig(fpath, bbox_inches="tight")
            if fpath not in self.output_files:
                self.output_files.append(fpath)
        if show:
            try:
                pylab.plt.show(fig)
            except TypeError:
                pylab.plt.show()

    def _get_varnames(self, df_columns):
        """Get varnames from columns of a pandas dataframe.

        :param list(tuple) dataframe: pandas dataframe
        """
        new_columns = []
        for column in df_columns:
            if self.dataset.sizes[column[1]] == 1:
                new_columns.append(column[1])
            else:
                new_columns.append(column[1] + "(" + str(column[2]) + ")")
        return new_columns

    def _get_label(self, varname):
        """Returns string label associated to a variable name,
        as well as refactored variable name.

        :param varname: variable name, either a string
            or a (str, int) tuple.
        """
        if isinstance(varname, string_types):
            label = varname
            varname = (self.dataset.get_group(varname), varname, "0")
        elif hasattr(varname, "__len__") and len(varname) == 3:
            is_string = isinstance(varname[0], string_types)
            is_string = is_string and isinstance(varname[1], string_types)
            is_number = isinstance(varname[2], Number)
            if is_string and is_number:
                label = varname[1] + "(" + str(varname[2]) + ")"
                varname[2] = str(varname[2])
                varname = tuple(varname)
            else:
                raise TypeError(
                    "varname must be either a string or a tuple"
                    " whose first component is a string and second"
                    " one is an integer."
                )
        else:
            raise TypeError(
                "varname must be either a string or a tuple"
                " whose first component is a string and second"
                " one is an integer."
            )
        return label, varname
