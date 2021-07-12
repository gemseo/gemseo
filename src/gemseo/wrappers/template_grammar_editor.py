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
"""GUI for edition of templates for inputs and outputs files To be used by
:class:`.DiscFromExe`

Run this file with no argument to open the GUI
"""
from __future__ import division, unicode_literals

import logging
import sys
from os.path import dirname, exists, join

LOGGER = logging.getLogger(__name__)

try:
    from PySide2.QtCore import QRegExp
    from PySide2.QtGui import QColor, QIcon, QTextCursor
    from PySide2.QtWidgets import (
        QAction,
        QApplication,
        QFileDialog,
        QInputDialog,
        QMainWindow,
        QTextEdit,
    )
except ImportError:
    # Both libraries are fully interchangeable so we are not contaminated by
    # the GPL license here
    LOGGER.warning("PySide2 cannot be imported.")
    from PyQt5.QtCore import QRegExp
    from PyQt5.QtGui import QColor, QIcon, QTextCursor
    from PyQt5.QtWidgets import (
        QAction,
        QApplication,
        QFileDialog,
        QInputDialog,
        QMainWindow,
        QTextEdit,
    )

    LOGGER.warning(
        "Your Python environment uses PyQt5"
        ": it is distributed under the GNU GPL v3.0 license "
        "unless you have acquired a commercial license for it."
    )


class QtTemplateEditor(QMainWindow):
    """GUI template generator.

    GUI to generate templates for input and output files edition
    Input text file data is replaced by a mark that specifies where to read the
    data.
    idem for outputs.
    Works with any text file format.

    To open the GUI, run this python file

    Example, for an input json file :
    {
    "a": 1.01515112125,
    "b": 2.00151511213,
    "c": 3.00151511213
    }

    Generates a template :
    {
    "a": GEMSEO_INPUT{a::1.0},
    "b": GEMSEO_INPUT{b::2.0},
    "c": GEMSEO_INPUT{c::3.0}
    }

    Same for outputs.
    """

    def __init__(self, in_sep="GEMSEO_INPUT", out_sep="GEMSEO_OUTPUT"):
        """Constructor :

        :param in_sep: separator name for the input tag, default GEMSEO_INPUT
        :param out_sep: separator name for the output tag, default GEMSEO_OUTPUT
        """

        self.in_sep = in_sep
        self.out_sep = out_sep

        QMainWindow.__init__(self, None)
        self.q_text_e = QTextEdit(self)
        self._setup_toolbars()

        self.q_text_e.setTabStopWidth(12)
        self.setCentralWidget(self.q_text_e)

        self.setGeometry(100, 100, 600, 800)

    def _setup_toolbars(self):
        """Setup the toolbars, the icons and shortcuts."""
        self.toolbar = self.addToolBar("Actions")

        self.add_action("Open", "Open existing document", "Ctrl+Shift+O", self.open_doc)
        self.add_action("Save", "Save document", "Ctrl+S", self.save_doc)
        self.toolbar.addSeparator()

        self.add_action("Input", "Make input", "Ctrl+I", self.make_input)
        self.add_action("Output", "Make output", "Ctrl+O", self.make_output)

        self.toolbar.addSeparator()

        self.add_action("Copy", "Copy text to clipboard", "Ctrl+C", self.q_text_e.copy)
        self.add_action(
            "Paste", "Paste text from clipboard", "Ctrl+V", self.q_text_e.paste
        )
        self.add_action("Cut", "Cut text from clipboard", "Ctrl+X", self.q_text_e.cut)

        self.toolbar.addSeparator()

        self.add_action("Undo", "Undo last action", "Ctrl+Z", self.q_text_e.undo)
        self.add_action("Redo", "Redo last action", "Ctrl+Shift+Z", self.q_text_e.redo)

        self.toolbar.addSeparator()

        self.addToolBarBreak()

    def __get_open_filename(self, name):
        """Open a dialog to select a file to open."""
        filename, _ = QFileDialog.getOpenFileName(self, name)
        is_not_empty = bool(filename)
        if is_not_empty and filename is not None:
            return filename, True
        return "", False

    def __get_save_filename(self, name):
        """Open a dialog to select a file to save."""
        filename, _ = QFileDialog.getSaveFileName(self, name)
        is_not_empty = bool(filename)
        if is_not_empty and filename is not None:
            return filename, True
        return "", False

    def add_action(self, name, status_tip, shortcut, connect):
        """Add an action with a button and icon.

        :param name: name of the action
        :param status_tip: tip for the user to browse with the mouse
        :param shortcut: keyboard shortcut (Ctrl+S) for instance
        :param connect: method to call at trigger
        """
        icon_path = join(dirname(__file__), "icons", name + ".png")
        if exists(icon_path):
            icon = QIcon(icon_path)
            action = QAction(icon, name, self)
        else:
            action = QAction(name, self)
        action.setStatusTip(status_tip)
        action.setShortcut(shortcut)
        action.triggered.connect(connect)
        self.toolbar.addAction(action)
        return action

    def open_doc(self):
        """Open the document for edition of the template."""
        filename, is_ok = self.__get_open_filename("Open File")
        if is_ok:
            f_handler = open(filename, "r")
            filedata = f_handler.read()
            self.q_text_e.setText(filedata)
            f_handler.close()
            self.hightlight(self.in_sep, "green")
            self.hightlight(self.out_sep)

    def save_doc(self):
        """Save the template to a file."""
        filename, is_ok = self.__get_save_filename("Save File")
        if is_ok:
            f_handler = open(filename, "w")
            filedata = self.q_text_e.toPlainText()
            f_handler.write(filedata)
            f_handler.close()

    def make_input(self):
        """Make an input from the selected data."""
        name, is_ok = QInputDialog.getText(self, "Input name", "Enter the input name:")
        if is_ok:
            name = str(name)
            cursor = self.q_text_e.textCursor()

            selection = cursor.selection().toPlainText()

            tag = self.in_sep + "{" + name + "::" + selection + "}"
            cursor.insertText(tag)
            self.hightlight(self.in_sep, "green")

    def make_output(self):
        """Make an output from the selected data."""
        name, is_ok = QInputDialog.getText(
            self, "Output name", "Enter the output name:"
        )
        if is_ok:
            name = str(name)
            cursor = self.q_text_e.textCursor()
            selection = cursor.selection().toPlainText()
            tag = self.out_sep + "{" + name + "::" + selection + "}"
            cursor.insertText(tag)
            self.hightlight(self.out_sep)

    def hightlight(self, sep, color="red"):
        """Highight some text.

        :param sep: the regex that validates the text to highlight
        :param color: the color to be used
        """
        # Setup the desired format for matches
        color = QColor(color)
        color.setAlpha(100)
        # Setup the regex engine
        regex = QRegExp(sep)
        # Process the displayed document
        index = regex.indexIn(self.q_text_e.toPlainText(), 0)
        cursor = self.q_text_e.textCursor()

        while index != -1:
            # Select the matched text and apply the desired format
            cursor.setPosition(index)
            cursor.movePosition(QTextCursor.EndOfWord, QTextCursor.KeepAnchor, n=1)
            charfmt = cursor.charFormat()
            charfmt.setBackground(color)
            cursor.setCharFormat(charfmt)

            # Move to the next match
            pos = index + regex.matchedLength()
            index = regex.indexIn(self.q_text_e.toPlainText(), pos)


def main():
    """Entry point."""
    app = QApplication(sys.argv)
    editor = QtTemplateEditor()
    editor.show()
    sys.exit(app.exec_())
