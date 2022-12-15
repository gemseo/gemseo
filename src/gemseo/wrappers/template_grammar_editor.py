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
from __future__ import annotations

import sys
from os.path import dirname
from os.path import exists
from os.path import join

try:
    from PySide6.QtCore import QRegularExpression
    from PySide6.QtGui import QAction
    from PySide6.QtGui import QColor
    from PySide6.QtGui import QIcon
    from PySide6.QtGui import QTextCursor
    from PySide6.QtWidgets import QApplication
    from PySide6.QtWidgets import QFileDialog
    from PySide6.QtWidgets import QInputDialog
    from PySide6.QtWidgets import QMainWindow
    from PySide6.QtWidgets import QTextEdit

    WITH_PYSIDE6 = True
except ImportError:
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

    WITH_PYSIDE6 = False


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
        """
        Args:
            in_sep: The separator name for the input tag.
            out_sep: The separator name for the output tag.
        """
        self.in_sep = in_sep
        self.out_sep = out_sep

        QMainWindow.__init__(self, None)
        self.q_text_e = QTextEdit(self)
        self._setup_toolbars()

        if WITH_PYSIDE6:
            self.q_text_e.setTabStopDistance(12)
        else:
            self.q_text_e.setTabStopWidth(12)

        self.setCentralWidget(self.q_text_e)

        self.setGeometry(100, 100, 600, 800)

    def _setup_toolbars(self):
        """Set up the toolbars, the icons and shortcuts."""
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
        """Open a dialog to select a file to open.

        Args:
            name: The file name.

        Returns:
            The name of the file, and whether the file contains data.
        """
        filename, _ = QFileDialog.getOpenFileName(self, name)
        is_not_empty = bool(filename)
        if is_not_empty and filename is not None:
            return filename, True
        return "", False

    def __get_save_filename(self, name):
        """Open a dialog to select a file to save.

        Args:
            name: The file name.

        Returns:
            The name of the file, and whether the file contains data.
        """
        filename, _ = QFileDialog.getSaveFileName(self, name)
        is_not_empty = bool(filename)
        if is_not_empty and filename is not None:
            return filename, True
        return "", False

    def add_action(self, name, status_tip, shortcut, connect):
        """Add an action with a button and icon.

        Args:
            name: The name of the action.
            status_tip: The tip for the user to browse with the mouse.
            shortcut: The keyboard shortcut (Ctrl+S) for instance.
            connect: The method to call at trigger.
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
            f_handler = open(filename)
            self.q_text_e.setText(f_handler.read())
            f_handler.close()
            self.highlight(self.in_sep, "green")
            self.highlight(self.out_sep)

    def save_doc(self):
        """Save the template to a file."""
        filename, is_ok = self.__get_save_filename("Save File")
        if is_ok:
            f_handler = open(filename, "w")
            f_handler.write(self.q_text_e.toPlainText())
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
            self.highlight(self.in_sep, "green")

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
            self.highlight(self.out_sep)

    def highlight(self, sep, color="red"):
        """Highlight some text.

        Args:
            sep: The regex that validates the text to highlight.
            color: The color to be used.
        """
        if WITH_PYSIDE6:
            # Set up the desired format for matches
            color = QColor(color)
            color.setAlpha(100)
            # Set up the regex engine
            regex = QRegularExpression(sep)
            # Process the displayed document
            match = regex.match(self.q_text_e.toPlainText())
            index = match.capturedStart()
            cursor = self.q_text_e.textCursor()

            while index != -1:
                # Select the matched text and apply the desired format
                cursor.setPosition(index)
                cursor.movePosition(QTextCursor.EndOfWord, QTextCursor.KeepAnchor)
                char_fmt = cursor.charFormat()
                char_fmt.setBackground(color)
                cursor.setCharFormat(char_fmt)

                # Move to the next match
                pos = index + match.capturedLength()
                match = regex.match(self.q_text_e.toPlainText(), pos)
                index = match.capturedStart()
        else:
            # Set up the desired format for matches
            color = QColor(color)
            color.setAlpha(100)
            # Set up the regex engine
            regex = QRegExp(sep)
            # Process the displayed document
            index = regex.indexIn(self.q_text_e.toPlainText(), 0)
            cursor = self.q_text_e.textCursor()

            while index != -1:
                # Select the matched text and apply the desired format
                cursor.setPosition(index)
                cursor.movePosition(QTextCursor.EndOfWord, QTextCursor.KeepAnchor)
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
    if WITH_PYSIDE6:
        sys.exit(app.exec())
    else:
        sys.exit(app.exec_())
