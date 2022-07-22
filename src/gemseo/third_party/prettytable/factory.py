# -*- coding: utf-8 -*-
# Copyright (c) 2009-2014, Luke Maurits <luke@maurits.id.au>
# All rights reserved.
# With contributions from:
# * Chris Clark
#  * Klein Stephane
#  * John Filleau
# PTable is forked from original Google Code page in April, 2015, and now
# maintained by Kane Blueriver <kxxoling@gmail.com>.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * The name of the author may not be used to endorse or promote products
#   derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# Taken from :
# https://github.com/kxxoling/PTable
# https://pypi.python.org/pypi/PTable
"""
Pretty table factory
********************
"""
import csv

from ._compact import HTMLParser
from .prettytable import PrettyTable


def from_csv(filepath, field_names=None, **kwargs):
    """Loads table from CSV file

    :param filepath: param field_names:  (Default value = None)
    :param field_names:  (Default value = None)

    """
    fmtparams = {}
    for param in ["delimiter", "doublequote", "escapechar", "lineterminator",
                  "quotechar", "quoting", "skipinitialspace", "strict"]:
        if param in kwargs:
            fmtparams[param] = kwargs.pop(param)
    if fmtparams:
        reader = csv.reader(filepath, **fmtparams)
    else:
        dialect = csv.Sniffer().sniff(filepath.read(1024))
        filepath.seek(0)
        reader = csv.reader(filepath, dialect)

    table = PrettyTable(**kwargs)
    if field_names:
        table.field_names = field_names
        next(reader)
    else:
        table.field_names = [x.strip() for x in next(reader)]

    for row in reader:
        table.add_row([x.strip() for x in row if len(x.strip()) > 0])

    return table


def from_db_cursor(cursor, **kwargs):
    """

    :param cursor:
    :param kwargs:

    """
    if cursor.description:
        table = PrettyTable(**kwargs)
        table.field_names = [col[0] for col in cursor.description]
        for row in cursor.fetchall():
            table.add_row(row)
        return table
    return None


class TableHandler(HTMLParser):
    """TableHandler"""

    def __init__(self, **kwargs):
        HTMLParser.__init__(self)
        self.kwargs = kwargs
        self.tables = []
        self.last_row = []
        self.rows = []
        self.max_row_width = 0
        self.active = None
        self.last_content = ""
        self.is_last_row_header = False
        self.colspan = 0

    def handle_starttag(self, tag, attrs):
        """

        :param tag: param attrs:
        :param attrs:

        """
        self.active = tag
        if tag == "th":
            self.is_last_row_header = True
        for (key, value) in attrs:
            if key == "colspan":
                self.colspan = int(value)

    def handle_endtag(self, tag):
        """

        :param tag:

        """
        if tag in ["th", "td"]:
            stripped_content = self.last_content.strip()
            self.last_row.append(stripped_content)
            if self.colspan:
                for _ in range(1, self.colspan):
                    self.last_row.append("")
                self.colspan = 0

        if tag == "tr":
            self.rows.append(
                (self.last_row, self.is_last_row_header))
            self.max_row_width = max(self.max_row_width, len(self.last_row))
            self.last_row = []
            self.is_last_row_header = False
        if tag == "table":
            table = self.generate_table(self.rows)
            self.tables.append(table)
            self.rows = []
        self.last_content = " "
        self.active = None

    def handle_data(self, data):
        """

        :param data:

        """
        self.last_content += data

    def generate_table(self, rows):
        """Generates from a list of rows a PrettyTable object.

        :param rows:

        """
        table = PrettyTable(**self.kwargs)
        for row in self.rows:
            if len(row[0]) < self.max_row_width:
                appends = self.max_row_width - len(row[0])
                for _ in range(1, appends):
                    row[0].append("-")

            if row[1] is True:
                self.make_fields_unique(row[0])
                table.field_names = row[0]
            else:
                table.add_row(row[0])
        return table

    @staticmethod
    def make_fields_unique(fields):
        """iterates over the row and make each field unique

        :param fields: fields
        """
        for i in range(0, len(fields)):
            for j in range(i + 1, len(fields)):
                if fields[i] == fields[j]:
                    fields[j] += "'"


def from_html(html_code, **kwargs):
    """Generates a list of PrettyTables from a string of HTML code.

    Each <table> in the HTML becomes one PrettyTable object.

    :param html_code: HTML code
    :type html_code: str
    :param kwargs: additional arguments
    """

    parser = TableHandler(**kwargs)
    parser.feed(html_code)
    return parser.tables


def from_html_one(html_code, **kwargs):
    """Generates a PrettyTables from a string of HTML code which contains only
    a single <table>

    :param html_code: HTML code
    :type html_code: str
    :param kwargs: additional arguments
    """

    tables = from_html(html_code, **kwargs)
    try:
        assert len(tables) == 1
    except AssertionError:
        raise Exception("More than one <table> in provided HTML code!"
                        + "  Use from_html instead.")
    return tables[0]


def from_md(markdown, **kwargs):
    """Generate PrettyTable from markdown string.

    :param markdown: markdown type string.
    :param kwargs: additional arguments
    :returns: a PrettyTable object.

    """
    rows = markdown.split('\n')
    title_row = rows[0]
    content_rows = rows[2:]
    table = PrettyTable(**kwargs)
    table.field_names = split_md_row(title_row)
    list(map(table.add_row, list(map(split_md_row,
                                     [x for x in content_rows if x]))))
    return table


def strip_md_content(markdown):
    """Strip the blank space and `:` in markdown table content cell.

    :param markdown: a row of markdown table
    :returns: stripped content cell
    """
    return markdown.strip().strip(':').strip()


def split_md_row(row):
    """Split markdown table.

    :param row: a row of markdown table
    :returns: Split content list
    """
    return [strip_md_content(s) for s in row.strip('|').split('|')]
