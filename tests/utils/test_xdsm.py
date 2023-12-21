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
"""Tests for XDSM."""

from __future__ import annotations

import re
from pathlib import Path
from unittest import mock

import pytest

import gemseo.utils.xdsm as xdsm_module
from gemseo.utils.xdsm import XDSM


@pytest.fixture()
def xdsm() -> XDSM:
    """The view of an XDSM."""
    return XDSM({"foo": "bar"}, Path("xdsm_path"))


@pytest.fixture()
def xdsm_without_html_file() -> XDSM:
    """The view of an XDSM without an HTML file."""
    return XDSM({"foo": "bar"}, None)


def test_html_file(xdsm):
    """Check HTML file path."""
    assert xdsm.html_file_path == Path("xdsm_path")


def test_no_html_file(xdsm_without_html_file):
    """Check HTML file path when missing."""
    assert xdsm_without_html_file.html_file_path is None


def test_json_schema(xdsm):
    """Check the JSON schema."""
    assert xdsm.json_schema == {"foo": "bar"}


def test_visualize(xdsm):
    """Check the visualization of a XDSM."""
    with mock.patch.object(xdsm_module, "webbrowser") as mock_object:
        xdsm.visualize()
        assert mock_object.open.call_args.args == ("file://xdsm_path",)


def test_visualize_without_html_file(xdsm_without_html_file):
    """Check the visualization without HTML file."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "A HTML file is required to visualize the XDSM in a web browser."
        ),
    ):
        xdsm_without_html_file.visualize()


def test__repr_html_(xdsm):
    """Check the HTML representation."""
    html = xdsm._repr_html_()
    expected = (
        xdsm._XDSM__XDSM_TEMPLATE.format(xdsm.json_schema)
        + "<div class='xdsm-toolbar'></div><div class='xdsm2'></div>"
    )
    assert html == expected
