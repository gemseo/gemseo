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

import gemseo.utils.xdsm.xdsm as xdsm_module
from gemseo.utils.xdsm import XDSM as XDSM_
from gemseo.utils.xdsm.xdsm import XDSM
from gemseo.utils.xdsm.xdsm_to_pdf import XDSMToPDFConverter
from gemseo.utils.xdsm.xdsmizer import XDSMizer
from gemseo.utils.xdsm_to_pdf import XDSMToPDFConverter as XDSMToPDFConverter_
from gemseo.utils.xdsmizer import XDSMizer as XDSMizer_


@pytest.fixture
def xdsm() -> XDSM:
    """The view of an XDSM."""
    return XDSM({"foo": "bar"}, Path("xdsm_path"))


@pytest.fixture
def xdsm_without_html_file() -> XDSM:
    """The view of an XDSM without an HTML file."""
    return XDSM({"foo": "bar"}, None)


def test_html_file(xdsm) -> None:
    """Check HTML file path."""
    assert xdsm.html_file_path == Path("xdsm_path")


def test_no_html_file(xdsm_without_html_file) -> None:
    """Check HTML file path when missing."""
    assert not xdsm_without_html_file.html_file_path


def test_json_schema(xdsm) -> None:
    """Check the JSON schema."""
    assert xdsm.json_schema == {"foo": "bar"}


def test_visualize(xdsm) -> None:
    """Check the visualization of a XDSM."""
    with mock.patch.object(xdsm_module, "webbrowser") as mock_object:
        xdsm.visualize()
        assert mock_object.open.call_args.args == ("file://xdsm_path",)


def test_visualize_without_html_file(xdsm_without_html_file) -> None:
    """Check the visualization without HTML file."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "A HTML file is required to visualize the XDSM in a web browser."
        ),
    ):
        xdsm_without_html_file.visualize()


def test__repr_html_(xdsm) -> None:
    """Check the HTML representation."""
    html = xdsm._repr_html_()
    expected = (
        xdsm._XDSM__XDSM_TEMPLATE.format(xdsm.json_schema)
        + "<div class='xdsm-toolbar'></div><div class='xdsm2'></div>"
    )
    assert html == expected


@pytest.mark.parametrize(
    ("cls_", "cls"),
    [(XDSM_, XDSM), (XDSMToPDFConverter_, XDSMToPDFConverter), (XDSMizer_, XDSMizer)],
)
def test_aliases(cls_, cls):
    """Check aliases related to deprecated modules."""
    assert cls_ == cls
