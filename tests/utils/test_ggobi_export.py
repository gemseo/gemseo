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
#    INITIAL AUTHORS - API and implementation and/or documentation
#       :author : Arthur Piat
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from gemseo.utils.ggobi_export import save_data_arrays_to_xml

DIR_PATH = Path(__file__).parent

NAME_FILE = "dummy.xml"
VAR_NAME = np.array(["x_1", "x_2", "x_3", "y_1", "y_2"])
VAR_VALUE = np.array([[1, 2, 3, 4, 5], [1, 2, 4, 6, 7], [1, 0, 2, 1, 3]])


def get_all_elements(root, tag_name):
    """Get a specific node in xml file, recursively.

    Args:
        root: The root node of the xml file to be analysed.
        tag_name: The name of the tag to be extracted recursively.
    """
    outlist = []
    for child in root:
        if child.tag == tag_name:
            outlist = [*outlist, child]

        list_children = get_all_elements(child, tag_name)
        if list_children:
            outlist = outlist + list_children
    return outlist


def test_generate_xml():
    """Test that the generated ggobi XML file exists."""
    save_data_arrays_to_xml(VAR_NAME, VAR_VALUE, NAME_FILE)
    exp_ggobi = Path(NAME_FILE)
    assert exp_ggobi.exists()
    exp_ggobi.unlink()


def test_saved_names():
    """Test that the saved names in the ggobi file match the expected names."""
    save_data_arrays_to_xml(VAR_NAME, VAR_VALUE, NAME_FILE)

    tree = ET.parse(NAME_FILE)
    root = tree.getroot()
    list_name = get_all_elements(root, "realvariable")
    variable_list = [node.get("name") for node in list_name]

    assert VAR_NAME.tolist() == variable_list

    exp_ggobi = Path(NAME_FILE)
    exp_ggobi.unlink()


def test_saved_values():
    """Test that the saved values in the ggobi file match the expected values."""
    save_data_arrays_to_xml(VAR_NAME, VAR_VALUE, NAME_FILE)

    tree = ET.parse(NAME_FILE)
    root = tree.getroot()
    list_values = get_all_elements(root, "record")
    value_list = [node.text.split(" ") for node in list_values]

    array_saved = np.array(value_list).astype(float)
    assert VAR_VALUE.tolist() == array_saved.tolist()

    exp_ggobi = Path(NAME_FILE)
    exp_ggobi.unlink()
