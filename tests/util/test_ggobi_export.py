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

from gemseo.util.ggobi_export import save_data_arrays_to_xml

dir_path = Path(__file__).parent

name_file = "dummy.xml"
var_name = np.array(["x_1", "x_2", "x_3", "y_1", "y_2"])
var_value = np.array([[1, 2, 3, 4, 5], [1, 2, 4, 6, 7], [1, 0, 2, 1, 3]])


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
    save_data_arrays_to_xml(var_name, var_value, name_file)
    exp_ggobi = Path(name_file)
    assert exp_ggobi.exists()
    exp_ggobi.unlink()


def test_saved_names():
    """Test that the saved names in the ggobi file match the expected names."""
    save_data_arrays_to_xml(var_name, var_value, name_file)

    tree = ET.parse(name_file)
    root = tree.getroot()
    list_name = get_all_elements(root, "realvariable")
    variable_list = [node.get("name") for node in list_name]

    assert var_name.tolist() == variable_list

    exp_ggobi = Path(name_file)
    exp_ggobi.unlink()


def test_saved_values():
    """Test that the saved values in the ggobi file match the expected values."""
    save_data_arrays_to_xml(var_name, var_value, name_file)

    tree = ET.parse(name_file)
    root = tree.getroot()
    list_values = get_all_elements(root, "record")
    value_list = [node.text.split(" ") for node in list_values]

    array_saved = np.array(value_list).astype(float)
    assert var_value.tolist() == array_saved.tolist()

    exp_ggobi = Path(name_file)
    exp_ggobi.unlink()
