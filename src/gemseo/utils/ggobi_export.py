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
#       :author : Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Export data to the XML file format needed by GGOBI."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from xml.dom import minidom
from xml.etree import ElementTree
from xml.etree.ElementTree import Comment
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import SubElement

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy import ndarray


def prettify(elem):
    """Return a pretty-printed XML string for the Element.

    Args:
        elem: The xml element.
    """
    rough_string = ElementTree.tostring(elem, "utf-8")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="\t")


def save_data_arrays_to_xml(
    variable_names: Sequence[str],
    values_array: ndarray,
    file_path: str | Path = "opt_hist.xml",
) -> None:
    """Save an optimization history in NumPy format to a xml file to be read by ggobi.

    Args:
        variable_names: The names of the variables.
        values_array: The history of the variables,
            of the form *(number of variables, number of iterations)*.
        file_path: The file path of the generated xml file.
    """
    file_path = Path(file_path)
    if file_path.exists():
        file_path.unlink()
    nb_records = values_array.shape[0]
    nb_variables = len(variable_names)

    root = Element("ggobidata")
    comment = Comment('DOCTYPE ggobidata SYSTEM "ggobi.dtd"')
    root.append(comment)
    ggobidata = SubElement(root, "ggobidata")
    SubElement(ggobidata, "brush", attrib={"colo": "6", "glyph": "fc 3"})
    data = SubElement(ggobidata, "data", attrib={"name": "opt_history"})
    SubElement(data, "description", attrib={"source": "Optimization history"})
    variables = SubElement(data, "variables", attrib={"count": str(nb_variables)})
    for var in variable_names:
        variables_attr = {"name": str(var)}
        SubElement(variables, "realvariable", attrib=variables_attr)
    records = SubElement(
        data,
        "records",
        attrib={"color": str(2), "count": str(nb_records), "missingValue": "NA"},
    )
    for i_rec in range(nb_records):
        rec = SubElement(
            records, "record", attrib={"color": str(1), "label": "Iter_" + str(i_rec)}
        )
        rec.text = str(values_array[i_rec, :]).replace("[", "").replace("]", "")

    with file_path.open("w") as xml_file:
        xml_file.write(prettify(root))
