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
#    INITIAL AUTHORS - API and implementation and/or documentation
#       :author : Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
GGOBI : interactive data visualization software
***********************************************

Export data to the XML file format needed by GGOBI
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
from builtins import open, range, str
from xml.dom import minidom
from xml.etree import ElementTree
from xml.etree.ElementTree import Comment, Element, SubElement

from future import standard_library

standard_library.install_aliases()


def prettify(elem):
    """Return a pretty-printed XML string for the Element.

    :param elem: the xml element
    """
    rough_string = ElementTree.tostring(elem, "utf-8")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="\t")


def save_data_arrays_to_xml(variables_names, values_array, file_path="opt_hist.xml"):
    """
    Saves an optimization history in numpy format to an xml
    file to be read by ggobi

    :param variables_names: list of the variables names
    :type variables_names: list(str)
    :param values_array: the variables history (nb variables,nb iterations)
    :param file_path: the file path of the generated xml file
    :type file_path: str
    """

    if os.path.exists(file_path):
        os.remove(file_path)
    nb_records = values_array.shape[0]
    nb_variables = len(variables_names)

    root = Element("ggobidata")
    comment = Comment('DOCTYPE ggobidata SYSTEM "ggobi.dtd"')
    root.append(comment)
    ggobidata = SubElement(root, "ggobidata")
    SubElement(ggobidata, "brush", attrib={"colo": "6", "glyph": "fc 3"})
    data = SubElement(ggobidata, "data", attrib={"name": "opt_history"})
    SubElement(data, "description", attrib={"source": "Optimization history"})
    variables = SubElement(data, "variables", attrib={"count": str(nb_variables)})
    for var in variables_names:
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

    with open(file_path, "w") as xml_file:
        xml_file.write(prettify(root))
