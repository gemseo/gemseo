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

import fnmatch
import os
import re


def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename


def parse(py_file):
    """
    Parse a file and stores in the requirement set
    @param tex_file : the latex source file
    """
    header_re = re.compile("'''(\n|\r)(.*?)Copyright(.*?)'''", re.DOTALL)
    threequotes = """'''"""
    with open(py_file) as source:
        data = source.readlines()
        out_l = []
        start_header = False
        end_header = False
        for line in data:
            if not start_header and not end_header:
                if threequotes in line:
                    start_header = True

            elif start_header:
                if threequotes in line:
                    end_header = True
                    start_header = False
                    out_l.append("\n")
                else:  # Se are in the header
                    line = line.replace("#", "")
                    line = "#" + line[2:]
                    line = line.replace("{", "").replace("}", "")
                    line = line.replace("@author", ":author")  # .replace("\n", "")
                    while line.endswith(" \n"):
                        line = line[:-1] + "\n"
                    if (
                        "INITIAL AUTHORS - initial API and implementation and/or initial documentation"
                        in line
                    ):
                        line = "# INITIAL AUTHORS - initial API and implementation and/or\n#                   initial documentation\n"
                    out_l.append(line)
                    # print line
            elif end_header:
                out_l.append(line)
    #         header_blocks = header_re.findall(data)
    #         for block in header_blocks:
    #             print block
    if len(out_l) == 0:
        return
    with open(py_file, "w") as source:
        if "-mode: python; py-indent-offset: 4" not in out_l[0]:
            out_l.insert(0, firstl)
        source.writelines(out_l)


for filename in find_files(
    "../mdo_examples", "*.py"
):  # find_files('../gemseo', '*.py')
    print "Found py source:", filename
    parse(filename)


#             if req_options_grp is not None and len(
#                     req_options_grp.groups()) == 1:
#                 options_text = req_options_grp.group(1)
#                 text = req_block.replace("[" + options_text, "")
#                 text = text.replace("{", "")
#                 if len(text) < 15:
#                     raise Exception("Too short description : " +
#                                     str(req_block))
#                 name_grp = req_name_re.search(options_text)
#                 if name_grp is None and name_grp.groups < 2:
#                     raise Exception("Requirement has no name ! " +
#                                     str(req_block))
#                 name = name_grp.group(2)
#                 requiremnt = Requirement(
#                     identifier=name,
#                     value=text)
#
#                 linkto = req_linkto_re.search(options_text)
#                 target = req_target_re.search(options_text)
