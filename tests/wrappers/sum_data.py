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
from __future__ import annotations

import argparse
import json


def execute(infile=None, outfile=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="input file")
    parser.add_argument("-o", help="output file")
    infile = infile or parser.parse_args().i
    outfile = outfile or parser.parse_args().o

    with open(infile) as input_f:
        data = json.load(input_f)

    with open(outfile, "w") as fout:
        sout = json.dumps({"out": sum(data.values())}, sort_keys=True, indent=4)
        fout.write(sout)


if __name__ == "__main__":
    execute()
