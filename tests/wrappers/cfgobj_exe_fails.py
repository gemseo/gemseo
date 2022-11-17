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
import sys
from ast import literal_eval


def parse_cfgobj(infile):
    data = {}
    with open(infile) as inf:
        for line in inf.readlines():
            if "=" in line:
                spl = line.strip().split("=")
                if len(spl) != 2:
                    raise ValueError("unbalanced = in line " + str(line))
                key = spl[0].strip()
                try:
                    data[key] = float(literal_eval(spl[1].strip()))
                except Exception:
                    raise ValueError("Failed to parse value as float " + str(spl[1]))
    return data


def write_output(out1, out2, outfile, failure=None):
    sout = """out 1 = {:1.18g}

[ "section 1" ]
    out 2 = {:1.18g}"""

    if failure == "wrong_len":

        sout = """out 1 = {:1.18g}

[ "section 1" ]
    out 2 = {:1.18g}
    toto"""

    with open(outfile, "w") as fout:
        fout.write(sout.format(out1, out2))


def execute(infile=None, outfile=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="input file")
    parser.add_argument("-o", help="output file")
    parser.add_argument("-f", help="type of failure")
    infile = infile or parser.parse_args().i
    outfile = outfile or parser.parse_args().o
    failure = None or parser.parse_args().f
    data = parse_cfgobj(infile)

    out1 = data["input 1"] * data["input 2"] * data["input 3"]
    out2 = data["input 1"] - data["input 2"] - data["input 3"]

    write_output(out1, out2, outfile, failure)

    return failure


if __name__ == "__main__":
    failure = execute()
    if failure == "err_code":
        sys.exit(666)
