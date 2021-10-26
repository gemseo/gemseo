# Copyright 2021 IRT Saint-Exupéry, https://www.irt-saintexupery.com
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
from gemseo_templator.blocks.template import Block, WebLink

block = Block(
    title="DOE &#x26; Trade-off",
    description=(
        "Define, solve and post-process a trade-off problem "
        "from a DOE (design of experiments) algorithm."
    ),
    url="algorithms/doe_algos.html",
    algorithms=[
        WebLink("axial", anchor="ot-axial"),
        WebLink("bilevel full-factorial", anchor="ff2n"),
        WebLink("Box-Behnken", anchor="bbdesignl"),
        WebLink("central-composite", anchor="ccdesign"),
        WebLink("composite", anchor="ot-composite"),
        WebLink("custom", anchor="customdoe"),
        WebLink("diagonal", anchor="diagonaldoe"),
        WebLink("Faure", anchor="faure"),
        WebLink("full-factorial", anchor="fullfact"),
        WebLink("Halton", anchor="ot-halton"),
        WebLink("Haselgrove", anchor="ot-haselgrove"),
        WebLink("LHS", anchor="ot-lhs"),
        WebLink("Monte Carlo", anchor="ot-monte-carlo"),
        WebLink("Plackett-Burman", anchor="pbdesign"),
        WebLink("reverse Halton", anchor="ot-reverse-halton"),
        WebLink("Sobol", anchor="ot-sobol"),
    ],
    dependencies=[
        WebLink("OpenTURNS", url="http://www.openturns.org/"),
        WebLink("pyDOE", url="https://pythonhosted.org/pyDOE/"),
    ],
    examples="examples/doe/index.html",
    info="mdo.html",
    options="algorithms/doe_algos.html",
)
