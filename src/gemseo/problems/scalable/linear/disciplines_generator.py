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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: François Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Dummy linear discipline generator
=================================

A utility that generates dummy disciplines from a specification.
The inputs and output names are specified by the user.
A linear random dependency between the inputs and outputs is created.
The size of the inputs and outputs can be parametrized by the user.
The MDA of the generated disciplines will always converge because all the outputs
are in [0, 1] if the inputs are in [0, 1].
The analytic Jacobian is provided.
"""
from __future__ import annotations

import string
from itertools import islice
from itertools import permutations

from numpy import arange
from numpy import array
from numpy import concatenate
from numpy import setdiff1d
from numpy import unique
from numpy.random import shuffle

from gemseo.core.discipline import MDODiscipline
from gemseo.problems.scalable.linear.linear_discipline import LinearDiscipline

DESC_5_DISC = [
    ("A", ["b"], ["a", "c"]),
    ("B", ["a"], ["b"]),
    ("C", ["c", "e"], ["d"]),
    ("D", ["d"], ["e", "f"]),
    ("E", ["f"], []),
]

DESC_16_DISC = [
    ("A", ["a"], ["b"]),
    ("B", ["c"], ["a", "n"]),
    ("C", ["b", "d"], ["c", "e"]),
    ("D", ["f"], ["d", "g"]),
    ("E", ["e"], ["f", "h", "o"]),
    ("F", ["g", "j"], ["i"]),
    ("G", ["i", "h"], ["k", "l"]),
    ("H", ["k", "m"], ["j"]),
    ("I", ["l"], ["m", "w"]),
    ("J", ["n", "o"], ["p", "q"]),
    ("K", ["y"], ["x"]),
    ("L", ["w", "x"], ["y", "z"]),
    ("M", ["p", "s"], ["r"]),
    ("N", ["r"], ["t", "u"]),
    ("O", ["q", "t"], ["s", "v"]),
    ("P", ["u", "v", "z"], ["obj"]),
]

DESC_3_DISC_WEAK = [
    ("A", ["x"], ["a"]),
    ("B", ["x", "a"], ["b"]),
    ("C", ["x", "a"], ["c"]),
]

DESC_4_DISC_WEAK = [
    ("A", ["x"], ["a"]),
    ("B", ["x", "a"], ["b"]),
    ("C", ["x", "a"], ["c"]),
    ("D", ["b", "c"], ["d"]),
]

DESC_DISC_REPEATED = [
    ("A", ["a"], ["b"]),
    ("A", ["a"], ["b"]),
    ("A", ["c"], ["d"]),
]

LETTERS = array(list(string.ascii_uppercase))


def _get_disc_names(
    nb_of_names: int,
) -> list[str]:
    """Generate names from alphabet characters combinations.

    For a given number of names, generates combinations of the characters.

    Args:
        nb_of_names: The number of names to generate.

    Returns:
        The names.
    """
    n_letters = 1

    while len(LETTERS) ** n_letters < nb_of_names:
        n_letters += 1

    return ["".join(c) for c in islice(permutations(LETTERS, n_letters), nb_of_names)]


def create_disciplines_from_sizes(
    nb_of_disc: int,
    nb_of_total_disc_io: int,
    nb_of_disc_inputs: int = 1,
    nb_of_disc_outputs: int = 1,
    inputs_size: int = 1,
    outputs_size: int = 1,
    grammar_type: str = MDODiscipline.JSON_GRAMMAR_TYPE,
    unique_disc_per_output=False,
    no_self_coupled=False,
    no_strong_couplings=False,
) -> list[LinearDiscipline]:
    """Generate a :class:`.LinearDiscipline` according to a specification.

    The names of the disciplines will be automatic combinations of capital letters.
    The names of the inputs and outputs are generated from string representations of
    integers.

    Args:
        nb_of_disc: The total number of disciplines.
        nb_of_total_disc_io: The total number of input and output data names
            in the overall process.
        nb_of_disc_inputs: The number of disciplines inputs, same for all
            disciplines.
        nb_of_disc_outputs: The number of disciplines outputs, same for all
            disciplines.
        inputs_size: The size of the input vectors,
            each input data is of shape (inputs_size,).
        outputs_size: The size of the output vectors,
            each output data is of shape (outputs_size,).
        grammar_type: The type of grammars used by the discipline.
        unique_disc_per_output: Whether to ensure that the outputs are computed by
            at most one discipline.
        no_self_coupled: Whether to ensure that no discipline has an output that is
            also an input.
        no_strong_couplings: Whether to ensure that there is no strong couplings in
            the problem.

    Returns:
        The :class:`.LinearDiscipline`.

    Raises:
        ValueError: If the number of disciplines is inconsistent with the
            total number of inputs or outputs.
    """
    if nb_of_disc_inputs > nb_of_total_disc_io:
        raise ValueError(
            "The number of disciplines inputs must be lower "
            "or equal than the total number of disciplines io"
        )

    if nb_of_disc_outputs > nb_of_total_disc_io:
        raise ValueError(
            "The number of disciplines outputs must be lower "
            "or equal than the total number of disciplines io"
        )

    disc_names = _get_disc_names(nb_of_disc)

    disc_descriptions = []

    output_names = arange(nb_of_total_disc_io)
    input_names = arange(nb_of_total_disc_io)
    used_outputs = []
    used_inputs = []

    for disc_name in disc_names:
        if no_strong_couplings:
            input_names = setdiff1d(input_names, used_outputs, True)
        # Choose inputs among all io
        shuffle(input_names)

        # There are always enough inputs because we remove outputs only when
        # using no_strong_couplings, and then outputs are empty before inputs
        disc_in_names = input_names[:nb_of_disc_inputs]

        if no_strong_couplings:
            used_inputs = unique(concatenate([used_inputs, disc_in_names]))
            output_names = setdiff1d(output_names, used_inputs, True)

        # Choose outputs
        shuffle(output_names)

        if no_self_coupled:
            output_names = setdiff1d(output_names, disc_in_names, True)

        if output_names.size < nb_of_disc_outputs:
            if output_names.size == 0:
                break
            disc_out_names = output_names
        else:
            disc_out_names = output_names[:nb_of_disc_outputs]

        if unique_disc_per_output:
            output_names = setdiff1d(output_names, disc_out_names, True)

        if no_strong_couplings:
            used_outputs = unique(concatenate([used_outputs, disc_out_names]))

        # Only create the discipline if it has at least 1 input and 1 output
        if disc_in_names.size and disc_out_names.size:
            disc_descriptions.append(
                (
                    disc_name,
                    array(disc_in_names, dtype="str").tolist(),
                    array(disc_out_names, dtype="str").tolist(),
                )
            )

    return create_disciplines_from_desc(
        disc_descriptions,
        inputs_size=inputs_size,
        outputs_size=outputs_size,
        grammar_type=grammar_type,
    )


def create_disciplines_from_desc(
    disc_descriptions,  # Sequence[Tuple[str,Sequence[str],Sequence[str]]]
    inputs_size: int = 1,
    outputs_size: int = 1,
    grammar_type: str = MDODiscipline.JSON_GRAMMAR_TYPE,
) -> list[LinearDiscipline]:
    """Generate :class:`.LinearDiscipline` classes according to a specification.

    The specification is as follows:

    .. code-block:: python

        [
        ("Disc_name1", ["in1"], ["out1", "out2"]),
        ("Disc_name2", ["in2", "out1"], ["out3", "out2"]),
        ]

    This will generate two disciplines:
      - One named "Disc_name1" with the inputs ["in1"] and the outputs ["out1", "out2"].
      - Another named "Disc_name2" with the inputs ["in2", "out1"]
        and the outputs ["out3", "out2"].

    Args:
        disc_descriptions: The specification of the disciplines,
            each item is (name, inputs_names, outputs_names),
            disciplines names may be non-unique.
        inputs_size: The size of the input vectors,
            each input data is of shape (inputs_size,).
        outputs_size: The size of the output vectors,
            each output data is of shape (outputs_size,).
        grammar_type: The type of grammars used by the disciplines.

    Returns:
        The :class:`.LinearDiscipline`.
    """
    return [
        LinearDiscipline(
            name, input_names, output_names, inputs_size, outputs_size, grammar_type
        )
        for name, input_names, output_names in disc_descriptions
    ]
