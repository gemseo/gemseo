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

import re

# \newacronym{doe}{DOE}{Design Of Experiments}
#
# \newglossaryentry{component}{
#     name=component,
#     description={wrapped software to be run in the workflow}
# }


class GlossaryEntry(object):
    def __init__(self, name, description):
        self.name = name
        self.description = self.__clean_txt(description)

    def __str__(self):
        return (
            "    " + str(self.name) + "\n" + "        " + str(self.description) + "\n"
        )

    @staticmethod
    def __clean_txt(text):
        text = text.replace("\n", " ").replace("\t", " ")
        while "  " in text:
            text = text.replace("  ", " ")
        return text


def convert(glossary_file="glossary.tex"):
    acronym_re = re.compile("\\\\newacronym{(.*?)}{(.*?)}{(.*?)}\n")
    gloss_re = re.compile(
        "\\\\newglossaryentry{.*?name={(.*?)}.*?description={(.*?)}", re.DOTALL
    )
    all_entries = []
    with open(glossary_file) as source:
        data = source.read()
        acronyms = acronym_re.findall(data)
        print len(acronyms), data.count("newacronym")
        # assert len(acronyms) == data.count("newacronym")
        for acronym in acronyms:
            all_entries.append(GlossaryEntry(*acronym[1:]))

        gloss_entr = gloss_re.findall(data)
        assert len(gloss_entr) == data.count("newglossaryentry")
        for entry in gloss_entr:
            all_entries.append(GlossaryEntry(*entry))

    with open(glossary_file.replace(".tex", ".rst"), "w") as out:
        lines = [".. glossary::\n"] + [str(entr) + "\n" for entr in all_entries]
        out.writelines(lines)


convert()
