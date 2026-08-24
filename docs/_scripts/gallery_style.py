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
"""Scheme-neutral matplotlib style for the gallery figures.

A figure is saved once but displayed on a white page in light mode and on a dark
page in slate mode, so neither black nor white ink works. The figures are
therefore saved with transparent backgrounds, letting the page show through, and
every decoration they draw on top of it uses a mid-grey that reads on both.

The gallery calls ``plt.rcdefaults()`` before every example, hence a resetter
rather than a style sheet or a module-level ``rcParams`` update.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from pathlib import Path

_INK = "#7f7f7f"
"""The color of the figure decorations.

The balance point between the two page backgrounds: a contrast ratio of 4.0
against the white page of the default scheme and 4.0 against the ``#1e2129``
page of the slate scheme. No single color does better on both.
"""

_RC_PARAMS: dict[str, Any] = {
    # Let the page background show through instead of painting a white one.
    # This makes both the figure patch and the Axes patches transparent.
    "savefig.transparent": True,
    # The ink of everything drawn on top of that background. The title follows
    # `text.color` since `axes.titlecolor` defaults to "auto".
    "text.color": _INK,
    "axes.labelcolor": _INK,
    "axes.edgecolor": _INK,
    "xtick.color": _INK,
    "ytick.color": _INK,
    # matplotlib 3.11 defaults `hatch.color` to "edge", i.e. the hatch follows the
    # edge color of the artist, but 3.10 defaults it to black; the docs are built
    # with the oldest supported Python, hence with matplotlib 3.10.
    "hatch.color": _INK,
    # `savefig.transparent` does not reach the legend patch, which would stay
    # white; only its edge is kept. A legend therefore masks nothing, so no figure
    # may rely on its patch to hide the data behind it.
    "legend.facecolor": "none",
    "legend.edgecolor": _INK,
    "legend.framealpha": 1.0,
    # Keep the grid lighter than the ink, as the default "#b0b0b0" is, but
    # without being nearly white on a dark page.
    "grid.color": _INK,
    "grid.alpha": 0.4,
}
"""The rcParams making the figures legible on both page backgrounds.

The colors of the data itself are left alone: the default property cycle reads on
both backgrounds, and re-mapping a colormap would change what a figure means.
"""


def reset_style(gallery_conf: dict[str, Any], fname: Path) -> None:
    """Apply the scheme-neutral style, undoing the ``plt.rcdefaults()`` of the gallery.

    Args:
        gallery_conf: The gallery configuration, unused.
        fname: The example about to run, unused.
    """
    import matplotlib

    matplotlib.rcParams.update(_RC_PARAMS)
