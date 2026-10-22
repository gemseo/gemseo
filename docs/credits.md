---
status: draft
description: ""
tags: []
search:
  boost: 1
---

<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

--8<-- "_docs/credits.md"

<!-- markdownlint-disable-next-line MD041 -->
### Internal dependencies

GEMSEO source code and documentation include software with compatible licenses that are listed below.

| Project | Location | License |
| ------- | -------- | ------- |
| [canvas-toBlob.js](https://github.com/eligrey/canvas-toBlob.js) | `gemseo/util/n2d3/js/canvas_toBlob.js` | MIT |
| colormap | `gemseo/post/_engine/colormap.py` | CC0 1.0 |
| [D3](https://github.com/d3/d3) 3.5.5 | `gemseo/util/n2d3/js/d3.v3.js` | BSD 3-Clause |
| [D3](https://github.com/d3/d3) modules d3-color, d3-fetch, d3-selection and d3-transition | Bundled in `xdsmjs.js`, see XDSMjs | ISC |
| [FileSaver.js](https://github.com/eligrey/FileSaver.js) | `gemseo/util/n2d3/js/FileSave.js` | MIT |
| [Font Awesome](https://fontawesome.com/) 4 icons, built with [Fontello](https://fontello.com/) | `docs/assets/css/fontello.css` | SIL OFL 1.1 |
| [Font Awesome Free](https://fontawesome.com/) 5.14.0 | `docs/_static/webfonts/` | SIL OFL 1.1 |
| [Materialize](https://materializecss.com/) 1.0.0 | `gemseo/util/n2d3/css/materialize*.css`, `gemseo/util/n2d3/js/materialize*.js` | MIT |
| [Parallel Coordinates](https://github.com/syntagmatic/parallel-coordinates) | `gemseo/util/n2d3/js/d3.parcoords.js` | BSD 3-Clause |
| [Reorder.js](https://github.com/jdfekete/reorder.js) | `gemseo/util/n2d3/js/reorder.v1.js` | BSD 3-Clause |
| [science.js](https://github.com/jasondavies/science.js) | `gemseo/util/n2d3/js/science.v1.js` | BSD 3-Clause |
| [tiny-queue](https://github.com/nolanlawson/tiny-queue) | `gemseo/util/n2d3/js/tiny-queue.js` | Apache v2.0 |
| [Werkzeug](https://github.com/pallets/werkzeug) | `gemseo/util/_filename_sanitizer.py` | BSD 3-Clause |
| [WhatsOpt-CLI](https://github.com/OneraHub/WhatsOpt-CLI) | `gemseo/util/xdsm/html.py` | Apache v2.0 |
| [XDSMjs](https://github.com/OneraHub/XDSMjs) 2.1.0 | `docs/assets/js/xdsmjs.js`, `docs/assets/css/xdsmjs.css` | Apache v2.0 |

The N2 chart pages of the documentation, `docs/_static/n2.html` and `docs/_static/n2_small.html`,
embed the JavaScript and CSS files of `gemseo/util/n2d3` listed above.
