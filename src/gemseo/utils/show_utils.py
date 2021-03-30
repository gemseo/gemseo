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

"""
Copyright ONERA, taken from WhatsOpt-CLI
https://github.com/OneraHub/WhatsOpt-CLI/blob/master/whatsopt/show_utils.py

Distributed under the Apache 2.0 license

Minor modifications by Francois Gallard : merge the two methods an comment
"""
from xdsmjs import bundlejs, css

HTML_TEMPLATE = """
<!doctype html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
<style type="text/css">
{}
</style>
<script type="text/javascript">
{}
</script>
<script type="text/javascript">
    document.addEventListener('DOMContentLoaded', () => {{
      const mdo = {};
      const config = {{
        labelizer: {{
            ellipsis: 5,
            subSupScript: false,
            showLinkNbOnly: false,
        }},
        layout: {{
            origin: {{ x: 50, y: 20 }},
            cellsize: {{ w: 150, h: 50 }},
            padding: 10,
        }},
        withDefaultDriver: true,
        withTitleTooltip: true,
      }};
      xdsmjs.XDSMjs(config).createXdsm(mdo);
    }});
</script>
</head>
<body>
    <div class="xdsm-toolbar"></div>
    <div class="xdsm2"></div>
</body>
</html>
"""


def generate_xdsm_html(xdsm, out_file_path="xdsm.html"):
    """
    Generates a self contained HTML file

    :parma xdsm: xdsm dict data
    :param out_file_path: path to the output HTML file
    """
    html = HTML_TEMPLATE.format(css(), bundlejs(), xdsm)

    with open(out_file_path, "w") as stream:
        stream.write(html)
