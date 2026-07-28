---
status: draft
description: ""
tags: []
search:
  boost: 1
hide:
  - navigation
  - toc
---

<!--
Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->
<h1 style="display: none;">Home</h1>

<div class="gemseo-home">

<section class="gh-intro">
  <div>
    <img class="gh-logo" src="assets/images/gemseo/gemseo_logo_transparent.svg" alt="GEMSEO" />
    <p class="gh-intro__name">Generic Engine for Multidisciplinary Scenarios, Exploration and Optimization</p>
    <p class="gh-intro__lead">An open-source Python library for multidisciplinary studies.</p>
    <p class="gh-intro__note">Looking for plugins, blog posts, and contributing advice? Those live on <a href="https://gemseo.org" target="_blank" rel="noopener">gemseo.org</a></p>
  </div>
  <div class="gh-card">
    <div class="gh-card__head">
      <span class="gh-card__title">Quick start</span>
    </div>
    <div class="gh-qtabs" role="tablist" aria-label="Quick start steps">
      <button type="button" class="gh-qtab gh-qtab--active" role="tab" aria-selected="true" data-panel="install">Install</button>
      <button type="button" class="gh-qtab" role="tab" aria-selected="false" data-panel="firstrun">First run</button>
    </div>
    <div class="gh-panels">
      <div class="gh-panel gh-panel--active" data-panel="install" role="tabpanel">
        <div class="gh-tabs" role="tablist" aria-label="Package manager">
          <button type="button" class="gh-tab gh-tab--active" role="tab" aria-selected="true" data-tab="uv" data-cmd="uv pip install gemseo[all]">uv</button>
          <button type="button" class="gh-tab" role="tab" aria-selected="false" data-tab="pip" data-cmd="pip install gemseo[all]">pip</button>
          <button type="button" class="gh-tab" role="tab" aria-selected="false" data-tab="conda" data-cmd="conda install -c conda-forge gemseo">conda</button>
        </div>
        <div class="gh-cmd">
          <span class="gh-cmd__prompt">$</span>
          <span class="gh-cmd__text">uv pip install gemseo[all]</span>
          <button type="button" class="gh-cmd__copy" data-copy="uv pip install gemseo[all]">Copy</button>
        </div>
      </div>
      <div class="gh-panel" data-panel="firstrun" role="tabpanel" hidden>
        <div class="gh-code__wrap">
          <button type="button" class="gh-cmd__copy gh-code__copy" data-copy="from gemseo.algos.design_space import DesignSpace&#10;from gemseo.disciplines import AnalyticDiscipline&#10;from gemseo.scenarios.mdo import MDOScenario&#10;from gemseo.settings.opt import SLSQP_Settings&#10;&#10;discipline = AnalyticDiscipline(&#10;    expressions={&quot;y&quot;: &quot;(x - 2) ** 2 + 1&quot;},&#10;)&#10;&#10;design_space = DesignSpace()&#10;design_space.add_variable(&#10;    &quot;x&quot;,&#10;    lower_bound=-5.0,&#10;    upper_bound=5.0,&#10;    value=0.0,&#10;)&#10;&#10;scenario = MDOScenario(&#10;    [discipline],&#10;    design_space,&#10;)&#10;scenario.add_objective(&quot;y&quot;)&#10;scenario.execute(SLSQP_Settings(max_iter=10))&#10;">Copy</button>
          <pre class="gh-code"><code>from gemseo.scenarios.mdo import MDOScenario
from gemseo.settings.opt import SLSQP_Settings
from gemseo.algos.design_space import DesignSpace
from gemseo.disciplines.analytic import AnalyticDiscipline

discipline = AnalyticDiscipline(
    expressions={"y": "(x - 2) ** 2 + 1"},
)

design_space = DesignSpace()
design_space.add_variable(
    "x",
    lower_bound=-5.0,
    upper_bound=5.0,
    value=0.0,
)

scenario = MDOScenario(
    [discipline],
    design_space,
)
scenario.add_objective("y")
scenario.execute(SLSQP_Settings(max_iter=10))
</code></pre>
        </div>
      </div>
    </div>
    <div class="gh-card__foot">
      <a href="software/installation/">Installation &amp; setup</a>
      <a href="generated/examples/tutorials/basic/plot_gemseo_in_10_minutes/">GEMSEO in 10 minutes</a>
    </div>
  </div>
</section>

<section class="gh-lp">
  <div id="gemseo-lp"></div>
</section>

</div>
