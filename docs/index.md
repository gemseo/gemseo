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
    <p class="gh-intro__text">GEMSEO is an open-source Python library designed for use in multidisciplinary studies. It stands for Generic Engine for Multidisciplinary Scenarios, Exploration and Optimization. This documentation contains tutorials, explanations, how-to guides, and references. Looking for plugins, blog posts and advice on contributing? You will find those on <a href="https://gemseo.org" target="_blank" rel="noopener">gemseo.org</a>.</p>
    <a class="gh-intro__btn" href="changelog/">Changelog</a>
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
        <div class="gh-cmds">
          <div class="gh-cmd">
            <span class="gh-cmd__label">uv</span>
            <span class="gh-cmd__prompt">$</span>
            <span class="gh-cmd__text">uv pip install gemseo[all]</span>
            <button type="button" class="gh-cmd__copy" aria-label="Copy the uv install command" data-copy="uv pip install gemseo[all]">Copy</button>
          </div>
          <div class="gh-cmd">
            <span class="gh-cmd__label">pip</span>
            <span class="gh-cmd__prompt">$</span>
            <span class="gh-cmd__text">pip install gemseo[all]</span>
            <button type="button" class="gh-cmd__copy" aria-label="Copy the pip install command" data-copy="pip install gemseo[all]">Copy</button>
          </div>
          <div class="gh-cmd">
            <span class="gh-cmd__label">conda</span>
            <span class="gh-cmd__prompt">$</span>
            <span class="gh-cmd__text">conda install -c conda-forge gemseo</span>
            <button type="button" class="gh-cmd__copy" aria-label="Copy the conda install command" data-copy="conda install -c conda-forge gemseo">Copy</button>
          </div>
        </div>
      </div>
      <div class="gh-panel" data-panel="firstrun" role="tabpanel" hidden>
        <div class="gh-code__wrap">
          <button type="button" class="gh-cmd__copy gh-code__copy" data-copy="from gemseo.discipline.analytic import AnalyticDiscipline&#10;from gemseo.optimization import SLSQP_Settings&#10;from gemseo.scenario.mdo import MDOScenario&#10;from gemseo.space.design import DesignSpace&#10;&#10;discipline = AnalyticDiscipline({&quot;y&quot;: &quot;(x-2)**2+1&quot;})&#10;&#10;space = DesignSpace()&#10;space.add_variable(&quot;x&quot;, lower_bound=-5.0, upper_bound=5.0)&#10;&#10;scenario = MDOScenario([discipline], space)&#10;scenario.add_objective(&quot;y&quot;)&#10;scenario.execute(SLSQP_Settings(max_iter=10))&#10;">Copy</button>
          <div class="gh-code">
<!-- rumdl-disable MD040 MD046 -->
```python
from gemseo.discipline.analytic import AnalyticDiscipline
from gemseo.optimization import SLSQP_Settings
from gemseo.scenario.mdo import MDOScenario
from gemseo.space.design import DesignSpace

discipline = AnalyticDiscipline({"y": "(x-2)**2+1"})

space = DesignSpace()
space.add_variable("x", lower_bound=-5.0, upper_bound=5.0)

scenario = MDOScenario([discipline], space)
scenario.add_objective("y")
scenario.execute(SLSQP_Settings(max_iter=10))

```
          </div>
        </div>
      </div>
    </div>
    <div class="gh-card__foot">
      <a href="software/installation/">Installation</a>
      <a href="generated/examples/tutorials/basic/plot_gemseo_in_10_minutes/">GEMSEO in 10 minutes</a>
    </div>
  </div>
</section>

<section class="gh-lp">
  <div id="gemseo-lp"></div>
</section>

</div>
