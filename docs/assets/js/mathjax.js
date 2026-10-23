/*
MathJax configuration and on-demand loading.

The library weighs about one megabyte and only a handful of pages carry formulas,
so it is not listed in `extra_javascript`: it is fetched here the first time a
page containing math is displayed. A per-page condition in the templates would
not work, because with the `navigation.instant` feature only the scripts of the
very first page loaded are ever executed; landing on a page without math would
then leave MathJax unavailable for the whole session.
*/

window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

(function () {
  "use strict";

  var SRC = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js";
  var requested = false;

  document$.subscribe(function () {
    // `pymdownx.arithmatex` in generic mode wraps every formula in an element
    // carrying this class, so its absence means the page has no math at all.
    if (!document.querySelector(".arithmatex")) return;

    if (!requested) {
      requested = true;
      var script = document.createElement("script");
      script.src = SRC;
      script.async = true;
      // MathJax typesets the document by itself once it has loaded, so there is
      // nothing left to do for this first page.
      document.head.appendChild(script);
      return;
    }

    // Later pages reached through instant navigation: the library is already
    // there (or still loading, in which case it will typeset them on arrival).
    if (window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.startup.output.clearCache();
      window.MathJax.typesetClear();
      window.MathJax.texReset();
      window.MathJax.typesetPromise();
    }
  });
})();
