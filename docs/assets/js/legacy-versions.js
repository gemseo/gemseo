/*
 * Append the Sphinx-era versions to the mike version selector.
 *
 * The entries are deliberately indistinguishable from the mike-managed ones: same markup
 * (`.md-version__link` and nothing else), same styling, same navigation in the same tab.
 * A visual-only marker was announced to nobody, and `target="_blank"` on some entries and
 * not others is exactly the unexpected context change WCAG 2.1 SC 3.2.5 is about, so the
 * distinction is gone rather than spelled out - do not reintroduce a marker here.
 */
(function () {
  var legacyVersions = [
    { label: "6.3.3", url: "https://gemseo.readthedocs.io/en/6.3.3/index.html" },
    { label: "6.2.0", url: "https://gemseo.readthedocs.io/en/6.2.0/index.html" },
    { label: "6.1.0", url: "https://gemseo.readthedocs.io/en/6.1.0/index.html" },
    { label: "6.0.0", url: "https://gemseo.readthedocs.io/en/6.0.0/index.html" },
    { label: "5.3.2", url: "https://gemseo.readthedocs.io/en/5.3.2/index.html" },
    { label: "5.2.0", url: "https://gemseo.readthedocs.io/en/5.2.0/index.html" },
    { label: "5.1.1", url: "https://gemseo.readthedocs.io/en/5.1.1/index.html" },
    { label: "5.0.1", url: "https://gemseo.readthedocs.io/en/5.0.1/index.html" },
  ];

  /* The links carry no marker of their own, so the guard against a second pass lives on
     the list instead of on them. */
  function appendLegacy(list) {
    if (list.dataset.legacyVersions) return;
    list.dataset.legacyVersions = "1";

    legacyVersions.forEach(function (v) {
      var li = document.createElement("li");
      li.className = "md-version__item";
      var a = document.createElement("a");
      a.href = v.url;
      a.className = "md-version__link";
      a.textContent = v.label;
      li.appendChild(a);
      list.appendChild(li);
    });
  }

  /* Material appends the whole selector to the `.md-header__topic` the header override
     keeps for it, in one `appendChild` once versions.json resolves, so the list arrives as
     a direct child of that node: watch it without `subtree`. On a build without mike -
     `mkdocs build` or `mkdocs serve` from a checkout - the fetch 404s and no list is ever
     injected, hence the timeout, so that the observer goes away in that case too. */
  var WAIT_MS = 10000;

  function initialize() {
    var list = document.querySelector(".md-version__list");
    if (!list) return false;
    appendLegacy(list);
    return true;
  }

  if (!initialize()) {
    var target = document.querySelector(".md-header__topic");
    var observer = new MutationObserver(function () {
      if (initialize()) {
        clearTimeout(timer);
        observer.disconnect();
      }
    });
    var timer = setTimeout(function () {
      observer.disconnect();
    }, WAIT_MS);
    observer.observe(target || document.body, { childList: true, subtree: !target });
  }
})();
