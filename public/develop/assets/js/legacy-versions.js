(function () {
  var legacyVersions = [
    { label: "6.3.0 (legacy)", url: "https://gemseo.readthedocs.io/en/stable/index.html" },
    { label: "6.2.0 (legacy)", url: "https://gemseo.readthedocs.io/en/6.2.0/index.html" },
    { label: "6.1.0 (legacy)", url: "https://gemseo.readthedocs.io/en/6.1.0/index.html" },
    { label: "6.0.0 (legacy)", url: "https://gemseo.readthedocs.io/en/6.0.0/index.html" },
    { label: "5.3.2 (legacy)", url: "https://gemseo.readthedocs.io/en/5.3.2/index.html" },
    { label: "5.2.0 (legacy)", url: "https://gemseo.readthedocs.io/en/5.2.0/index.html" },
    { label: "5.1.1 (legacy)", url: "https://gemseo.readthedocs.io/en/5.1.1/index.html" },
    { label: "5.0.1 (legacy)", url: "https://gemseo.readthedocs.io/en/5.0.1/index.html" },
  ];

  function appendLegacy(list) {
    if (list.querySelector(".md-version__item--divider")) return;

    var divider = document.createElement("li");
    divider.className = "md-version__item md-version__item--divider";
    divider.textContent = "Versions legacy (Sphinx)";
    list.appendChild(divider);

    legacyVersions.forEach(function (v) {
      var li = document.createElement("li");
      li.className = "md-version__item";
      var a = document.createElement("a");
      a.href = v.url;
      a.className = "md-version__link md-version__link--legacy";
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      a.textContent = v.label + " ↗";
      li.appendChild(a);
      list.appendChild(li);
    });
  }

  var observer = new MutationObserver(function () {
    var list = document.querySelector(".md-version__list");
    if (list) {
      appendLegacy(list);
    }
  });

  observer.observe(document.body, { childList: true, subtree: true });
})();
