/*
Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
Commons, PO Box 1866, Mountain View, CA 94042, USA.
*/

/*
Documentation home orientation panel. Renders a master/detail learning-path
widget from the JSON produced at build time by docs/_scripts/learning_paths.py.
Runs only on the home page (guarded by the presence of #gemseo-lp) and is
compatible with the mkdocs-material instant-navigation observable `document$`.
*/

(function () {
  "use strict";

  var DATA_FILE = "assets/learning_paths.json";
  // Reading times computed at build time by docs/hooks/reading_time.py, keyed
  // by page url. Pages that opt out (reading_time: false) are absent from it.
  var TIMES_FILE = "assets/reading_times.json";
  // Complexity levels computed at build time by docs/hooks/complexity.py, keyed
  // by page url. Pages that do not opt in (no valid `complexity` frontmatter)
  // are absent from it.
  var LEVELS_FILE = "assets/complexities.json";
  var LS_ROLE = "gemseo.lastRole";

  var TYPES = {
    explanation: { kicker: "Understand", label: "Explanation", dot: "oklch(0.6 0.13 300)" },
    tutorial: { kicker: "Learn", label: "Tutorial", dot: "oklch(0.58 0.12 155)" },
    howto: { kicker: "Do", label: "How-to", dot: "oklch(0.58 0.14 256)" },
    reference: { kicker: "Look up", label: "Reference", dot: "oklch(0.66 0.12 65)" },
  };
  var TYPE_ORDER = ["explanation", "tutorial", "howto", "reference"];
  var LEVEL_NAME = { beginner: "Beginner", intermediate: "Intermediate", advanced: "Advanced" };

  var cache = null; // parsed JSON, reused across instant-navigation loads
  var times = {}; // page url -> minutes, reused across instant-navigation loads
  var levels = {}; // page url -> complexity level, reused across instant-nav

  // Normalise a resource path or page url to a shared key (no surrounding
  // slashes) so both sides of the lookup match.
  function timeKey(path) {
    return String(path).replace(/^\/+/, "").replace(/\/+$/, "");
  }

  // Reading-time label for a resource, or null when none should be shown.
  // Reference resources keep a static "Reference" label; how-to and tutorial
  // never show a time; explanation resources show the computed time when their
  // target page opted in (reading_time: true), and nothing otherwise.
  function readTime(resource) {
    if (resource.type === "reference") return "Reference";
    if (resource.type === "howto" || resource.type === "tutorial") return null;
    var minutes = times[timeKey(resource.path)];
    if (minutes == null) return null;
    return minutes + " min read";
  }

  // Complexity label for a resource, or null when none should be shown. How-to
  // and tutorial resources never show one (their targets are gallery-generated
  // pages with no frontmatter); every other resource shows the level only when
  // its target page opted in via `complexity:` frontmatter.
  function complexity(resource) {
    if (resource.type === "howto" || resource.type === "tutorial") return null;
    var level = levels[timeKey(resource.path)];
    if (level == null) return null;
    return LEVEL_NAME[level] || null;
  }

  // Small DOM helper. `props.text` sets textContent (safe); everything else is
  // an attribute. `children` is an array of nodes.
  function el(tag, props, children) {
    var node = document.createElement(tag);
    props = props || {};
    Object.keys(props).forEach(function (key) {
      if (key === "text") node.textContent = props[key];
      else if (key === "class") node.className = props[key];
      else node.setAttribute(key, props[key]);
    });
    (children || []).forEach(function (child) {
      if (child) node.appendChild(child);
    });
    return node;
  }

  function isExternal(path) {
    return /^https?:\/\//i.test(path);
  }

  // Turn a resource/prereq path into anchor attributes.
  function linkAttrs(path) {
    if (isExternal(path)) return { href: path, target: "_blank", rel: "noopener" };
    return { href: path };
  }

  function onActivate(node, handler) {
    node.addEventListener("click", handler);
    node.addEventListener("keydown", function (event) {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        handler(event);
      }
    });
  }

  // ---------- rendering ----------

  function renderRole(state, role) {
    var active = role.id === state.selRole;
    var row = el("div", {
      class: "gh-role" + (active ? " gh-role--active" : ""),
      role: "button",
      tabindex: "0",
      title: role.title,
      "aria-pressed": active ? "true" : "false",
    });
    row.appendChild(el("span", { class: "gh-badge", text: role.code }));
    var textWrap = el("span", { class: "gh-role__text" }, [
      el("span", { class: "gh-role__title", text: role.title }),
      el("span", { class: "gh-role__blurb", text: role.blurb }),
    ]);
    row.appendChild(textWrap);
    onActivate(row, function () {
      if (state.selRole === role.id) return;
      state.selRole = role.id;
      try {
        localStorage.setItem(LS_ROLE, role.id);
      } catch (e) {}
      render(state);
    });
    return row;
  }

  function renderPrereqs(role) {
    var list = role.prerequisites || [];
    if (!list.length) return null;
    var chips = el("div", { class: "gh-prereq__chips" });
    list.forEach(function (pr) {
      var attrs = linkAttrs(pr.path);
      attrs.class = "gh-chip";
      var chip = el("a", attrs, [document.createTextNode(pr.title)]);
      chips.appendChild(chip);
    });
    return el("div", { class: "gh-prereq" }, [
      el("span", { class: "gh-prereq__label", text: "Prerequisites" }),
      chips,
    ]);
  }

  function renderResource(resource) {
    var attrs = linkAttrs(resource.path);
    attrs.class = "gh-res";
    var level = complexity(resource);
    var line = el("span", { class: "gh-res__line" }, [
      el("span", { class: "gh-res__title", text: resource.title }),
      level ? el("span", { class: "gh-res__level", text: level }) : null,
    ]);
    var label = readTime(resource);
    var body = el("span", { class: "gh-res__body" }, [
      line,
      el("span", { class: "gh-res__desc", text: resource.desc }),
      label ? el("span", { class: "gh-res__time", text: label }) : null,
    ]);
    return el("a", attrs, [body]);
  }

  function renderGroups(role) {
    var groups = TYPE_ORDER.map(function (type) {
      var items = (role.resources || []).filter(function (r) {
        return r.type === type;
      });
      return { type: type, items: items };
    }).filter(function (g) {
      return g.items.length > 0;
    });

    if (!groups.length) {
      var hint = el("span", { class: "gh-empty__hint" });
      hint.appendChild(document.createTextNode("Browse the full documentation from the tabs above, or "));
      hint.appendChild(el("a", { href: "https://gitlab.com/gemseo/dev/gemseo/-/issues", target: "_blank", rel: "noopener", text: "open an issue" }));
      hint.appendChild(document.createTextNode(" to request one."));
      return el("div", { class: "gh-empty" }, [
        el("span", { class: "gh-empty__title", text: "No curated pages for this goal yet." }),
        hint,
      ]);
    }

    var grid = el("div", { class: "gh-groups" });
    groups.forEach(function (group) {
      var meta = TYPES[group.type];
      var head = el("div", { class: "gh-group__head" }, [
        el("span", { class: "gh-group__dot", style: "background:" + meta.dot }),
        el("span", { class: "gh-group__kicker", text: meta.kicker }),
        el("span", { class: "gh-group__label", text: "· " + meta.label }),
      ]);
      var items = el("div", { class: "gh-group__items" });
      group.items.forEach(function (resource) {
        items.appendChild(renderResource(resource));
      });
      grid.appendChild(el("div", { class: "gh-group" }, [head, items]));
    });
    return grid;
  }

  function render(state) {
    var root = state.root;
    var goals = state.data.goals;
    var selected =
      goals.filter(function (g) {
        return g.id === state.selRole;
      })[0] || goals[0];
    state.selRole = selected.id;

    root.textContent = "";

    // Section header.
    root.appendChild(
      el("div", { class: "gh-lp__head" }, [el("h2", { text: "What do you want to do?" })])
    );

    // Master column.
    var collapseBtn = el("button", {
      class: "gh-icon-btn",
      type: "button",
      "aria-label": state.sidebarOpen ? "Collapse goal list" : "Show goal list",
      title: state.sidebarOpen ? "Collapse goal list" : "Show goal list",
      text: state.sidebarOpen ? "‹" : "›",
    });
    collapseBtn.addEventListener("click", function () {
      state.sidebarOpen = !state.sidebarOpen;
      render(state);
    });

    var masterHead = el("div", { class: "gh-master__head" });
    if (state.sidebarOpen) {
      masterHead.appendChild(el("span", { class: "gh-master__title", text: "Goals" }));
    }
    masterHead.appendChild(collapseBtn);

    var masterList = el("div", { class: "gh-master__list" });
    goals.forEach(function (role) {
      masterList.appendChild(renderRole(state, role));
    });
    var master = el("div", { class: "gh-master" }, [masterHead, masterList]);

    // Detail column.
    var detailHead = el("div", { class: "gh-detail__head" });
    detailHead.appendChild(el("span", { class: "gh-detail__code", text: selected.code }));
    detailHead.appendChild(
      el("div", { style: "min-width:0" }, [
        el("div", { class: "gh-detail__title", text: selected.title }),
        el("div", { class: "gh-detail__blurb", text: selected.blurb }),
      ])
    );

    var detail = el("div", { class: "gh-detail" }, [detailHead]);
    var prereqs = renderPrereqs(selected);
    if (prereqs) detail.appendChild(prereqs);
    detail.appendChild(renderGroups(selected));

    root.appendChild(
      el("div", { class: "gh-grid" + (state.sidebarOpen ? "" : " gh-grid--collapsed") }, [master, detail])
    );
  }

  // ---------- boot ----------

  function wireQuickStart() {
    var card = document.querySelector(".gemseo-home .gh-card");
    if (!card || card.dataset.wired) return;
    card.dataset.wired = "1";

    // Top tabs: Install / Verify / First run each reveal one fixed-height panel.
    var qtabs = card.querySelectorAll(".gh-qtab");
    var panels = card.querySelectorAll(".gh-panel");
    qtabs.forEach(function (qtab) {
      qtab.addEventListener("click", function () {
        var target = qtab.getAttribute("data-panel");
        qtabs.forEach(function (other) {
          var active = other === qtab;
          other.classList.toggle("gh-qtab--active", active);
          other.setAttribute("aria-selected", active ? "true" : "false");
        });
        panels.forEach(function (panel) {
          var active = panel.getAttribute("data-panel") === target;
          panel.classList.toggle("gh-panel--active", active);
          if (active) {
            panel.removeAttribute("hidden");
          } else {
            panel.setAttribute("hidden", "");
          }
        });
      });
    });

    // Copy buttons: every button carries its own `data-copy` payload (install
    // command, verify command, first-run snippet). One handler serves all.
    card.querySelectorAll(".gh-cmd__copy").forEach(function (button) {
      button.addEventListener("click", function () {
        var payload = button.getAttribute("data-copy") || "";
        if (!payload || !navigator.clipboard) return;
        navigator.clipboard.writeText(payload).then(
          function () {
            button.textContent = "Copied ✓";
            clearTimeout(button._ct);
            button._ct = setTimeout(function () {
              button.textContent = "Copy";
            }, 1500);
          },
          function () {}
        );
      });
    });
  }

  function initialRole(goals) {
    try {
      var last = localStorage.getItem(LS_ROLE);
      if (last && goals.some(function (g) { return g.id === last; })) {
        return last;
      }
    } catch (e) {}
    return goals[0].id;
  }

  function mount(root, data) {
    wireQuickStart();

    if (!data.goals || !data.goals.length) return;
    var state = {
      root: root,
      data: data,
      selRole: initialRole(data.goals),
      sidebarOpen: true,
    };
    render(state);
  }

  function boot() {
    var root = document.getElementById("gemseo-lp");
    if (!root) {
      // Not the home page: nothing to render.
      return;
    }
    if (root.dataset.ready === "1") return;
    root.dataset.ready = "1";

    if (cache) {
      mount(root, cache);
      return;
    }
    // Reading times are best-effort: if the file is missing, times stay empty
    // and the listing simply shows no "min read" labels.
    fetch(new URL(TIMES_FILE, document.baseURI).href)
      .then(function (response) {
        return response.ok ? response.json() : {};
      })
      .catch(function () {
        return {};
      })
      .then(function (data) {
        // Re-key by the normalised path so lookups match regardless of
        // trailing slashes (page urls keep them, resource paths may not).
        var norm = {};
        Object.keys(data || {}).forEach(function (key) {
          norm[timeKey(key)] = data[key];
        });
        times = norm;
        // Complexity levels are best-effort too: a missing file leaves levels
        // empty and no complexity tag is shown.
        return fetch(new URL(LEVELS_FILE, document.baseURI).href);
      })
      .then(function (response) {
        return response.ok ? response.json() : {};
      })
      .catch(function () {
        return {};
      })
      .then(function (data) {
        var norm = {};
        Object.keys(data || {}).forEach(function (key) {
          norm[timeKey(key)] = data[key];
        });
        levels = norm;
        return fetch(new URL(DATA_FILE, document.baseURI).href);
      })
      .then(function (response) {
        if (!response.ok) throw new Error("HTTP " + response.status);
        return response.json();
      })
      .then(function (data) {
        cache = data;
        mount(root, data);
      })
      .catch(function (error) {
        root.dataset.ready = "";
        root.appendChild(
          el("p", { class: "gh-lp__hint", text: "Could not load the learning paths (" + error.message + ")." })
        );
      });
  }

  if (typeof window.document$ !== "undefined" && window.document$.subscribe) {
    window.document$.subscribe(function () {
      boot();
    });
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
