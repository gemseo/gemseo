/*
 * Make the mike version selector behave like a select box: mkdocs-material opens it on
 * hover (and keeps it open on `:focus-within`), which reads as a modal flickering under
 * the pointer. Here a click toggles it and it stays open until dismissed; the matching
 * `.md-version--open` rules live in assets/css/header.css.
 *
 * The handlers are delegated to `document` rather than bound to the button, because
 * material injects the selector asynchronously, after fetching versions.json.
 */
(function () {
  var OPEN_CLASS = "md-version--open";

  function setOpen(selector, open) {
    selector.classList.toggle(OPEN_CLASS, open);
    var button = selector.querySelector(".md-version__current");
    if (button) {
      button.setAttribute("aria-expanded", open ? "true" : "false");
    }
  }

  function closeAll() {
    var selectors = document.querySelectorAll("." + OPEN_CLASS);
    for (var i = 0; i < selectors.length; i++) {
      setOpen(selectors[i], false);
    }
  }

  /* Material ships the button without `aria-expanded`, since it has no open state to
     report; announce the collapsed one as soon as the button exists. The selector then
     stays in place — instant navigation only swaps `[data-md-component=header-topic]`,
     which the header override does not mark — so the observer is a one-shot. */
  function initialize() {
    var button = document.querySelector(".md-version__current");
    if (!button) return false;
    button.setAttribute("aria-expanded", "false");
    return true;
  }

  /* Material appends the selector to the `.md-header__topic` the header override keeps
     for it (a single `appendChild` once versions.json resolves), so watch that node
     rather than the whole document: no `subtree`, and nothing else mutates it. On a build
     without mike — `mkdocs build` or `mkdocs serve` from a checkout — the fetch 404s and
     the selector never comes, hence the timeout: the observer has to go away in that case
     too, and the button it would have annotated does not exist to be clicked anyway. */
  var WAIT_MS = 10000;

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

  document.addEventListener("click", function (event) {
    var target = event.target;
    if (!(target instanceof Element)) {
      closeAll();
      return;
    }

    var button = target.closest(".md-version__current");
    if (!button) {
      /* Anywhere else on the page: the list is dismissed by the next click, wherever it
         lands. A version link navigates in the same tab, legacy ones included (see
         assets/js/legacy-versions.js), so this only has to matter for the clicks that do
         not navigate. */
      closeAll();
      return;
    }

    var selector = button.closest(".md-version");
    if (!selector) return;

    var open = !selector.classList.contains(OPEN_CLASS);
    closeAll();
    setOpen(selector, open);
  });

  document.addEventListener("keydown", function (event) {
    if (event.key !== "Escape") return;

    var selector = document.querySelector("." + OPEN_CLASS);
    if (!selector) return;

    setOpen(selector, false);
    var button = selector.querySelector(".md-version__current");
    if (button) {
      button.focus();
    }
  });
})();
