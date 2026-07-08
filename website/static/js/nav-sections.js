function normalizePath(pathname) {
  if (!pathname) return "/";
  var p = pathname
    .replace(/\/{2,}/g, "/")
    .replace(/\/index\.html$/i, "")
    .replace(/\/+$/, "");
  return p === "" ? "/" : p;
}

function slugify(text) {
  return text
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9\s-]/g, "")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-");
}

document.addEventListener("DOMContentLoaded", function () {
  var navLinks = Array.prototype.slice.call(
    document.querySelectorAll(".sidebar > .container > ul > li > a")
  );
  if (navLinks.length === 0) return;

  var currentPath = normalizePath(window.location.pathname);

  var activeLink = null;
  var bestLen = -1;
  navLinks.forEach(function (link) {
    try {
      var linkPath = normalizePath(new URL(link.href, window.location.origin).pathname);
      var exact = linkPath === currentPath;
      var prefix =
        linkPath !== "/" &&
        (currentPath === linkPath || currentPath.indexOf(linkPath + "/") === 0);
      if ((exact || prefix) && linkPath.length > bestLen) {
        bestLen = linkPath.length;
        activeLink = link;
      }
    } catch (_e) {
      // ignore malformed links
    }
  });

  if (!activeLink) return;
  activeLink.classList.add("active-section");

  var headings = Array.prototype.slice.call(
    document.querySelectorAll(".content h2")
  ).filter(function (h) {
    return h.textContent && h.textContent.trim().length > 0;
  });

  if (headings.length === 0) return;

  headings.forEach(function (h) {
    if (!h.id) {
      h.id = slugify(h.textContent);
    }
  });

  var subList = document.createElement("ul");
  subList.className = "subsection-links";

  headings.forEach(function (h) {
    var li = document.createElement("li");
    var a = document.createElement("a");
    a.href = "#" + h.id;
    a.textContent = h.textContent.trim();
    li.appendChild(a);
    subList.appendChild(li);
  });

  var parentLi = activeLink.closest("li");
  if (!parentLi) return;

  parentLi.appendChild(subList);
  activeLink.classList.add("has-subsections");
  subList.classList.add("open");

  activeLink.addEventListener("click", function (ev) {
    ev.preventDefault();
    subList.classList.toggle("open");
  });
});
