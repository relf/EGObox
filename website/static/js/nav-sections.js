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

function headingTextWithoutAnchor(heading) {
  var clone = heading.cloneNode(true);
  var anchor = clone.querySelector(".zola-anchor");
  if (anchor) {
    anchor.remove();
  }
  return clone.textContent ? clone.textContent.trim() : "";
}

document.addEventListener("DOMContentLoaded", function () {
  var navLinks = Array.prototype.slice.call(
    document.querySelectorAll(".documentation__sidebar a[data-page-link='true']")
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
  var activeLi = activeLink.closest("li");
  if (activeLi) {
    activeLi.classList.add("active");
  }

  var headings = Array.prototype.slice.call(
    document.querySelectorAll(".documentation__content h2")
  ).filter(function (h) {
    return headingTextWithoutAnchor(h).length > 0;
  });

  if (headings.length === 0) return;

  headings.forEach(function (h) {
    if (!h.id) {
      h.id = slugify(headingTextWithoutAnchor(h));
    }
  });

  var subList = document.createElement("ul");
  subList.className = "subsection-links";

  headings.forEach(function (h, index) {
    if (index > 10) {
      return;
    }
    var li = document.createElement("li");
    var a = document.createElement("a");
    a.href = "#" + h.id;
    a.textContent = headingTextWithoutAnchor(h);
    li.appendChild(a);
    subList.appendChild(li);
  });

  var parentLi = activeLink.closest("li");
  if (!parentLi) return;

  parentLi.appendChild(subList);
  if (subList.children.length > 0) {
    subList.classList.add("open");
  }
});
