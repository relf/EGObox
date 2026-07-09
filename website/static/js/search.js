function debounce(func, wait) {
  var timeout;

  return function () {
    var context = this;
    var args = arguments;
    clearTimeout(timeout);

    timeout = setTimeout(function () {
      timeout = null;
      func.apply(context, args);
    }, wait);
  };
}

function escapeRegExp(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function stripText(text) {
  return (text || '')
    .replace(/\s+/g, ' ')
    .replace(/\u00a0/g, ' ')
    .trim();
}

function buildSnippet(text, term) {
  var normalized = stripText(text);
  if (!normalized) {
    return '';
  }

  var lowerText = normalized.toLowerCase();
  var lowerTerm = term.toLowerCase();
  var position = lowerText.indexOf(lowerTerm);

  if (position === -1) {
    return normalized.slice(0, 180);
  }

  var start = Math.max(0, position - 60);
  var end = Math.min(normalized.length, position + term.length + 120);
  var snippet = normalized.slice(start, end);
  var highlight = new RegExp(escapeRegExp(term), 'ig');

  if (start > 0) {
    snippet = '…' + snippet;
  }
  if (end < normalized.length) {
    snippet += '…';
  }

  return snippet.replace(highlight, function (match) {
    return '<b>' + match + '</b>';
  });
}

function initSearch() {
  var root = document.querySelector('[data-search-root]');
  if (!root) return;

  var searchInput = document.getElementById('site-search');
  var searchResults = document.querySelector('[data-search-results]');
  var MAX_ITEMS = 10;
  var currentTerm = '';
  var pageCache = null;
  var pageTextCache = new Map();

  function getSearchTargets() {
    var links = Array.prototype.slice.call(document.querySelectorAll('.documentation__sidebar a[data-page-link="true"]'));
    var seen = new Set();
    return links
      .map(function (link) {
        return {
          title: stripText(link.textContent),
          href: link.href
        };
      })
      .filter(function (item) {
        if (!item.href || seen.has(item.href)) {
          return false;
        }
        seen.add(item.href);
        return true;
      });
  }

  function fetchPageText(page) {
    if (pageTextCache.has(page.href)) {
      return pageTextCache.get(page.href);
    }

    var promise = fetch(page.href, { credentials: 'same-origin' })
      .then(function (response) {
        if (!response.ok) {
          throw new Error('Failed to fetch ' + page.href);
        }
        return response.text();
      })
      .then(function (html) {
        var parser = new DOMParser();
        var doc = parser.parseFromString(html, 'text/html');
        var content = doc.querySelector('.documentation__content') || doc.querySelector('main') || doc.body;
        var text = stripText(content ? content.textContent : '');
        return {
          title: page.title,
          href: page.href,
          text: text,
          textLower: text.toLowerCase()
        };
      })
      .catch(function () {
        return {
          title: page.title,
          href: page.href,
          text: page.title,
          textLower: page.title.toLowerCase()
        };
      });

    pageTextCache.set(page.href, promise);
    return promise;
  }

  function ensurePagesLoaded() {
    if (pageCache) {
      return pageCache;
    }

    var pages = getSearchTargets();
    pageCache = Promise.all(pages.map(fetchPageText));
    return pageCache;
  }

  function clearResults() {
    searchResults.innerHTML = '';
    searchResults.hidden = true;
  }

  function renderResults(items, query) {
    searchResults.innerHTML = '';

    if (items.length === 0) {
      var empty = document.createElement('div');
      empty.className = 'search-result';
      empty.textContent = 'No results';
      searchResults.appendChild(empty);
      searchResults.hidden = false;
      return;
    }

    items.slice(0, MAX_ITEMS).forEach(function (page) {
      var link = document.createElement('a');
      link.className = 'search-result';
      link.href = page.href;

      var title = document.createElement('span');
      title.className = 'search-result__title';
      title.textContent = page.title;
      link.appendChild(title);

      var summary = document.createElement('span');
      summary.className = 'search-result__summary';
      summary.innerHTML = buildSnippet(page.text, query);
      link.appendChild(summary);

      searchResults.appendChild(link);
    });

    searchResults.hidden = false;
  }

  function scorePage(page, query) {
    var needle = query.toLowerCase();
    var score = 0;

    if (page.title.toLowerCase() === needle) score += 50;
    if (page.title.toLowerCase().indexOf(needle) === 0) score += 25;
    if (page.textLower.indexOf(needle) !== -1) score += 10;

    return score;
  }

  function searchPages(query) {
    if (!query.trim()) return Promise.resolve([]);

    return ensurePagesLoaded().then(function (pages) {
      return pages
        .map(function (page) {
          return { page: page, score: scorePage(page, query) };
        })
        .filter(function (item) {
          return item.score > 0;
        })
        .sort(function (left, right) {
          return right.score - left.score;
        })
        .map(function (item) {
          return item.page;
        });
    });
  }

  function handleInput() {
    var term = searchInput.value.trim();
    if (term === currentTerm) {
      return;
    }

    currentTerm = term;

    if (term === '') {
      clearResults();
      return;
    }

    searchPages(term).then(function (results) {
      if (searchInput.value.trim() !== term) {
        return;
      }
      renderResults(results, term);
    }).catch(function () {
      clearResults();
    });
  }

  searchInput.addEventListener('input', debounce(handleInput, 120));

  searchInput.addEventListener('focus', function () {
    if (searchInput.value.trim()) {
      handleInput();
    }
  });

  document.addEventListener('click', function (event) {
    if (!root.contains(event.target)) {
      clearResults();
    }
  });
}

if (document.readyState === 'complete' || (document.readyState !== 'loading' && !document.documentElement.doScroll)) {
  initSearch();
} else {
  document.addEventListener('DOMContentLoaded', initSearch);
}
