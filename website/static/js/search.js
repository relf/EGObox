(function() {
  function debounce(func, wait) {
    var timeout;
    return function() {
      var context = this;
      var args = arguments;
      clearTimeout(timeout);
      timeout = setTimeout(function() {
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
    return snippet.replace(highlight, function(match) {
      return '<b>' + match + '</b>';
    });
  }

  function initSearch() {
    var root = document.querySelector('[data-search-root]');
    if (!root) {
      console.error('Search: data-search-root element not found');
      return;
    }

    var searchInput = document.getElementById('site-search');
    var searchResults = document.querySelector('[data-search-results]');
    var MAX_ITEMS = 10;
    var currentTerm = '';
    var searchIndex = null;

    function loadSearchIndex(callback) {
      if (searchIndex) {
        callback(searchIndex);
        return;
      }
      var indexUrlElement = document.querySelector('[data-search-index-url]');
      var fallbackUrlElement = document.querySelector('[data-search-index-fallback]');
      var url = indexUrlElement ? indexUrlElement.getAttribute('data-search-index-url') : 'search_index.en.json';
      if (!url || url === '') {
        url = fallbackUrlElement ? fallbackUrlElement.getAttribute('data-search-index-fallback') : 'search_index.json';
      }
      console.log('Search: Loading index from', url);
      fetch(url)
        .then(function(response) {
          if (!response.ok) {
            throw new Error('Failed to load search index: ' + response.status);
          }
          return response.json();
        })
        .then(function(data) {
          console.log('Search: Index loaded successfully, fields:', data.fields);
          searchIndex = elasticlunr.Index.load(data);
          callback(searchIndex);
        })
        .catch(function(err) {
          console.error('Search: Failed to load search index:', err);
          callback(null);
        });
    }

    function searchIndexPages(query) {
      if (!searchIndex) {
        console.error('Search: Index not loaded yet');
        return [];
      }
      if (!query || !query.trim()) {
        return [];
      }
      console.log('Search: Searching for', query);
      
      // Get document store if available
      var docStore = searchIndex.documentStore;
      console.log('Search: Document store available:', !!docStore);
      
      var results = searchIndex.search(query.trim(), {
        fields: {
          title: { boost: 3 },
          body: { boost: 1 }
        },
        bool: "OR",
        expand: true
      });
      console.log('Search: Found', results.length, 'results');
      
      return results.map(function(r) {
        var doc = docStore ? docStore.getDoc(r.ref) : (r.doc || null);
        console.log('Search: Result doc for', r.ref, ':', doc);
        return {
          title: doc && doc.title ? doc.title : r.ref.split('/').pop().replace(/-/g, ' ').replace(/\//g, ' '),
          url: r.ref,
          body: doc && doc.body ? doc.body : ''
        };
      });
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
      items.slice(0, MAX_ITEMS).forEach(function(item) {
        var link = document.createElement('a');
        link.className = 'search-result';
        link.href = item.url;
        var title = document.createElement('span');
        title.className = 'search-result__title';
        title.textContent = item.title;
        link.appendChild(title);
        var summary = document.createElement('span');
        summary.className = 'search-result__summary';
        summary.innerHTML = buildSnippet(item.body, query);
        link.appendChild(summary);
        searchResults.appendChild(link);
      });
      searchResults.hidden = false;
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
      loadSearchIndex(function(index) {
        if (searchInput.value.trim() !== term) {
          return;
        }
        console.log('Search: Callback called, index available:', !!searchIndex);
        var results = searchIndexPages(term);
        renderResults(results, term);
      });
    }

    searchInput.addEventListener('input', debounce(handleInput, 120));
    searchInput.addEventListener('focus', function() {
      if (searchInput.value.trim()) {
        handleInput();
      }
    });
    document.addEventListener('click', function(event) {
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
})();