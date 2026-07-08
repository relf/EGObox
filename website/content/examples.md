+++
title = "Examples"
+++

# Examples

The Python snippets below are included directly from source files under `python/egobox/examples/` and exercised by pytest in the docs workflow before the site is published.

## Optimization with _Egor_

### Unconstrained optimization: Rastrigin

{{ include_code(
    url="https://raw.githubusercontent.com/relf/EGObox/refs/heads/master/python/egobox/examples/rastrigin.py",
    lang="python") 
}}

![Rastrigin function optimization](img/website_rastrigin.png)

### Constrained optimization: G24

{{ include_code(
    url="https://raw.githubusercontent.com/relf/EGObox/refs/heads/master/python/egobox/examples/g24.py",
    lang="python") 
}}

## Surrogate modeling with _Gpx_

{{ include_code(
    url="https://raw.githubusercontent.com/relf/EGObox/refs/heads/master/python/egobox/examples/website_gpx.py",
    lang="python") 
}}

![Gpx surrogate model](img/website_gpx.png)