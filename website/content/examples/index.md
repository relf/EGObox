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

```bash
Using infill strategy: InfillStrategy.LOG_EI

===== Optimization Result =====
Best value (y*): [0.00348963]
Best point (x*): [ 0.00415341 -0.00058284]
```

![Rastrigin function optimization](img/rastrigin.png)

### Constrained optimization: G24

{{ include_code(
    url="https://raw.githubusercontent.com/relf/EGObox/refs/heads/master/python/egobox/examples/g24.py",
    lang="python") 
}}

```bash
Optimization f=[-5.50853583e+00  8.65985077e-04  3.83913510e-04] at [2.32948272 3.17905311]
```

## Surrogate modeling with _Gpx_

{{ include_code(
    url="https://raw.githubusercontent.com/relf/EGObox/refs/heads/master/python/egobox/examples/kriging.py",
    lang="python") 
}}

![Gpx surrogate model](img/kriging.png) 