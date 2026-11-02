"""Efficient global multidisciplinary design optimization (EGMDO and C-EGMDO, or EGMDA).

The disciplinary solvers of a multidisciplinary design optimization problem are replaced by
Gaussian process surrogates, enriched adaptively: where the optimum is likely to be (EGMDO,
`Mdegor(method="egmdo")`), or where `Egor` needs the multidisciplinary analysis solved
(EGMDA, `Mdegor(method="egmda")`). See `Mdegor` and `Discipline`.

References
----------
S. Dubreuil, N. Bartoli, C. Gogu, T. Lefebvre, Towards an efficient global multidisciplinary
design optimization algorithm, Struct. Multidisc. Optim. 62:1739-1765, 2020.

I. Cardoso, S. Dubreuil, N. Bartoli, C. Gogu, M. Salaün, Constrained efficient global
multidisciplinary design optimization using adaptive disciplinary surrogate enrichment,
Struct. Multidisc. Optim. 67:23, 2024.
"""

from ..egobox import (
    CorrelationSpec,
    Discipline,
    EnrichmentStrategy,
    InfillStrategy,
    Mdegor,
    MdegorRequest,
    MdegorResult,
)

__all__ = [
    "CorrelationSpec",
    "Discipline",
    "EnrichmentStrategy",
    "InfillStrategy",
    "Mdegor",
    "MdegorRequest",
    "MdegorResult",
]
