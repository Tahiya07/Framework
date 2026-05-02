"""
report.py
===========================================================
Generates paper-ready tables for results section.
"""

import numpy as np


def generate_latex_table(results: dict) -> str:
    """
    Converts evaluation dict → LaTeX table.
    """

    rows = []
    for k, v in results.items():
        rows.append(f"{k} & {v:.4f} \\\\")

    table = r"""
\begin{table}[h]
\centering
\begin{tabular}{lc}
Metric & Score \\
\hline
""" + "\n".join(rows) + r"""
\end{tabular}
\caption{Bloom classification evaluation results}
\end{table}
"""

    return table