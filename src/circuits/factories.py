"""
cqed.circuits.factories
=======================
qultra circuit factory functions.

Each function builds and returns a qu.QCircuit, or None if the parameter
combination is physically invalid (e.g. degenerate frequencies).

HOW TO ADD A NEW CIRCUIT
--------------------------
1. Add a create_<n>() function here, following the pattern below.
2. Register it in cqed/evaluation/evaluator.py (QULTRA_RUNNERS).
"""

import numpy as np

try:
    import qultra as qu
    _HAS_QULTRA = True
except ImportError:
    _HAS_QULTRA = False
    qu = None  # type: ignore


# ---------------------------------------------------------------------------
# Shared settings
# ---------------------------------------------------------------------------

F_MIN = 1   # [GHz]  lower frequency bound for QCircuit
F_MAX = 9   # [GHz]  upper frequency bound


def _check_freq(l_res: float, l_q: float, c_q: float) -> bool:
    """Return True if qubit and resonator frequencies are not degenerate."""
    c = 3e8
    res_freq = (c / np.sqrt(12.9 / 2)) / (4 * l_res)
    q_freq   = 1 / (2 * np.pi * np.sqrt(l_q * c_q))
    return q_freq != res_freq


# ---------------------------------------------------------------------------
# Circuit factories
# ---------------------------------------------------------------------------

def create_qubit(Cj: float, Lj: float):
    """Single transmon qubit."""
    if not _HAS_QULTRA:
        raise ImportError("qultra is required for circuit simulation.")
    net     = [qu.C(0, 1, Cj), qu.J(0, 1, Lj, 1)]
    circuit = qu.QCircuit(net, F_MIN, F_MAX)
    return circuit


def create_cpw_resonator(l: float):
    """Single CPW resonator."""
    if not _HAS_QULTRA:
        raise ImportError("qultra is required for circuit simulation.")
    net     = [qu.CPW(0, 1, l)]
    circuit = qu.QCircuit(net, F_MIN, F_MAX)
    return circuit


def create_qubit_resonator_feedline(
    Cj: float, Lj: float, Cg: float, l_res: float, Cp: float, Rout: float
):
    """Qubit–resonator–feedline (capacitive coupling)."""
    if not _HAS_QULTRA:
        raise ImportError("qultra is required for circuit simulation.")
    if not _check_freq(l_res, Lj, Cj):
        return None
    net = [
        qu.C(0, 1, Cj),
        qu.J(0, 1, Lj, 1),
        qu.C(1, 2, Cg),
        qu.CPW(2, 0, l_res),
        qu.C(2, 3, Cp),
        qu.R(3, 0, Rout),
    ]
    circuit = qu.QCircuit(net, F_MIN, F_MAX)
    return circuit


def create_qubit_resonator_feedline_inductive_nognd(
    Cj: float, Lj: float, Cc: float, l_res: float, Rout: float,
    d: float, l_coup: float
):
    """Qubit–resonator–feedline with inductive coupler (no ground)."""
    if not _HAS_QULTRA:
        raise ImportError("qultra is required for circuit simulation.")
    if not _check_freq(l_res, Lj, Cj):
        return None

    l = abs(l_res - l_coup)
    net = [
        qu.C(0, 1, Cj),
        qu.J(0, 1, Lj, 1),
        qu.C(1, 2, Cc),
        qu.CPW(2, 3, l),
        qu.CPW_coupler(
            [3, 0, 4, 5],
            [9, d, 9],
            [15, 15],
            l_coup,
        ),
        qu.R(4, 0, Rout),
        qu.R(5, 0, Rout),
    ]
    try:
        circuit = qu.QCircuit(net, F_MIN, F_MAX)
    except Exception:
        return None
    return circuit


def create_two_qubit_capacitive(
    Cj1: float, Lj1: float, Cj2: float, Lj2: float, Cc: float
):
    """Two capacitively-coupled transmons."""
    if not _HAS_QULTRA:
        raise ImportError("qultra is required for circuit simulation.")
    net = [
        qu.C(0, 1, Cj1), qu.J(0, 1, Lj1, 1),
        qu.C(0, 2, Cj2), qu.J(0, 2, Lj2, 1),
        qu.C(1, 2, Cc),
    ]
    circuit = qu.QCircuit(net, F_MIN, F_MAX)
    return circuit
