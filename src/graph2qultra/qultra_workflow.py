"""
qultra_workflow.py
==================

Workflow completo, con numero di modi attesi calcolato dalla topologia:

    CQEDTopology prevista dal modello
        -> espansione opzionale in primitive
        -> topology_to_net(...)
        -> net qultra reale
        -> qu.QCircuit(net, f_min, f_max)
        -> mode_frequencies(), run_epr(), kappa()

Questo file va messo nella root della repo, accanto a:

    src/
    topology_to_qultra_general_complex.py

Uso rapido:

    python qultra_workflow_modes.py --demo

Oppure da Python:

    from qultra_workflow import simulate_topology

    result = simulate_topology(my_topology, f_min=1.0, f_max=9.0)
    print(result.frequencies)
    print(result.chi_matrix)
    print(result.kappa)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable
import argparse
import importlib
import json
import gc
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if os.path.isdir(SRC) and SRC not in sys.path:
    sys.path.insert(0, SRC)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from graph2qultra.topology_to_qultra import (  # noqa: E402
    QElement,
    TESTS,
    describe_topology,
    net_to_string,
    print_net,
    topology_to_net,
)


@dataclass
class QultraResult:
    topology_name: str
    ok: bool
    expected_modes: int = 0
    frequencies: list[float | str] | None = None
    chi_matrix: list[list[float | str]] | None = None
    kappa: list[float | str] | None = None
    error: str | None = None
    net_string: str | None = None


def import_qultra():
    """
    Importa qultra in modo flessibile.

    Nel tuo codice usi lo stile:

        circuit = qu.QCircuit(net, f_min, f_max)

    quindi qui provo prima `import qultra as qu`. Se nella tua installazione
    il modulo ha un nome diverso, aggiungilo nella lista `candidates`.
    """
    candidates = [
        "qultra",
        "qu",
    ]
    errors: list[str] = []
    for name in candidates:
        try:
            return importlib.import_module(name)
        except ImportError as exc:
            errors.append(f"{name}: {exc}")

    raise ImportError(
        "Non riesco a importare qultra. Ho provato: "
        + ", ".join(candidates)
        + ". Errori: "
        + " | ".join(errors)
    )


def _to_number(value: Any) -> Any:
    """Converte stringhe numeriche tipo '1.0000e-13' in float."""
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    if isinstance(value, list):
        return [_to_number(v) for v in value]
    return value


def qelement_to_qultra(element: QElement, qu) -> Any:
    """Converte un QElement testuale/intermedio in un oggetto qultra reale."""
    kind = element.kind
    nodes = [_to_number(n) for n in element.nodes]
    params = [_to_number(p) for p in element.params]

    if kind == "C":
        return qu.C(int(nodes[0]), int(nodes[1]), params[0])

    if kind == "J":
        return qu.J(int(nodes[0]), int(nodes[1]), params[0], params[1])

    if kind == "R":
        return qu.R(int(nodes[0]), int(nodes[1]), params[0])

    if kind == "CPW":
        return qu.CPW(int(nodes[0]), int(nodes[1]), params[0])

    if kind == "CPW_coupler":
        # QElement per CPW_coupler:
        #   nodes  = [n_res, 0, n_feed_aux, n_feed_load]
        #   params = [[9, D, 9], [15, 15], l]
        return qu.CPW_coupler(
            [int(n) for n in nodes],
            params[0],
            params[1],
            params[2],
        )

    raise ValueError(f"Elemento qultra non supportato: {kind}")


def qelements_to_qultra_net(elements: list[QElement], qu) -> list[Any]:
    """Converte tutta la net intermedia in lista di elementi qultra reali."""
    return [qelement_to_qultra(el, qu) for el in elements]



def _subgtype_name(node: Any) -> str:
    """Ritorna il nome del tipo di subgraph in modo robusto."""
    st = getattr(node, "subg_type", None)
    return getattr(st, "name", str(st))


def expected_mode_count(topology) -> int:
    """
    Numero di modi fisici attesi in modo generico:

        n_modes = n_nodi - n_coupler - n_feedline

    Cosi', se in futuro aggiungi FLUXONIUM o altri componenti non-coupler,
    vengono contati automaticamente come modi fisici.
    """
    nodes = list(getattr(topology, "_nodes", []))
    n_nodes = len(nodes)
    n_coupler = sum("COUPLER" in _subgtype_name(n) for n in nodes)
    n_feedline = sum(_subgtype_name(n) == "FEEDLINE" for n in nodes)
    return max(0, n_nodes - n_coupler - n_feedline)


def _float_list(values: Any, n_expected: int) -> list[float | str]:
    """Converte un array/lista in list[float], tronca/padda con '-' se mancano valori."""
    try:
        arr = np.asarray(values, dtype=object).ravel()
        out: list[float | str] = []
        for x in arr[:n_expected]:
            try:
                out.append(float(x))
            except Exception:
                out.append("-")
    except Exception:
        out = []
    if len(out) < n_expected:
        out.extend(["-"] * (n_expected - len(out)))
    return out


def _float_matrix(values: Any, n_expected: int) -> list[list[float | str]]:
    """Converte chi in matrice n_expected x n_expected; valori mancanti = '-'."""
    if n_expected <= 0:
        return []
    mat: list[list[float | str]] = [["-" for _ in range(n_expected)] for _ in range(n_expected)]
    try:
        arr = np.asarray(values, dtype=object)
        if arr.ndim == 0:
            raise ValueError("scalar chi")
        if arr.ndim == 1:
            side = int(np.sqrt(arr.size))
            if side * side == arr.size:
                arr = arr.reshape(side, side)
            else:
                flat = arr.ravel()[: n_expected * n_expected]
                for k, v in enumerate(flat):
                    try:
                        mat[k // n_expected][k % n_expected] = float(v)
                    except Exception:
                        pass
                return mat
        r = min(n_expected, arr.shape[0])
        c = min(n_expected, arr.shape[1])
        for i in range(r):
            for j in range(c):
                try:
                    mat[i][j] = float(arr[i, j])
                except Exception:
                    pass
        return mat
    except Exception:
        return mat

def default_expand_to_primitives(topology):
    """
    Hook per l'espansione macronodale -> primitiva.

    Se il tuo oggetto CQEDTopology predetto e' gia' primitivo, ritorna se stesso.
    Se invece hai gia' nel repo una funzione/metodo di espansione, questo hook
    prova alcuni nomi comuni. Puoi sostituirlo passando `expand_fn=...` a
    simulate_topology(...).
    """
    for method_name in (
        "to_primitive",
        "to_primitives",
        "expand_to_primitive",
        "expand_to_primitives",
        "primitive",
    ):
        method = getattr(topology, method_name, None)
        if callable(method):
            return method()
    return topology


def simulate_topology(
    topology,
    f_min: float,
    f_max: float,
    attrs: dict | None = None,
    rout: float = 50.0,
    expand_fn: Callable[[Any], Any] | None = default_expand_to_primitives,
    qu=None,
    verbose: bool = True,
) -> QultraResult:
    """
    Esegue il workflow completo su una singola topologia.

    Parameters
    ----------
    topology:
        CQEDTopology predetta o gia' primitiva.
    f_min, f_max:
        Range di frequenze passato a qu.QCircuit.
    attrs:
        Override parametri per label, come nel converter topology_to_net.
    rout:
        Resistenza di uscita delle feedline.
    expand_fn:
        Funzione di espansione macronodale -> primitiva. Usa None se la topologia
        e' sicuramente gia' primitiva e vuoi saltare questo passaggio.
    qu:
        Modulo qultra gia' importato. Se None, viene importato automaticamente.
    verbose:
        Se True stampa topologia, net e risultati.
    """
    topology_name = getattr(topology, "name", "unnamed_topology")

    try:
        qu = qu or import_qultra()

        primitive_topology = expand_fn(topology) if expand_fn is not None else topology
        primitive_name = getattr(primitive_topology, "name", topology_name)

        if verbose:
            print(f"\n=== TOPOLOGIA PRIMITIVA: {primitive_name} ===")
            describe_topology(primitive_topology)
            print()

        qelements = topology_to_net(primitive_topology, attrs=attrs, rout=rout)
        net_string = net_to_string(qelements, primitive_name)

        if verbose:
            print_net(qelements, primitive_name)
            print()

        qultra_net = qelements_to_qultra_net(qelements, qu)
        n_expected = expected_mode_count(primitive_topology)

        if verbose:
            print(f"# MODI ATTESI: {n_expected} = n_nodi - n_coupler - n_feedline")

        circuit = qu.QCircuit(qultra_net, f_min, f_max)

        # Frequenze: prendi solo tante frequenze quanti sono i modi attesi.
        try:
            frequencies_raw = circuit.mode_frequencies()
        except Exception:
            frequencies_raw = []
        frequencies = _float_list(frequencies_raw, n_expected)

        # Chi: se non esiste o fallisce, matrice di zeri n_expected x n_expected.
        try:
            chi_matrix_raw, _ = circuit.run_epr()
        except Exception:
            chi_matrix_raw = None
        chi_matrix = _float_matrix(chi_matrix_raw, n_expected)

        # Kappa: se non esiste o fallisce, vettore di zeri lungo n_expected.
        try:
            kappa_raw = circuit.kappa()
        except Exception:
            kappa_raw = []
        kappa = _float_list(kappa_raw, n_expected)

        result = QultraResult(
            topology_name=primitive_name,
            ok=True,
            expected_modes=n_expected,
            frequencies=frequencies,
            chi_matrix=chi_matrix,
            kappa=kappa,
            net_string=net_string,
        )

        if verbose:
            print("# RISULTATI QULTRA")
            print("frequencies =", result.frequencies)
            print("chi_matrix =")
            print(np.asarray(result.chi_matrix, dtype=object))
            print("kappa =", result.kappa)

        # Evita di tenere QCircuit in memoria nei batch.
        del circuit
        gc.collect()
        return result

    except Exception as exc:
        result = QultraResult(
            topology_name=topology_name,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )
        if verbose:
            print(f"ERRORE su {topology_name}: {result.error}")
        return result


def run_demo_topologies(
    f_min: float,
    f_max: float,
    rout: float = 50.0,
    max_topologies: int | None = None,
    output_json: str | None = "qultra_results.json",
) -> list[QultraResult]:
    """Esegue il workflow su tutte le topologie demo definite nel converter."""
    qu = import_qultra()
    results: list[QultraResult] = []

    selected_tests = TESTS if max_topologies is None else TESTS[:max_topologies]
    sep = "=" * 80

    for title, factory in selected_tests:
        print(f"\n{sep}\n{title}\n{sep}")
        try:
            topology, attrs = factory()
            result = simulate_topology(
                topology,
                attrs=attrs,
                f_min=f_min,
                f_max=f_max,
                rout=rout,
                qu=qu,
                verbose=True,
            )
        except Exception as exc:
            result = QultraResult(
                topology_name=title,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )
            print(f"ERRORE factory {title}: {result.error}")
        results.append(result)

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\nRisultati salvati in: {output_json}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Workflow CQEDTopology -> qultra -> f, chi, kappa")
    parser.add_argument("--demo", action="store_true", help="Esegue le topologie demo del converter")
    parser.add_argument("--f-min", type=float, default=1.0, help="Frequenza minima per qu.QCircuit")
    parser.add_argument("--f-max", type=float, default=9.0, help="Frequenza massima per qu.QCircuit")
    parser.add_argument("--rout", type=float, default=50.0, help="Feedline output resistance")
    parser.add_argument("--max-topologies", type=int, default=None, help="Limita il numero di topologie demo")
    parser.add_argument("--output-json", default="qultra_results.json", help="Path JSON risultati")
    args = parser.parse_args()

    if not args.demo:
        parser.error("Per ora usa --demo, oppure importa simulate_topology(...) da Python.")

    run_demo_topologies(
        f_min=args.f_min,
        f_max=args.f_max,
        rout=args.rout,
        max_topologies=args.max_topologies,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
