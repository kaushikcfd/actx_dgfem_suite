from __future__ import annotations

import dataclasses as dc
from functools import cached_property
from typing import TYPE_CHECKING, cast

import pytato as pt
from bidict import frozenbidict
from constantdict import constantdict
from pytools import UniqueNameGenerator
from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Iterable


def _fset_union(s: Iterable[frozenset[pt.Array]]) -> frozenset[pt.Array]:
    from functools import reduce

    return reduce(lambda x, y: x | y, s, frozenset[pt.Array]())


@dc.dataclass(frozen=True)
class DataFlowGraph:
    succs: constantdict[pt.Array, frozenset[pt.Array]]
    preds: constantdict[pt.Array, frozenset[pt.Array]]
    node_to_id: frozenbidict[pt.Array, str]

    def __post_init__(self) -> None:
        all_nodes = frozenset(self.node_to_id)
        assert all_nodes == frozenset(self.succs)
        assert all_nodes == frozenset(self.preds)
        assert _fset_union(self.succs.values()) <= all_nodes
        assert _fset_union(self.preds.values()) <= all_nodes

    @cached_property
    def id_preds(self) -> constantdict[str, frozenset[str]]:
        return constantdict(
            {
                self.node_to_id[k]: frozenset(self.node_to_id[v] for v in vs)
                for k, vs in self.preds.items()
            }
        )

    @cached_property
    def id_succs(self) -> constantdict[str, frozenset[str]]:
        return constantdict(
            {
                self.node_to_id[k]: frozenset(self.node_to_id[v] for v in vs)
                for k, vs in self.succs.items()
            }
        )


class DataflowBuilder(pt.transform.CachedWalkMapper[[]]):
    def __init__(self) -> None:
        super().__init__()
        self.succs: dict[pt.Array, set[pt.Array]] = {}
        self.preds: dict[pt.Array, set[pt.Array]] = {}
        self.node_to_id: dict[pt.Array, str] = {}
        self.direct_pred_getter: pt.analysis.DirectPredecessorsGetter = (
            pt.analysis.DirectPredecessorsGetter()
        )
        self.vng: UniqueNameGenerator = UniqueNameGenerator([""])

    @override
    def get_cache_key(
        self, expr: pt.transform.ArrayOrNames
    ) -> pt.transform.ArrayOrNames:
        return expr

    def add_edge(self, from_: pt.Array, to: pt.Array) -> None:
        self.succs.setdefault(from_, set()).add(to)
        self.preds.setdefault(to, set()).add(from_)

    @override
    def post_visit(
        self, expr: pt.transform.ArrayOrNames | pt.function.FunctionDefinition
    ) -> None:
        if isinstance(expr, pt.Array):
            assert not isinstance(expr, pt.NamedArray)
            for pred in self.direct_pred_getter(expr):
                if isinstance(pred, pt.Array):
                    self.add_edge(pred, expr)

            self.preds.setdefault(expr, set())
            self.succs.setdefault(expr, set())
            self.node_to_id[expr] = self.vng("")


def get_dataflow_graph(expr: pt.transform.ArrayOrNames) -> DataFlowGraph:
    builder = DataflowBuilder()
    builder(expr)
    return DataFlowGraph(
        succs=constantdict({k: frozenset(v) for k, v in builder.succs.items()}),
        preds=constantdict({k: frozenset(v) for k, v in builder.preds.items()}),
        node_to_id=frozenbidict(builder.node_to_id),
    )


def get_einsum_tiebreak_cost(ensm: pt.Einsum) -> int:
    assert isinstance(ensm, pt.Einsum)
    if pt.analysis.is_einsum_similar_to_subscript(ensm, "ik,kfj->ifj"):
        return 0
    elif pt.analysis.is_einsum_similar_to_subscript(ensm, "ik,rkj->rij"):
        return 1
    elif pt.analysis.is_einsum_similar_to_subscript(ensm, "ifj,fe,fej->ei"):
        return 2
    elif pt.analysis.is_einsum_similar_to_subscript(ensm, "xre,rij,xej->ei"):
        return 3
    else:
        assert pt.analysis.is_einsum_similar_to_subscript(ensm, "re,rij,ej->ei")
        return 4


def solve_dgfem_materialization_eq_using_z3(
    dfg: DataFlowGraph, *, use_tiebreaks: bool = True
) -> frozenset[pt.Array]:
    """
    :param use_tiebreaks: If *True*, ensures that a deterministic ordering is
        returned each time at the cost of additional computations.
    """
    import z3  # pyright: ignore[reportMissingTypeStubs]

    V = frozenset(dfg.node_to_id.values())
    c = constantdict(
        {
            name: (
                0
                if isinstance(
                    node,
                    (
                        pt.AdvancedIndexInContiguousAxes,
                        pt.AdvancedIndexInNoncontiguousAxes,
                    ),
                )
                else (1 if isinstance(node, pt.Einsum) else 2)
            )
            for node, name in dfg.node_to_id.items()
        }
    )
    einsum_nodes = frozenset({k for k, v in c.items() if v == 1})
    preds = dfg.id_preds
    succs = dfg.id_succs
    opt = z3.Optimize()
    # f[v]: Materialization decision variables (See Defn. (TODO) in paper)
    f = constantdict(
        {v: z3.Bool(f"f_{v}") for v in V}  # pyright: ignore[reportUnknownMemberType]
    )
    # U_f_E[v][u]: True iff node u is in U_f^E(v) (See Defn. (TODO) in paper)
    # TODO: optimize this by considering only recursive preds of v.
    U_f_E = constantdict(
        {
            v: constantdict(
                {
                    u: z3.Bool(  # pyright: ignore[reportUnknownMemberType]
                        f"U_{v}_{u}"
                    )
                    for u in einsum_nodes
                }
            )
            for v in V
        }
    )

    # Construct U_f^E(v) (See Defn. (TODO) in the paper)
    for v in V:
        for u in einsum_nodes:
            if not preds[v]:
                # If no predecessors, U is empty
                opt.add(  # pyright: ignore[reportUnknownMemberType]
                    U_f_E[v][u] == False  # ruff: ignore[true-false-comparison]
                )
            else:
                u_terms: list[object] = []
                for p in preds[v]:
                    if c[p] == 1:
                        u_terms.append(
                            z3.And(  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                                z3.Not(  # pyright: ignore[reportUnknownMemberType]
                                    f[p]
                                ),
                                z3.Or(  # pyright: ignore[reportUnknownMemberType]
                                    u == p, U_f_E[p][u]
                                ),
                            )
                        )
                    else:
                        u_terms.append(
                            z3.And(  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                                z3.Not(  # pyright: ignore[reportUnknownMemberType]
                                    f[p]
                                ),
                                U_f_E[p][u],
                            )
                        )
                # u is in the set if it comes from ANY predecessor path
                opt.add(  # pyright: ignore[reportUnknownMemberType]
                    U_f_E[v][u]
                    == z3.Or(u_terms)  # pyright: ignore[reportUnknownMemberType]
                )

    # Apply Constraints (See Defn. TODO)
    for v in V:
        if c[v] == 2:
            # |U_f^E(v)| <= 1
            opt.add(  # pyright: ignore[reportUnknownMemberType]
                z3.Sum(  # pyright: ignore[reportUnknownMemberType]
                    [
                        z3.If(  # pyright: ignore[reportUnknownMemberType]
                            U_f_E[v][u], 1, 0
                        )
                        for u in einsum_nodes
                    ]
                )
                <= 1
            )
        else:
            assert c[v] == 0 or c[v] == 1
            for u in einsum_nodes:
                opt.add(  # pyright: ignore[reportUnknownMemberType]
                    U_f_E[v][u] == False  # ruff: ignore[true-false-comparison]
                )
        # Boundary Condition: f(v) = 0 if preds(v) is empty
        if not preds[v]:
            opt.add(  # pyright: ignore[reportUnknownMemberType]
                f[v] == False  # ruff: ignore[true-false-comparison]
            )
        # Boundary Condition: f(v) = 1 if succs(v) is empty
        if not succs[v]:
            opt.add(  # pyright: ignore[reportUnknownMemberType]
                f[v] == True  # ruff: ignore[true-false-comparison]
            )

    # Minimize the sum of materialized nodes (See Defn. (TODO) of the paper.)
    obj = z3.Sum(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        [z3.If(f[v], 1, 0) for v in V]  # pyright: ignore[reportUnknownMemberType]
    )
    opt.minimize(obj)  # pyright: ignore[reportUnknownMemberType]
    if use_tiebreaks:
        tie_break_0 = z3.Sum(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            [
                z3.If(  # pyright: ignore[reportUnknownMemberType]
                    f[v], int(c[v] == 1), int(c[v] == 1)
                )
                for v in V
            ]
        )
        tie_break_1 = z3.Sum(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            [
                z3.If(  # pyright: ignore[reportUnknownMemberType]
                    U_f_E[v][u],
                    get_einsum_tiebreak_cost(
                        cast("pt.Einsum", dfg.node_to_id.inv[u])
                    ),
                    0,
                )
                for u in einsum_nodes
                for v in V
            ]
        )
        opt.minimize(tie_break_0)  # pyright: ignore[reportUnknownMemberType]
        opt.minimize(tie_break_1)  # pyright: ignore[reportUnknownMemberType]

    if opt.check() == z3.sat:  # pyright: ignore[reportUnknownMemberType]
        m = opt.model()
        if False:
            print(
                "Optimal Materialization Strategy:"
            )  # pyright: ignore[reportUnreachable]
            i = 0
            for v in sorted(V):
                val = 1 if z3.is_true(m[f[v]]) else 0
                if val:
                    print(f"{i}: {v}, {c[v] = }.")
                i += val
            print("Z3 solver stats:", opt.statistics())

        return frozenset(
            {
                dfg.node_to_id.inv[v]
                for v in V
                if z3.is_true(m[f[v]]) and (len(succs[v]) > 0)
            }
        )
    else:
        raise NotImplementedError("The materialization problem was unsatisfiable.")


def get_arrays_to_materialize(
    dfg: DataFlowGraph, *, use_tiebreaks: bool = False
) -> frozenset[pt.Array]:
    """
    :param use_tiebreaks: If *True*, ensures that a deterministic ordering is
        returned each time at the cost of additional computations.
    """
    return solve_dgfem_materialization_eq_using_z3(dfg, use_tiebreaks=use_tiebreaks)


def materialize_for_dgfem_opt(
    expr: pt.transform.ArrayOrNamesTc, *, use_tiebreaks: bool = False
) -> pt.transform.ArrayOrNamesTc:
    """
    :param use_tiebreaks: If *True*, ensures that a deterministic ordering is
        returned each time at the cost of additional computations.
    """
    materialized_arrays = get_arrays_to_materialize(get_dataflow_graph(expr))

    def materialize_if_needed(
        expr: pt.transform.ArrayOrNames,
    ) -> pt.transform.ArrayOrNames:
        new_expr = expr
        if expr in materialized_arrays:
            new_expr = new_expr.tagged(pt.tags.ImplStored())
        return new_expr

    return pt.transform.map_and_copy(expr, materialize_if_needed)


def make_einsum_operands_as_subst(
    expr: pt.transform.ArrayOrNamesTc,
) -> pt.transform.ArrayOrNamesTc:
    from pytato.target.loopy import ImplSubstitution

    arg_to_passthru_subst: dict[pt.Array, pt.Array] = {}

    def make_einsum_operands_as_subst(
        expr: pt.transform.ArrayOrNames,
    ) -> pt.transform.ArrayOrNames:
        if isinstance(expr, pt.Einsum):
            expr = expr.replace_if_different(
                args=tuple(
                    arg_to_passthru_subst.setdefault(
                        arg, arg[:].tagged(ImplSubstitution())
                    )
                    for arg in expr.args
                )
            )
        return expr

    return pt.transform.map_and_copy(expr, make_einsum_operands_as_subst)
