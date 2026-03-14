from __future__ import annotations

import dataclasses as dc
from functools import cached_property
from typing import TYPE_CHECKING, cast

import feinsum as fnsm
import pytato as pt
from bidict import frozenbidict
from constantdict import constantdict
from pytato.array import EinsumReductionAxis
from pytools import UniqueNameGenerator
from pytools.tag import Tag, tag_dataclass

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


def _fset_union(s: Iterable[frozenset[pt.Array]]) -> frozenset[pt.Array]:
    from functools import reduce

    return reduce(lambda x, y: x | y, s, frozenset())


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
        self.direct_pred_getter = pt.analysis.DirectPredecessorsGetter()
        self.vng = UniqueNameGenerator([""])

    def get_cache_key(
        self, expr: pt.transform.ArrayOrNames
    ) -> pt.transform.ArrayOrNames:
        return expr

    def add_edge(self, from_: pt.Array, to: pt.Array) -> None:
        self.succs.setdefault(from_, set()).add(to)
        self.preds.setdefault(to, set()).add(from_)

    def post_visit(self, expr: pt.ArrayOrNames | pt.FunctionDefinition) -> None:
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


def solve_dgfem_materialization_eq_using_z3_legacy(dfg: DataFlowGraph):
    # Discontinued using it since quadratic.
    import z3

    V = frozenset(dfg.node_to_id.values())
    c = constantdict(
        {name: isinstance(node, pt.Einsum) for node, name in dfg.node_to_id.items()}
    )
    preds = dfg.id_preds
    succs = dfg.id_succs
    opt = z3.Optimize()
    # f[v]: Materialization decision variables (See Defn. (TODO) in paper)
    f = constantdict({v: z3.Bool(f"f_{v}") for v in V})
    # P[v][u]: True iff node u is in P_f(v) (See Defn. (TODO) in paper)
    # TODO: optimize this by considering only recursive preds of v.
    P = constantdict(
        {v: constantdict({u: z3.Bool(f"P_{v}_{u}") for u in V}) for v in V}
    )
    # U[v][u]: True iff node u is in U_f^E(v) (See Defn. (TODO) in paper)
    # TODO: optimize this by considering only recursive preds of v.
    U = constantdict(
        {v: constantdict({u: z3.Bool(f"U_{v}_{u}") for u in V}) for v in V}
    )

    print("Started assembling...")

    # Apply Constraints (See Defn. (TODO) in paper)
    for v in V:
        for u in V:
            if not preds[v]:
                # If no predecessors, sets are empty
                opt.add(P[v][u] == False)  # noqa: E712
                opt.add(U[v][u] == False)  # noqa: E712
            else:
                p_terms = []
                u_terms = []
                for p in preds[v]:
                    # Construct P_f(v) (See Defn. (TODO) in the paper)
                    p_terms.append(
                        z3.Or(z3.And(f[p], u == p), z3.And(z3.Not(f[p]), P[p][u]))
                    )
                    # Construct U_f^E(v) (See Defn. (TODO) in the paper)
                    if c[p] == 1:
                        u_terms.append(z3.And(z3.Not(f[p]), z3.Or(u == p, U[p][u])))
                    else:
                        u_terms.append(z3.And(z3.Not(f[p]), U[p][u]))
                # u is in the set if it comes from ANY predecessor path
                opt.add(P[v][u] == z3.Or(p_terms))
                opt.add(U[v][u] == z3.Or(u_terms))

    # Apply Constraints (See Defn. TODO)
    for v in V:
        # |P_f(v)| <= 2
        opt.add(z3.Sum([z3.If(P[v][u], 1, 0) for u in V]) <= 2)
        # |U_f^E(v)| <= 1
        opt.add(z3.Sum([z3.If(U[v][u], 1, 0) for u in V]) <= 1)
        # c(v) = 1 -> U_f^E(v) is empty
        if c[v] == 1:
            for u in V:
                opt.add(U[v][u] == False)  # noqa: E712
        # Boundary Condition: f(v) = 0 if preds(v) is empty
        if not preds[v]:
            opt.add(f[v] == False)  # noqa: E712
        # Boundary Condition: f(v) = 1 if succs(v) is empty
        if not succs[v]:
            opt.add(f[v] == True)  # noqa: E712

    # Minimize the sum of materialized nodes (See Defn. (TODO) of the paper.)
    obj = z3.Sum([z3.If(f[v], 1, 0) for v in V])
    print("Started solving...")
    opt.minimize(obj)

    if opt.check() == z3.sat:
        m = opt.model()
        print("Optimal Materialization Strategy:")
        i = 0
        for v in V:
            val = 1 if z3.is_true(m[f[v]]) else 0
            if val:
                print(f"{i}: {v}, {c[v] = }.")
            i += val
        print("Z3 solver stats:", opt.statistics())
    else:
        print("The problem is unsatisfiable!")


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


def _pt_einsum_to_feinsum(expr: pt.Einsum) -> fnsm.BatchedEinsum:
    from pytato.utils import get_einsum_subscript_str

    vng = UniqueNameGenerator()
    arg_to_name: dict[pt.Array, str] = {}
    for arg in expr.args:
        if arg not in arg_to_name:
            arg_to_name[arg] = vng("arg")

    return fnsm.einsum(
        get_einsum_subscript_str(expr),
        *[fnsm.Array(arg_to_name[arg], arg.shape, arg.dtype) for arg in expr.args],
    )


@tag_dataclass
class IncomingEisumTag(Tag):
    """
    Records the incoming einsum for a subexpression in the computational graph.
    """

    einsum: fnsm.BatchedEinsum

    def __post_init__(self) -> None:
        assert self.einsum.b == 1

    @staticmethod
    def from_pt_einsum(expr: pt.Einsum) -> IncomingEisumTag:
        return IncomingEisumTag(_pt_einsum_to_feinsum(expr))


@tag_dataclass
class EinsumAxisTag(Tag):
    """
    Tag that is attached to an ouptut axis for an array that which is of the
    form ``y <- f(y1, ... yk, einsum)``, where ``f`` is a composition of
    elementwise operations.
    """

    ensm: fnsm.BatchedEinsum
    index: str

    def __post_init__(self):
        assert fnsm.canonicalize_einsum(self.ensm) == self.ensm
        assert self.index in self.ensm.all_indices

    @staticmethod
    def from_non_canon_form(ensm: fnsm.BatchedEinsum, index: str) -> EinsumAxisTag:
        from feinsum.canonicalization import (
            get_substitution_mapping_between_isomorphic_batched_einsums,
        )

        assert index in ensm.all_indices
        canon_ensm = fnsm.canonicalize_einsum(ensm)
        subst_map = get_substitution_mapping_between_isomorphic_batched_einsums(
            ensm, canon_ensm
        )
        return EinsumAxisTag(canon_ensm, subst_map[index])


def solve_dgfem_materialization_eq_using_z3(
    dfg: DataFlowGraph,
) -> tuple[frozenset[pt.Array], Mapping[pt.Array, Tag]]:
    import z3

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
    f = constantdict({v: z3.Bool(f"f_{v}") for v in V})
    # U_f_E[v][u]: True iff node u is in U_f^E(v) (See Defn. (TODO) in paper)
    # TODO: optimize this by considering only recursive preds of v.
    U_f_E = constantdict(
        {
            v: constantdict({u: z3.Bool(f"U_{v}_{u}") for u in einsum_nodes})
            for v in V
        }
    )

    # Construct U_f^E(v) (See Defn. (TODO) in the paper)
    for v in V:
        for u in einsum_nodes:
            if not preds[v]:
                # If no predecessors, U is empty
                opt.add(U_f_E[v][u] == False)  # noqa: E712
            else:
                u_terms = []
                for p in preds[v]:
                    if c[p] == 1:
                        u_terms.append(
                            z3.And(z3.Not(f[p]), z3.Or(u == p, U_f_E[p][u]))
                        )
                    else:
                        u_terms.append(z3.And(z3.Not(f[p]), U_f_E[p][u]))
                # u is in the set if it comes from ANY predecessor path
                opt.add(U_f_E[v][u] == z3.Or(u_terms))

    # Apply Constraints (See Defn. TODO)
    for v in V:
        if c[v] == 2:
            # |U_f^E(v)| <= 1
            opt.add(z3.Sum([z3.If(U_f_E[v][u], 1, 0) for u in einsum_nodes]) <= 1)
        else:
            assert c[v] == 0 or c[v] == 1
            for u in einsum_nodes:
                opt.add(U_f_E[v][u] == False)  # noqa: E712
        # Boundary Condition: f(v) = 0 if preds(v) is empty
        if not preds[v]:
            opt.add(f[v] == False)  # noqa: E712
        # Boundary Condition: f(v) = 1 if succs(v) is empty
        if not succs[v]:
            opt.add(f[v] == True)  # noqa: E712

    # Minimize the sum of materialized nodes (See Defn. (TODO) of the paper.)
    obj = z3.Sum([z3.If(f[v], 1, 0) for v in V])
    tie_break_0 = z3.Sum([z3.If(f[v], int(c[v] == 1), int(c[v] == 1)) for v in V])
    tie_break_1 = z3.Sum(
        [
            z3.If(U_f_E[v][u], get_einsum_tiebreak_cost(dfg.node_to_id.inv[u]), 0)
            for u in einsum_nodes
            for v in V
        ]
    )
    opt.minimize(obj)
    opt.minimize(tie_break_0)
    opt.minimize(tie_break_1)

    if opt.check() == z3.sat:
        m = opt.model()
        if False:
            print("Optimal Materialization Strategy:")
            i = 0
            for v in sorted(V):
                val = 1 if z3.is_true(m[f[v]]) else 0
                if val:
                    print(f"{i}: {v}, {c[v] = }.")
                i += val
            print("Z3 solver stats:", opt.statistics())

        materialized_nodes_to_einsum_evaled = {
            v: cast(
                "pt.Einsum",
                dfg.node_to_id.inv[
                    (
                        v
                        if c[v] == 1
                        else next(
                            iter(
                                u for u in einsum_nodes if z3.is_true(m[U_f_E[v][u]])
                            )
                        )
                    )
                ],
            )
            for v in V
            if z3.is_true(m[f[v]])
        }
        return frozenset(
            {
                dfg.node_to_id.inv[v]
                for v in V
                if z3.is_true(m[f[v]]) and (len(succs[v]) > 0)
            }
        ), constantdict(
            {
                dfg.node_to_id.inv[v]: IncomingEisumTag.from_pt_einsum(ensm)
                for v, ensm in materialized_nodes_to_einsum_evaled.items()
            }
        )
    else:
        raise NotImplementedError("The materialization problem was unsatisfiable.")


def get_arrays_to_materialize(
    dfg: DataFlowGraph,
) -> tuple[frozenset[pt.Array], Mapping[pt.Array, Tag]]:
    return solve_dgfem_materialization_eq_using_z3(dfg)


def materialize_for_dgfem_opt(
    expr: pt.transform.ArrayOrNamesTc,
) -> pt.transform.ArrayOrNamesTc:
    materialized_arrays, tags = get_arrays_to_materialize(get_dataflow_graph(expr))

    def materialize_if_needed(
        expr: pt.transform.ArrayOrNames,
    ) -> pt.transform.ArrayOrNames:
        new_expr = expr
        if expr in materialized_arrays:
            new_expr = new_expr.tagged(pt.tags.ImplStored())
        try:
            tag = tags[expr]
        except KeyError:
            pass
        else:
            new_expr = new_expr.tagged(tag)

        return new_expr

    return pt.transform.map_and_copy(expr, materialize_if_needed)


def propagate_einsum_axes_tags(
    expr: pt.transform.ArrayOrNamesTc,
) -> pt.transform.ArrayOrNamesTc:
    def propagate_axis_t(
        expr: pt.transform.ArrayOrNames,
    ) -> pt.transform.ArrayOrNames:
        if isinstance(expr, pt.Array) and expr.tags_of_type(IncomingEisumTag):
            (incoming_einsum_tag,) = expr.tags_of_type(IncomingEisumTag)
            assert expr.shape == incoming_einsum_tag.einsum.shape
            new_axes = tuple(
                axis.tagged(
                    EinsumAxisTag.from_non_canon_form(
                        incoming_einsum_tag.einsum, idx
                    )
                )
                for idx, axis in zip(
                    incoming_einsum_tag.einsum.out_idx_set, expr.axes, strict=True
                )
            )
            expr = expr.replace_if_different(axes=new_axes)
        if isinstance(expr, pt.Einsum):
            fnsm_einsum = _pt_einsum_to_feinsum(expr)
            seen_redn_axis: set[EinsumReductionAxis] = set()
            for acc_descrs, in_idx_list in zip(
                expr.access_descriptors, fnsm_einsum.in_idx_sets, strict=True
            ):
                for in_idx, acc_descr in zip(in_idx_list, acc_descrs, strict=True):
                    if (
                        isinstance(acc_descr, EinsumReductionAxis)
                        and acc_descr not in seen_redn_axis
                    ):
                        expr = expr.with_tagged_reduction(
                            acc_descr,
                            EinsumAxisTag.from_non_canon_form(fnsm_einsum, in_idx),
                        )
                        seen_redn_axis.add(acc_descr)
        return expr

    return pt.transform.map_and_copy(expr, propagate_axis_t)


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
