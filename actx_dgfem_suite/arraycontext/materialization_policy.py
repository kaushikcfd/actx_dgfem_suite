import dataclasses as dc
from collections.abc import Iterable
from functools import cached_property

import pytato as pt
from bidict import frozenbidict
from constantdict import constantdict
from pytools import UniqueNameGenerator


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
        node_to_id=frozenbidict(builder.node_to_id)
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
                    p_terms.append(z3.Or(
                        z3.And(f[p], u == p),
                        z3.And(z3.Not(f[p]), P[p][u])
                    ))
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


def solve_dgfem_materialization_eq_using_z3(dfg: DataFlowGraph):
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

    print("# einsum_nodes =", len(einsum_nodes))
    print("# indirection nodes =", len({k for k, v in c.items() if v == 0}))
    print("Started assembling...")

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


def get_arrays_to_materialize(dfg: DataFlowGraph) -> frozenset[pt.Array]:
    solve_dgfem_materialization_eq_using_z3(dfg)
    _ = 1 / 0
    raise NotImplementedError


def materialize_for_dgfem_opt(
    expr: pt.transform.ArrayOrNamesTc,
) -> pt.transform.ArrayOrNamesTc:
    materialized_arrays = get_arrays_to_materialize(get_dataflow_graph(expr))

    def materialize_if_needed(
        expr: pt.transform.ArrayOrNames,
    ) -> pt.transform.ArrayOrNames:
        if expr in materialized_arrays:
            return expr.tagged(pt.tags.ImplStored())
        else:
            return expr

    return pt.transform.map_and_copy(expr, materialize_if_needed)
