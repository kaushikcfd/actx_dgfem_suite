import dataclasses as dc
from collections.abc import Iterable

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
    nodes: frozenset[pt.Array]


class DataflowBuilder(pt.transform.CachedWalkMapper[[]]):
    def __init__(self) -> None:
        super().__init__()
        self.succs: dict[pt.Array, set[pt.Array]] = {}
        self.preds: dict[pt.Array, set[pt.Array]] = {}
        self.nodes: set[pt.Array] = set()
        self.direct_pred_getter = pt.analysis.DirectPredecessorsGetter()

    def get_cache_key(
        self, expr: pt.transform.ArrayOrNames
    ) -> pt.transform.ArrayOrNames:
        return expr

    def add_edge(self, from_: pt.Array, to: pt.Array) -> None:
        self.succs.setdefault(from_, set()).add(to)
        self.preds.setdefault(to, set()).add(from_)
        self.nodes.update({from_, to})

    def post_visit(self, expr: pt.ArrayOrNames | pt.FunctionDefinition) -> None:
        if isinstance(expr, pt.Array):
            for pred in self.direct_pred_getter(expr):
                if isinstance(pred, pt.Array):
                    self.add_edge(pred, expr)


def get_dataflow_graph(expr: pt.transform.ArrayOrNames) -> DataFlowGraph:
    builder = DataflowBuilder()
    builder(expr)
    return DataFlowGraph(
        succs=constantdict({k: frozenset(v) for k, v in builder.succs.items()}),
        preds=constantdict({k: frozenset(v) for k, v in builder.preds.items()}),
        nodes=frozenset(builder.nodes)
    )


@dc.dataclass(frozen=True)
class MaterializationState:
    materialized_pred: pt.Array | None
    einsum: pt.Array | None
    is_materialized: bool


def solve_dgfem_materialization_eq_using_z3(dfg: DataFlowGraph):
    import z3

    vng = UniqueNameGenerator([""])
    node_to_name = frozenbidict({node: vng("") for node in dfg.nodes})
    V = frozenset(node_to_name.values())
    c = constantdict(
        {name: isinstance(node, pt.Einsum) for node, name in node_to_name.items()}
    )
    preds = constantdict(
        {
            node_to_name[node]: frozenset(
                {
                    node_to_name[node_pred]
                    for node_pred in dfg.preds.get(node, frozenset())
                }
            )
            for node in dfg.nodes
        }
    )
    succs = constantdict(
        {
            node_to_name[node]: frozenset(
                {
                    node_to_name[node_succ]
                    for node_succ in dfg.succs.get(node, frozenset())
                }
            )
            for node in dfg.nodes
        }
    )
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
