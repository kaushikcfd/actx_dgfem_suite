import dataclasses as dc
from collections.abc import Iterable

import pytato as pt
from constantdict import constantdict


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


def get_arrays_to_materialize(dfg: DataFlowGraph) -> frozenset[pt.Array]:
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
