import hashlib
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytato as pt
from arraycontext import ArrayContext

if TYPE_CHECKING:
    from pytato.array import ShapeType


class ValueDeduper(pt.transform.CopyMapper):
    def __init__(self, actx: ArrayContext) -> None:
        super().__init__()
        self.value_cache: dict[
            tuple[ShapeType, np.dtype[Any], str], list[pt.DataWrapper]
        ] = {}
        self.actx = actx

    def map_data_wrapper(self, expr: pt.DataWrapper) -> pt.DataWrapper:
        expr_np = self.actx.to_numpy(expr)
        hash_key = (
            expr.shape,
            expr.dtype,
            hashlib.sha256(expr_np.tobytes()).hexdigest(),
        )
        try:
            collision_arys = self.value_cache[hash_key]
        except KeyError:
            self.value_cache[hash_key] = [expr]
            return expr
        else:
            for collision_ary in collision_arys:
                collision_ary_np = self.actx.to_numpy(collision_ary)
                if np.all(collision_ary_np == expr_np):
                    return collision_ary
            self.value_cache[hash_key].append(expr)
            return expr


def dedup_datawrappers_having_same_value(
    dag: pt.transform.ArrayOrNamesTc,
    comptime_actx: ArrayContext
) -> pt.transform.ArrayOrNamesTc:
    mapper = ValueDeduper(comptime_actx)
    return cast("pt.transform.ArrayOrNamesTc", mapper(dag))
