__copyright__ = """
Copyright (C) 2026 Kaushik Kulkarni
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import hashlib
from typing import TYPE_CHECKING, Any

import numpy as np
import pytato as pt
from arraycontext import ArrayContext
from typing_extensions import override

if TYPE_CHECKING:
    from pytato.array import ShapeType


class ValueDeduper(pt.transform.CopyMapper):
    actx: ArrayContext

    def __init__(self, actx: ArrayContext) -> None:
        super().__init__(err_on_collision=False)
        self.value_cache: dict[
            tuple[ShapeType, np.dtype[Any], str], list[pt.DataWrapper]
        ] = {}
        self.actx = actx

    @override
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
                if np.all(collision_ary_np == expr_np):  # pyright: ignore[reportAny]
                    return collision_ary
            self.value_cache[hash_key].append(expr)
            return expr


def dedup_datawrappers_having_same_value(
    dag: pt.transform.ArrayOrNamesTc, comptime_actx: ArrayContext
) -> pt.transform.ArrayOrNamesTc:
    mapper = ValueDeduper(comptime_actx)
    return pt.transform.deduplicate(mapper(dag))
