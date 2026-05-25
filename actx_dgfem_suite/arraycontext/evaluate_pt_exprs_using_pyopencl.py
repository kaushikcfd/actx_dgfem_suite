from __future__ import annotations

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

import keyword
from collections.abc import Mapping
from typing import TYPE_CHECKING, Never, overload

import loopy as lp
import numpy as np
import pymbolic.mapper
import pyopencl.array as cla
import pytato.reductions as pt_red
from constantdict import constantdict
from pyopencl import clmath
from pytato.array import (
    AbstractResultWithNamedArrays,
    Array,
    ArrayOrScalar,
    IndexLambda,
    NormalizedSlice,
)
from pytato.scalar_expr import (
    INT_CLASSES,
    Reduce,
    ScalarExpression,
    TypeCast,
    substitute,
)
from pytato.transform import ArrayOrNames, CachedMapper
from pytato.transform.lower_to_index_lambda import to_index_lambda
from pytato.utils import are_shapes_equal, get_einsum_specification
from pytools import UniqueNameGenerator

if TYPE_CHECKING:
    import pytato as pt
    from arraycontext import PyOpenCLArrayContext
    from pymbolic.typing import Integer


class PytatoToLoopyExprMapper(pymbolic.mapper.CachedIdentityMapper[[]]):
    _PYTATO_REDUCTION_TO_LOOPY_REDUCTION: Mapping[
        type[pt_red.ReductionOperation], str
    ] = {
        pt_red.SumReductionOperation: "sum",
        pt_red.ProductReductionOperation: "product",
        pt_red.MaxReductionOperation: "max",
        pt_red.MinReductionOperation: "min",
        pt_red.AllReductionOperation: "all",
        pt_red.AnyReductionOperation: "any",
    }

    def map_reduce(
        self,
        expr: Reduce,
    ) -> pymbolic.typing.Expression:
        from loopy.symbolic import Reduction as LoopyReduction

        try:
            loopy_redn_op = self._PYTATO_REDUCTION_TO_LOOPY_REDUCTION[type(expr.op)]
        except KeyError as err:
            raise NotImplementedError(expr.op) from err

        return LoopyReduction(
            loopy_redn_op,
            tuple(expr.bounds),
            self.rec(expr.inner_expr),
        )

    def map_type_cast(
        self,
        expr: TypeCast,
    ) -> pymbolic.typing.Expression:
        return lp.TypeCast(lp.to_loopy_type(expr.dtype), self.rec(expr.inner_expr))


def pt_scalar_expr_to_loopy_expr(
    expr: ScalarExpression,
) -> pymbolic.typing.Expression:
    mapper = PytatoToLoopyExprMapper()
    return mapper(expr)


def get_t_unit_for_index_lambda(expr: IndexLambda) -> lp.TranslationUnit:
    """
    Returns a :class:`loopy.TranslationUnit` that takes the bindings of *expr*
    as inputs and the evaluate array *expr* as output.
    """

    from pymbolic import var
    from pytato.scalar_expr import WalkMapper

    class ReductionBoundsCollector(WalkMapper):
        def __init__(self) -> None:
            self.bounds: dict[
                str, tuple[pymbolic.typing.Expression, pymbolic.typing.Expression]
            ] = {}

        def map_reduce(self, expr: Reduce) -> None:
            self.bounds.update(expr.bounds)
            super().map_reduce(expr)

        def map_type_cast(self, expr: TypeCast) -> None:
            self.rec(expr.inner_expr)

    reduction_bounds_collector = ReductionBoundsCollector()
    reduction_bounds_collector(expr.expr)

    dim_to_bounds = {
        **{f"_{i}": (0, dim) for i, dim in enumerate(expr.shape)},
        **reduction_bounds_collector.bounds,
    }
    all_dims = ", ".join(list(dim_to_bounds))
    bounds = " and ".join(
        [
            f"{lbound} <= {dim} < {ubound}"
            for dim, (lbound, ubound) in list(dim_to_bounds.items())
        ]
    )
    out_var = var("out")[tuple(var(f"_{i}") for i in range(expr.ndim))]

    # FIXME : Need to remove the if condition for null domains
    domain = f"{{ [{all_dims}]: {bounds} }}" if dim_to_bounds else "{:}"

    return lp.make_kernel(
        domains=domain,
        instructions=[
            lp.Assignment(
                out_var,
                pt_scalar_expr_to_loopy_expr(expr.expr),
                within_inames=frozenset({f"_{i}" for i in range(expr.ndim)}),
            )
        ],
        kernel_data=[
            lp.GlobalArg(  # pyright: ignore[reportUnknownMemberType]
                "out", shape=expr.shape, dtype=expr.dtype
            ),
            *[
                lp.GlobalArg(  # pyright: ignore[reportUnknownMemberType]
                    name, shape=bnd.shape, dtype=bnd.dtype, offset=lp.auto
                )
                for name, bnd in sorted(expr.bindings.items())
            ],
        ],
        options=lp.Options(return_dict=True),
        lang_version=(2018, 2),
    )


class PyOpenCLEvaluator(
    CachedMapper[cla.Array | Mapping[str, cla.Array], Never, []]
):
    """
    Eagerly evaluates operations in a :mod:`pytato` expression using
    :class:`arraycontext.PyOpenCLArrayContext`.
    """

    def __init__(
        self,
        actx: PyOpenCLArrayContext,
    ) -> None:
        super().__init__()
        self.actx = actx

    @staticmethod
    def _sanitize_index_lambda_binding_names(expr: IndexLambda) -> IndexLambda:
        from pymbolic import var

        name_gen = UniqueNameGenerator({*keyword.kwlist, "out", *expr.bindings})
        rename_map = {
            name: (
                name_gen(name if name.isidentifier() else "_in")
                if keyword.iskeyword(name)
                or not name.isidentifier()
                or name == "out"
                else name
            )
            for name in expr.bindings
        }

        if all(name == new_name for name, new_name in rename_map.items()):
            return expr

        return expr.copy(
            expr=substitute(
                expr.expr,
                {name: var(new_name) for name, new_name in rename_map.items()},
            ),
            bindings=constantdict(
                {rename_map[name]: bnd for name, bnd in expr.bindings.items()}
            ),
        )

    def _eval_idx_lambda_using_lpy_kernel(self, expr: IndexLambda) -> cla.Array:
        expr = self._sanitize_index_lambda_binding_names(expr)
        rec_bindings = {name: self.rec(bnd) for name, bnd in expr.bindings.items()}
        return self.actx.call_loopy(
            get_t_unit_for_index_lambda(expr), **rec_bindings
        )["out"]

    def map_index_lambda(self, expr: IndexLambda) -> cla.Array:
        from pytato.raising import (
            BinaryOp,
            BinaryOpType,
            BroadcastOp,
            C99CallOp,
            FullOp,
            LogicalNotOp,
            ReduceOp,
            TypeCastOp,
            WhereOp,
            ZerosLikeOp,
            index_lambda_to_high_level_op,
        )

        hlo = index_lambda_to_high_level_op(expr)

        def _rec_ary_or_scalar(arg: ArrayOrScalar) -> ArrayOrScalar | cla.Array:
            if isinstance(arg, Array):
                return self.rec(arg)
            return arg

        def _has_nonscalar_broadcast(*args: ArrayOrScalar) -> bool:
            arrays = [arg for arg in args if isinstance(arg, Array)]
            return any(not are_shapes_equal(ary.shape, expr.shape) for ary in arrays)

        if isinstance(hlo, FullOp):
            if hlo.fill_value == 0:
                return self.actx.np.zeros(expr.shape, expr.dtype)

            result = cla.empty(
                self.actx.queue,
                expr.shape,
                expr.dtype,
                allocator=self.actx.allocator,
            )
            result.fill(hlo.fill_value)
            return result

        if isinstance(hlo, ZerosLikeOp):
            return self.actx.np.zeros(expr.shape, expr.dtype)

        if isinstance(hlo, BinaryOp):
            if _has_nonscalar_broadcast(hlo.x1, hlo.x2):
                return self._eval_idx_lambda_using_lpy_kernel(expr)

            x1 = _rec_ary_or_scalar(hlo.x1)
            x2 = _rec_ary_or_scalar(hlo.x2)

            match hlo.binary_op:
                case BinaryOpType.ADD:
                    return x1 + x2
                case BinaryOpType.SUB:
                    return x1 - x2
                case BinaryOpType.MULT:
                    return x1 * x2
                case BinaryOpType.POWER:
                    return x1**x2
                case BinaryOpType.TRUEDIV:
                    return x1 / x2
                case BinaryOpType.BITWISE_OR:
                    return x1 | x2
                case BinaryOpType.BITWISE_XOR:
                    return x1 ^ x2
                case BinaryOpType.BITWISE_AND:
                    return x1 & x2
                case BinaryOpType.EQUAL:
                    return x1 == x2
                case BinaryOpType.NOT_EQUAL:
                    return x1 != x2
                case BinaryOpType.LESS:
                    return x1 < x2
                case BinaryOpType.LESS_EQUAL:
                    return x1 <= x2
                case BinaryOpType.GREATER:
                    return x1 > x2
                case BinaryOpType.GREATER_EQUAL:
                    return x1 >= x2
                case BinaryOpType.LOGICAL_OR:
                    return cla.logical_or(x1, x2, queue=self.actx.queue)
                case BinaryOpType.LOGICAL_AND:
                    return cla.logical_and(x1, x2, queue=self.actx.queue)
                case BinaryOpType.FLOORDIV | BinaryOpType.MOD:
                    return self._eval_idx_lambda_using_lpy_kernel(expr)
                case _:
                    raise NotImplementedError(hlo.binary_op)

        if isinstance(hlo, C99CallOp):
            # pyopencl.clmath does support broadcasts.
            if _has_nonscalar_broadcast(*hlo.args):
                return self._eval_idx_lambda_using_lpy_kernel(expr)

            clmath_name = hlo.function
            if not hasattr(clmath, clmath_name):
                return self._eval_idx_lambda_using_lpy_kernel(expr)

            clmath_func = getattr(clmath, clmath_name)
            return clmath_func(
                *(_rec_ary_or_scalar(arg) for arg in hlo.args),
                queue=self.actx.queue,
            )

        if isinstance(hlo, WhereOp):
            if _has_nonscalar_broadcast(hlo.condition, hlo.then, hlo.else_):
                return self._eval_idx_lambda_using_lpy_kernel(expr)

            condition = _rec_ary_or_scalar(hlo.condition)
            then = _rec_ary_or_scalar(hlo.then)
            else_ = _rec_ary_or_scalar(hlo.else_)
            if np.result_type(then) != np.result_type(else_):
                return self._eval_idx_lambda_using_lpy_kernel(expr)

            return cla.if_positive(
                condition,
                then,
                else_,
                queue=self.actx.queue,
            )

        if isinstance(hlo, BroadcastOp):
            return self._eval_idx_lambda_using_lpy_kernel(expr)

        if isinstance(hlo, TypeCastOp):
            return self.rec(hlo.x).astype(hlo.dtype)

        if isinstance(hlo, LogicalNotOp):
            return cla.logical_not(self.rec(hlo.x), queue=self.actx.queue)

        if isinstance(hlo, ReduceOp):

            x = self.rec(hlo.x)

            if tuple(sorted(hlo.axes)) != tuple(range(hlo.x.ndim)):
                return self._eval_idx_lambda_using_lpy_kernel(expr)

            if isinstance(hlo.op, pt_red.SumReductionOperation):
                return self.actx.np.sum(x).astype(expr.dtype)
            elif isinstance(hlo.op, pt_red.MaxReductionOperation):
                return self.actx.np.max(x).astype(expr.dtype)
            elif isinstance(hlo.op, pt_red.MinReductionOperation):
                return self.actx.np.min(x).astype(expr.dtype)
            elif isinstance(hlo.op, pt_red.AllReductionOperation):
                return self.actx.np.all(x).astype(expr.dtype)
            elif isinstance(hlo.op, pt_red.AnyReductionOperation):
                return self.actx.np.any(x).astype(expr.dtype)
            else:
                raise NotImplementedError

        raise NotImplementedError(type(hlo))

    def map_placeholder(self, expr: pt.Placeholder) -> cla.Array:
        raise RuntimeError(
            "PyOpenCLEvaluator does handle unbound variables."
            " Bind them as data wrappers before invoking."
        )

    def map_size_param(self, expr: pt.SizeParam) -> cla.Array:
        raise RuntimeError("PyOpenCLEvaluator does not support SizeParams.")

    def map_data_wrapper(self, expr: pt.DataWrapper) -> cla.Array:
        assert isinstance(expr.data, cla.Array)
        cl_ary = expr.data
        if cl_ary.queue is None:
            cl_ary = cl_ary.with_queue(self.actx.queue)
        return cl_ary

    def map_stack(self, expr: pt.Stack) -> cla.Array:
        return cla.stack([self.rec(ary) for ary in expr.arrays])

    def map_concatenate(self, expr: pt.Concatenate) -> cla.Array:
        return cla.concatenate([self.rec(ary) for ary in expr.arrays])

    def map_roll(self, expr: pt.Roll) -> cla.Array:
        # pyopencl does not implement roll
        return self._eval_idx_lambda_using_lpy_kernel(to_index_lambda(expr))

    def map_axis_permutation(self, expr: pt.AxisPermutation) -> cla.Array:
        rec_ary = self.rec(expr.array)
        return rec_ary.transpose(expr.axis_permutation)

    def map_reshape(self, expr: pt.Reshape) -> cla.Array:
        rec_ary = self.rec(expr.array)
        return rec_ary.reshape(expr.newshape, order=expr.order)

    def map_einsum(self, expr: pt.Einsum) -> cla.Array:
        return self.actx.einsum(
            get_einsum_specification(expr), *[self.rec(arg) for arg in expr.args]
        )

    def map_named_array(self, expr: pt.NamedArray) -> cla.Array:
        return self.rec(expr.expr)

    def map_dict_of_named_arrays(self, expr: pt.DictOfNamedArrays) -> cla.Array:
        return constantdict(
            {name: self.rec(ary) for name, ary in expr._data.items()}
        )

    def map_basic_index(self, expr: pt.BasicIndex) -> cla.Array:
        indices: list[slice | Integer] = []
        for idx in expr.indices:
            if isinstance(idx, INT_CLASSES):
                indices.append(idx)
            else:
                assert isinstance(idx, NormalizedSlice)
                assert isinstance(idx.start, INT_CLASSES)
                assert isinstance(idx.stop, INT_CLASSES)
                assert isinstance(idx.step, INT_CLASSES)
                indices.append(slice(idx.start, idx.stop, idx.step))
        return self.rec(expr.array)[tuple(indices)]

    def map_contiguous_advanced_index(
        self, expr: pt.AdvancedIndexInContiguousAxes
    ) -> cla.Array:
        # pyopencl does not support advanced indices
        return self._eval_idx_lambda_using_lpy_kernel(to_index_lambda(expr))

    def map_non_contiguous_advanced_index(
        self, expr: pt.AdvancedIndexInNoncontiguousAxes
    ) -> cla.Array:
        # pyopencl does not support advanced indices
        return self._eval_idx_lambda_using_lpy_kernel(to_index_lambda(expr))

    def map_sparse_matmul(self, expr: pt.SparseMatmul) -> cla.Array:
        # pyopencl does not have sparse matrix support
        return self._eval_idx_lambda_using_lpy_kernel(to_index_lambda(expr))

    def map_csr_matmul(self, expr: pt.CSRMatmul) -> cla.Array:
        # pyopencl does not have sparse matrix support
        return self._eval_idx_lambda_using_lpy_kernel(to_index_lambda(expr))


@overload
def eagerly_evaluate_using_pyopencl(
    expr: Array,
    *,
    actx: PyOpenCLArrayContext,
) -> cla.Array: ...


@overload
def eagerly_evaluate_using_pyopencl(
    expr: AbstractResultWithNamedArrays,
    *,
    actx: PyOpenCLArrayContext,
) -> Mapping[str, cla.Array]: ...


def eagerly_evaluate_using_pyopencl(
    expr: ArrayOrNames,
    *,
    actx: PyOpenCLArrayContext,
) -> cla.Array | Mapping[str, cla.Array]:
    """
    Evaluates the pytato expression, *expr*, using *actx* for every node in the
    graph.

    .. note::

        *expr* cannot contain any :class:`pytato.Placehoder` values since it prevents
        evaluation.
    """
    mapper = PyOpenCLEvaluator(actx)
    return mapper(expr)
