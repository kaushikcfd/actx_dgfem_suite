from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeVar,
    cast,
    overload,
)

import loopy as lp
import numpy as np
import opt_einsum  # pyright: ignore[reportMissingTypeStubs]
import pyopencl as cl
import pyopencl.tools as cl_tools
import pytato as pt
from constantdict import constantdict
from pymbolic.typing import Integer
from pytato.transform import CachedWalkMapper
from typing_extensions import override

from actx_dgfem_suite.arraycontext import DGFEMOptimizerArrayContext

if TYPE_CHECKING:
    import pyopencl.array as cl_array
    from pytato.array import ArrayOrScalar
    from pytato.loopy import LoopyCall

# {{{ actx to get the kernel with loop fusion, contraction


IRT = TypeVar("IRT", lp.TranslationUnit, "pt.AbstractResultWithNamedArrays")


class OptimizedDGFemIRInspectingActxError[
    IRT: (lp.TranslationUnit, "pt.AbstractResultWithNamedArrays")
](RuntimeError):
    """
    Raised during a compilation of :class:`PytatoDAGGetterActx`.
    """

    def __init__(self, ir: IRT) -> None:
        super().__init__()
        self.ir: IRT = ir


class OptimizedDGFemIRInspectingActx(DGFEMOptimizerArrayContext):
    """
    This is not a conventional arraycontext. During every execution, it simply
    raises a :class:`OptimizedDGFemIRInspectingActxError` with the transformed
    intermediate representation.
    """

    def __init__(
        self,
        ir_to_inspect: Literal["pytato", "loopy"],
        queue: cl.CommandQueue,
        allocator: cl_array.Allocator,
    ) -> None:
        super().__init__(queue=queue, allocator=allocator)
        self.ir_to_inspect: Literal["pytato", "loopy"] = ir_to_inspect

    @override
    def transform_dag(
        self, dag: pt.AbstractResultWithNamedArrays
    ) -> pt.AbstractResultWithNamedArrays:
        """
        Returns a transformed version of *dag*. Sub-classes are supposed to
        override this method to implement context-specific transformations on
        *dag* (most likely to perform domain-specific optimizations). Every
        :mod:`pytato` DAG that is compiled to a GPU-kernel is
        passed through this routine.

        :arg dag: An instance of :class:`pytato.DictOfNamedArrays`
        :returns: A transformed version of *dag*.
        """
        dag = super().transform_dag(dag)
        if self.ir_to_inspect == "pytato" and pt.analysis.get_num_nodes(dag) > 20:
            # FIXME: There has to be a more accurate way to check this.
            raise OptimizedDGFemIRInspectingActxError(dag)
        return dag

    @override
    def transform_loopy_program(
        self, t_unit: lp.TranslationUnit
    ) -> lp.TranslationUnit:
        from actx_dgfem_suite.arraycontext.kennedy_loop_fusion import (
            apply_kennedy_loop_fusion_for_einsum_tags,
        )
        from actx_dgfem_suite.arraycontext.metadata import (
            IncomingEisumTag,
        )
        from actx_dgfem_suite.arraycontext.no_fusion_actx import (
            add_gbarrier_between_disjoint_loop_nests,
        )

        if not any(
            tv.tags_of_type(IncomingEisumTag)
            for tv in t_unit.default_entrypoint.temporary_variables.values()
        ):
            return self.comptime_actx.transform_loopy_program(t_unit)

        t_unit = apply_kennedy_loop_fusion_for_einsum_tags(t_unit)
        t_unit = add_gbarrier_between_disjoint_loop_nests(t_unit)

        if self.ir_to_inspect == "loopy":
            raise OptimizedDGFemIRInspectingActxError(t_unit)
        return t_unit


# }}}

# {{{ count_flops_for_pytato_dag


@dataclass(frozen=True)
class OpCounts:
    """
    Records the operation counts in a :class:`pytato.Array` expression.

    :attr add: Number

    """

    add: ArrayOrScalar
    mult: ArrayOrScalar
    div: ArrayOrScalar
    floor_div: ArrayOrScalar
    comparison: ArrayOrScalar
    bitwise: ArrayOrScalar
    logical: ArrayOrScalar
    where: ArrayOrScalar
    function_calls: constantdict[str, ArrayOrScalar]

    @staticmethod
    def _zeroed_defaults(
        *,
        add: ArrayOrScalar = 0,
        mult: ArrayOrScalar = 0,
        div: ArrayOrScalar = 0,
        floor_div: ArrayOrScalar = 0,
        comparison: ArrayOrScalar = 0,
        bitwise: ArrayOrScalar = 0,
        logical: ArrayOrScalar = 0,
        where: ArrayOrScalar = 0,
        function_calls: constantdict[str, ArrayOrScalar] | None = None,
    ) -> OpCounts:
        if function_calls is None:
            function_calls = constantdict()
        return OpCounts(
            add=add,
            mult=mult,
            div=div,
            floor_div=floor_div,
            comparison=comparison,
            bitwise=bitwise,
            logical=logical,
            where=where,
            function_calls=function_calls,
        )

    @staticmethod
    def from_add(counts: ArrayOrScalar) -> OpCounts:
        return OpCounts._zeroed_defaults(add=counts)

    @staticmethod
    def from_mult(counts: ArrayOrScalar) -> OpCounts:
        return OpCounts._zeroed_defaults(mult=counts)

    @staticmethod
    def from_div(counts: ArrayOrScalar) -> OpCounts:
        return OpCounts._zeroed_defaults(div=counts)

    @staticmethod
    def from_floor_div(counts: ArrayOrScalar) -> OpCounts:
        return OpCounts._zeroed_defaults(floor_div=counts)

    @staticmethod
    def from_bitwise(counts: ArrayOrScalar) -> OpCounts:
        return OpCounts._zeroed_defaults(bitwise=counts)

    @staticmethod
    def from_logical(counts: ArrayOrScalar) -> OpCounts:
        return OpCounts._zeroed_defaults(logical=counts)

    @staticmethod
    def from_comparison(counts: ArrayOrScalar) -> OpCounts:
        return OpCounts._zeroed_defaults(comparison=counts)

    @staticmethod
    def from_where(counts: ArrayOrScalar) -> OpCounts:
        return OpCounts._zeroed_defaults(where=counts)

    @staticmethod
    def from_func(name: str, counts: ArrayOrScalar) -> OpCounts:
        return OpCounts._zeroed_defaults(function_calls=constantdict({name: counts}))

    def __add__(self, other: OpCounts) -> OpCounts:
        all_func_names = frozenset(self.function_calls.keys()) | frozenset(
            other.function_calls.keys()
        )
        return OpCounts(
            add=self.add + other.add,
            mult=self.mult + other.mult,
            div=self.div + other.div,
            floor_div=self.floor_div + other.floor_div,
            comparison=self.comparison + other.comparison,
            bitwise=self.bitwise + other.bitwise,
            logical=self.logical + other.logical,
            where=self.where + other.where,
            function_calls=constantdict(
                {
                    func_name: self.function_calls.get(func_name, 0)
                    + other.function_calls.get(func_name, 0)
                    for func_name in all_func_names
                }
            ),
        )


type FlopCounterResultT = Mapping[np.dtype[Any], OpCounts]


class FlopCounter(CachedWalkMapper[[]]):

    def __init__(self) -> None:
        self.dtype_to_counts: dict[np.dtype[Any], OpCounts] = {}
        super().__init__()

    @override
    def get_cache_key(
        self, expr: pt.transform.ArrayOrNames
    ) -> pt.transform.ArrayOrNames:
        return expr

    def update_dtype_to_counts(self, dtype: np.dtype[Any], count: OpCounts) -> None:
        self.dtype_to_counts[dtype] = (
            self.dtype_to_counts.get(dtype, OpCounts._zeroed_defaults()) + count
        )

    @override
    def map_index_lambda(self, expr: pt.IndexLambda) -> None:
        from pytato.raising import (
            BinaryOp,
            BinaryOpType,
            BroadcastOp,
            C99CallOp,
            LogicalNotOp,
            ReduceOp,
            WhereOp,
            ZerosLikeOp,
            index_lambda_to_high_level_op,
        )

        hlo = index_lambda_to_high_level_op(expr)
        if isinstance(hlo, BinaryOp):
            if isinstance(hlo.x1, pt.Array):
                self.rec(hlo.x1)
            if isinstance(hlo.x2, pt.Array):
                self.rec(hlo.x2)

            if hlo.binary_op in [BinaryOpType.ADD, BinaryOpType.SUB]:
                self.update_dtype_to_counts(expr.dtype, OpCounts.from_add(expr.size))
            elif hlo.binary_op == BinaryOpType.MULT:
                self.update_dtype_to_counts(
                    expr.dtype, OpCounts.from_mult(expr.size)
                )
            elif hlo.binary_op == BinaryOpType.TRUEDIV:
                self.update_dtype_to_counts(expr.dtype, OpCounts.from_div(expr.size))
            elif hlo.binary_op == BinaryOpType.FLOORDIV:
                assert isinstance(hlo.x1, pt.Array) and isinstance(hlo.x2, pt.Array)
                self.update_dtype_to_counts(
                    np.result_type(hlo.x1.dtype, hlo.x2.dtype),
                    OpCounts.from_floor_div(expr.size),
                )
            elif hlo.binary_op in [
                BinaryOpType.BITWISE_AND,
                BinaryOpType.BITWISE_OR,
                BinaryOpType.BITWISE_XOR,
            ]:
                self.update_dtype_to_counts(
                    expr.dtype, OpCounts.from_bitwise(expr.size)
                )
            elif hlo.binary_op in [
                BinaryOpType.LESS,
                BinaryOpType.GREATER,
                BinaryOpType.LESS_EQUAL,
                BinaryOpType.GREATER_EQUAL,
            ]:
                self.update_dtype_to_counts(
                    expr.dtype, OpCounts.from_comparison(expr.size)
                )
            else:
                raise NotImplementedError(
                    f"Unsupported binary op type {hlo.binary_op}."
                )
        elif isinstance(hlo, C99CallOp):
            for arg in hlo.args:
                if isinstance(arg, pt.Array):
                    self.rec(arg)

            self.update_dtype_to_counts(
                expr.dtype, OpCounts.from_func(hlo.function, expr.size)
            )
        elif isinstance(hlo, WhereOp):
            if isinstance(hlo.condition, pt.Array):
                self.rec(hlo.condition)
            if isinstance(hlo.then, pt.Array):
                self.rec(hlo.then)
            if isinstance(hlo.else_, pt.Array):
                self.rec(hlo.else_)

            self.update_dtype_to_counts(expr.dtype, OpCounts.from_where(expr.size))
        elif isinstance(hlo, BroadcastOp):
            self.rec(hlo.x)
        elif isinstance(hlo, LogicalNotOp):
            self.rec(hlo.x)
            self.update_dtype_to_counts(
                hlo.x.dtype, OpCounts.from_logical(expr.size)
            )
        elif isinstance(hlo, ZerosLikeOp):
            # do nothing node.
            pass
        elif isinstance(hlo, ReduceOp):
            from pytato.reductions import (
                AllReductionOperation,
                AnyReductionOperation,
                MaxReductionOperation,
                MinReductionOperation,
                ProductReductionOperation,
                SumReductionOperation,
            )

            self.rec(hlo.x)
            if hlo.axes:
                if isinstance(hlo.op, SumReductionOperation):
                    self.update_dtype_to_counts(
                        hlo.x.dtype, OpCounts.from_add(hlo.x.size)
                    )
                elif isinstance(hlo.op, ProductReductionOperation):
                    self.update_dtype_to_counts(
                        hlo.x.dtype, OpCounts.from_mult(hlo.x.size)
                    )
                elif isinstance(hlo.op, MaxReductionOperation):
                    self.update_dtype_to_counts(
                        hlo.x.dtype, OpCounts.from_func("max", hlo.x.size)
                    )
                elif isinstance(hlo.op, MinReductionOperation):
                    self.update_dtype_to_counts(
                        hlo.x.dtype, OpCounts.from_func("min", hlo.x.size)
                    )
                elif isinstance(
                    hlo.op, (AnyReductionOperation, AllReductionOperation)
                ):
                    self.update_dtype_to_counts(
                        hlo.x.dtype, OpCounts.from_logical(hlo.x.size)
                    )
                else:
                    raise NotImplementedError(type(hlo.op))
        else:
            raise NotImplementedError()

    @override
    def map_einsum(self, expr: pt.Einsum) -> None:
        for arg in expr.args:
            self.rec(arg)

        from pytato.utils import get_einsum_specification

        _, path_info = opt_einsum.contract_path(
            get_einsum_specification(expr), *expr.args, optimize="optimal"
        )
        flop_count = int(path_info.opt_cost)
        # FIXME: Is the FLOP count FMA (I suspect, we are double-counting here.)
        self.update_dtype_to_counts(
            expr.dtype,
            OpCounts._zeroed_defaults(add=flop_count, mult=flop_count),
        )

    @override
    def map_roll(self, expr: pt.Roll) -> None:
        self.rec(expr.array)

    @override
    def map_reshape(self, expr: pt.Reshape) -> None:
        self.rec(expr.array)

    @override
    def map_axis_permutation(self, expr: pt.AxisPermutation) -> None:
        self.rec(expr.array)

    @override
    def map_stack(self, expr: pt.Stack) -> None:
        for ary in expr.arrays:
            self.rec(ary)

    @override
    def map_concatenate(self, expr: pt.Concatenate) -> None:
        for ary in expr.arrays:
            self.rec(ary)

    @override
    def _map_index_base(self, expr: pt.IndexBase) -> None:
        self.rec(expr.array)
        for idx in expr.indices:
            if isinstance(idx, pt.Array):
                self.rec(expr.array)

    @override
    def map_loopy_call(self, expr: LoopyCall) -> None:
        # Use lp.get_op_map to solve this.
        raise NotImplementedError

    @override
    def map_basic_index(self, expr: pt.BasicIndex) -> None:
        return self._map_index_base(expr)

    @override
    def map_contiguous_advanced_index(
        self, expr: pt.AdvancedIndexInContiguousAxes
    ) -> None:
        return self._map_index_base(expr)

    @override
    def map_non_contiguous_advanced_index(
        self, expr: pt.AdvancedIndexInNoncontiguousAxes
    ) -> None:
        return self._map_index_base(expr)

    @override
    def map_placeholder(self, expr: pt.Placeholder) -> None:
        pass

    @override
    def map_size_param(self, expr: pt.SizeParam) -> None:
        pass

    @override
    def map_data_wrapper(self, expr: pt.DataWrapper) -> None:
        pass


def count_flops_for_pytato_dag(
    expr: pt.AbstractResultWithNamedArrays,
) -> constantdict[np.dtype[Any], OpCounts]:
    fc = FlopCounter()
    fc(expr)
    return constantdict(fc.dtype_to_counts)


# }}}


@overload
def _get_ir(
    ir_type: Literal["loopy"], equation: str, dim: int, degree: int
) -> lp.TranslationUnit: ...
@overload
def _get_ir(
    ir_type: Literal["pytato"], equation: str, dim: int, degree: int
) -> pt.AbstractResultWithNamedArrays: ...


@cache
def _get_ir(
    ir_type: Literal["pytato", "loopy"], equation: str, dim: int, degree: int
) -> lp.TranslationUnit | pt.AbstractResultWithNamedArrays:
    from meshmode.dof_array import array_context_for_pickling

    from actx_dgfem_suite.utils import (
        get_benchmark_ref_input_arguments_path,
        get_benchmark_rhs_invoker,
        is_dataclass_array_container,
    )

    rhs_invoker = get_benchmark_rhs_invoker(equation, dim, degree)
    cl_ctx = cl.create_some_context()
    cq = cl.CommandQueue(cl_ctx)
    alloc = cl_tools.MemoryPool(cl_tools.ImmediateAllocator(cq))

    actx = OptimizedDGFemIRInspectingActx(ir_type, cq, alloc)
    rhs_clbl: Callable[..., None] = rhs_invoker(actx)  # pyright: ignore[reportAny]

    with open(
        get_benchmark_ref_input_arguments_path(equation, dim, degree), "rb"
    ) as fp:
        import pickle

        with array_context_for_pickling(actx):
            loaded = cast(
                "tuple[tuple[object, ...], dict[str, object]]", pickle.load(fp)
            )
            np_args, np_kwargs = loaded

    if all(
        (
            is_dataclass_array_container(arg)
            or (
                isinstance(arg, np.ndarray)
                and arg.dtype == "O"
                and all(
                    is_dataclass_array_container(el)  # pyright: ignore[reportAny]
                    for el in arg  # pyright: ignore[reportAny]
                )
            )
            or np.isscalar(arg)
        )
        for arg in np_args
    ) and all(
        is_dataclass_array_container(arg) or np.isscalar(arg)
        for arg in np_kwargs.values()
    ):
        args, kwargs = np_args, np_kwargs
    elif any(is_dataclass_array_container(arg) for arg in np_args) or any(
        is_dataclass_array_container(arg) for arg in np_kwargs.values()
    ):
        raise NotImplementedError("Pickling not implemented for input" " types.")
    else:
        args, kwargs = (
            tuple(
                actx.from_numpy(arg)  # pyright: ignore[reportUnknownMemberType]
                for arg in np_args
            ),
            {
                kw: actx.from_numpy(arg)  # pyright: ignore[reportUnknownMemberType]
                for kw, arg in np_kwargs.items()
            },
        )

    try:
        rhs_clbl(*args, **kwargs)
    except (
        OptimizedDGFemIRInspectingActxError
    ) as e:  # pyright: ignore[reportUnknownVariableType]
        ir = cast("lp.TranslationUnit | pt.AbstractResultWithNamedArrays", e.ir)
        if ir_type == "pytato":
            assert isinstance(ir, pt.AbstractResultWithNamedArrays)
        else:
            assert ir_type == "loopy"
            assert isinstance(ir, lp.TranslationUnit)
        return ir
    else:
        raise RuntimeError("Was expecting a 'OptimizedDGFemIRInspectingActxError'")


@cache
def get_float64_flops(equation: str, dim: int, degree: int) -> int:
    expr = _get_ir("pytato", equation, dim, degree)
    assert isinstance(expr, pt.AbstractResultWithNamedArrays)
    dtype_to_counts = count_flops_for_pytato_dag(expr)

    float64_flops: ArrayOrScalar = 0
    f64 = np.dtype(np.float64)
    c128 = np.dtype(np.complex128)

    fp64_count = dtype_to_counts.get(f64, OpCounts._zeroed_defaults())

    float64_flops += fp64_count.add
    float64_flops += fp64_count.mult
    float64_flops += fp64_count.div
    float64_flops += fp64_count.floor_div
    float64_flops += fp64_count.bitwise
    float64_flops += fp64_count.logical
    float64_flops += fp64_count.where
    for func_name, count in fp64_count.function_calls.items():
        if func_name in ["max", "min", "isnan"]:
            float64_flops += count
        else:
            raise NotImplementedError(f"Flops for func name: {func_name}.")

    c128_count = dtype_to_counts.get(c128, OpCounts._zeroed_defaults())
    float64_flops += 2 * c128_count.add
    float64_flops += 6 * c128_count.mult
    float64_flops += 11 * c128_count.div
    float64_flops += 11 * c128_count.floor_div
    float64_flops += 2 * c128_count.bitwise
    float64_flops += 2 * c128_count.logical
    float64_flops += 2 * c128_count.where
    for func_name, count in c128_count.function_calls.items():
        if func_name == "isnan":
            float64_flops += 2 * count
        elif func_name == "abs":
            float64_flops += 3 * count
        else:
            raise NotImplementedError(f"Flops for func name: {func_name}.")

    assert isinstance(float64_flops, int)
    return float64_flops


@cache
def get_footprint_bytes(equation: str, dim: int, degree: int) -> int:
    from loopy.schedule import CallKernel
    from loopy.schedule.tools import get_subkernel_arg_info
    from pytools import product

    t_unit = _get_ir("loopy", equation, dim, degree)
    assert isinstance(t_unit, lp.TranslationUnit)
    t_unit = lp.linearize(lp.preprocess_program(t_unit))
    knl = t_unit.default_entrypoint
    assert knl.linearization is not None
    subkernel_names = tuple(
        sched_item.kernel_name
        for sched_item in knl.linearization
        if isinstance(sched_item, CallKernel)
    )
    footprint_bytes: Integer = 0
    for subknl_name in subkernel_names:
        subknl_arg_info = get_subkernel_arg_info(knl, subknl_name)
        for tv_name in subknl_arg_info.passed_temporaries:
            tv = knl.temporary_variables[tv_name]
            assert isinstance(
                tv.nbytes, Integer
            ), "Only int shape supported for now."
            footprint_bytes += tv.nbytes
        for arg_name in subknl_arg_info.passed_arg_names:
            arg = knl.arg_dict[arg_name]
            if isinstance(arg, lp.ArrayArg):
                assert arg.shape is not None
                assert arg.dtype is not None
                arg_nbytes = (
                    cast("int", product(cast("tuple[Any, ...]", arg.shape)))
                    * arg.dtype.itemsize
                )
            else:
                assert arg.dtype is not None
                arg_nbytes = arg.dtype.itemsize
            footprint_bytes += cast("Integer", arg_nbytes)

    return cast("int", footprint_bytes)


@cache
def get_roofline_flop_rate(
    equation: str,
    dim: int,
    degree: int,
    roofline_model: str = "libparanumal:global_ai",
    device_name: str | None = None,
) -> float:
    from actx_dgfem_suite.consts import DEV_TO_PEAK_BW, DEV_TO_PEAK_F64_GFLOPS

    if roofline_model == "libparanumal:global_ai":
        import pyopencl as cl

        cl_ctx = cl.create_some_context()
        if device_name is None:
            (device_name,) = {dev.name for dev in cl_ctx.devices}

        nflops = get_float64_flops(equation, dim, degree)
        nbytes = get_footprint_bytes(equation, dim, degree)

        try:
            t_runtime = max(
                ((nflops * 1e-9) / DEV_TO_PEAK_F64_GFLOPS[device_name]),
                ((nbytes * 1e-9) / DEV_TO_PEAK_BW[device_name]),
            )
        except KeyError:
            return np.nan
        else:
            return get_float64_flops(equation, dim, degree) / t_runtime
    else:
        raise NotImplementedError("Unknown roofline model:", roofline_model)
