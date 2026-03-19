import numpy as np
import pytato as pt
from arraycontext import PytatoPyOpenCLArrayContext
from pytato.array import ArrayOrScalar, NormalizedSlice
from pytools import memoize_on_first_arg


def _can_be_folded(
    expr: pt.Array, input_base_getter: pt.transform.InputGatherer
) -> bool:
    return all(
        not isinstance(input_base, pt.Placeholder)
        for input_base in input_base_getter(expr)
    )


@memoize_on_first_arg
def memoized_ravel(actx: PytatoPyOpenCLArrayContext, ary: pt.Array) -> pt.Array:
    assert isinstance(ary, pt.Array)
    return ary.reshape(-1)


def _fold_constant_einsum_indirection_args(
    expr: pt.transform.ArrayOrNames,
    *,
    actx: PytatoPyOpenCLArrayContext,
    input_base_getter: pt.transform.InputGatherer,
) -> pt.transform.ArrayOrNames:
    if isinstance(expr, pt.Einsum):
        return expr.replace_if_different(
            args=tuple(  # pyright: ignore[reportUnknownArgumentType]
                (  # pyright: ignore[reportUnknownArgumentType]
                    actx.freeze_thaw(  # pyright: ignore[reportUnknownMemberType]
                        arg
                    ).without_tags(pt.tags.ImplStored())
                    if _can_be_folded(arg, input_base_getter)
                    else arg
                )
                for arg in expr.args
                if not isinstance(arg, pt.DataWrapper)
            )
        )
    elif isinstance(
        expr, pt.AdvancedIndexInContiguousAxes | pt.AdvancedIndexInNoncontiguousAxes
    ):
        if all(
            isinstance(idx_ary, pt.DataWrapper)
            for idx_ary in expr.indices
            if isinstance(idx_ary, pt.Array)
        ):
            return expr
        if isinstance(expr, pt.AdvancedIndexInNoncontiguousAxes):
            # TODO. (not sure if this is needed.)
            raise NotImplementedError

        assert not any(isinstance(idx, NormalizedSlice) for idx in expr.indices)
        stride: ArrayOrScalar = 1
        raveled_idx: ArrayOrScalar = 0

        for axis_len, idx in zip(
            expr.array.shape[::-1], expr.indices[::-1], strict=True
        ):
            assert isinstance(
                idx, (pt.Array, int, np.integer)
            ), "not implemented for other cases."
            raveled_idx = raveled_idx + stride * idx
            stride *= axis_len

        thawed_idx = actx.freeze_thaw(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
            raveled_idx
        )
        assert isinstance(thawed_idx, pt.DataWrapper)

        return memoized_ravel(actx, expr.array)[
            thawed_idx.tagged(pt.tags.AssumeNonNegative()).without_tags(
                pt.tags.ImplStored()
            )
        ]
    else:
        return expr


def fold_constants_in_einsum_indirections(
    expr: pt.transform.ArrayOrNamesTc, comptime_actx: PytatoPyOpenCLArrayContext
) -> pt.transform.ArrayOrNamesTc:
    input_base_gatter = pt.transform.InputGatherer()

    return pt.transform.map_and_copy(
        expr,
        lambda expr: _fold_constant_einsum_indirection_args(
            expr, actx=comptime_actx, input_base_getter=input_base_gatter
        ),
    )
