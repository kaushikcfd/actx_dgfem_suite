from typing import TYPE_CHECKING, cast

import pytato as pt
from pytato.array import NormalizedSlice, ShapeComponent
from pytato.scalar_expr import INT_CLASSES
from pytato.transform import (
    ArrayOrNamesTc,
    CopyMapper,
    _verify_is_array,
)

if TYPE_CHECKING:
    from pymbolic.typing import Integer


def _is_slice_trivial(slice_: NormalizedSlice, axis_len: ShapeComponent) -> bool:
    """
    Return *True* only if *slice_* is equivalent to the trivial slice i.e.
    traverses an axis of length *axis_len* in unit steps.
    """
    from pytato.utils import are_shape_components_equal

    return (
        are_shape_components_equal(slice_.start, 0)
        and are_shape_components_equal(slice_.stop, axis_len)
        and slice_.step == 1
    )


class EinsumIndexPusher(CopyMapper):
    def map_basic_index(self, expr: pt.BasicIndex) -> pt.Array:
        if isinstance(expr.array, pt.Einsum):
            from pytato.array import EinsumElementwiseAxis

            descr_to_slice: dict[EinsumElementwiseAxis, slice | Integer] = {
                EinsumElementwiseAxis(dim): (
                    idx
                    if not isinstance(idx, NormalizedSlice)
                    else slice(idx.start, idx.stop, idx.step)
                )
                for dim, (idx, axis_len) in enumerate(
                    zip(expr.indices, expr.array.shape, strict=True)
                )
                if (
                    isinstance(idx, NormalizedSlice)
                    and not _is_slice_trivial(idx, axis_len)
                )
                or isinstance(idx, INT_CLASSES)
            }
            new_einsum_operands = []
            for ary, access_descrs in zip(
                expr.array.args, expr.array.access_descriptors, strict=True
            ):
                slices: Integer | slice = []
                atleast_one_non_trivial_slice = False
                for access_descr in access_descrs:
                    try:
                        slices.append(descr_to_slice[access_descr])
                    except KeyError:
                        slices.append(slice(None))
                    else:
                        atleast_one_non_trivial_slice = True

                new_einsum_operands.append(
                    _verify_is_array(self.rec(ary[*slices]))
                    if atleast_one_non_trivial_slice
                    else _verify_is_array(self.rec(ary))
                )

            from pytato.utils import get_einsum_specification

            new_subscript = get_einsum_specification(expr.array)
            _, output_str = new_subscript.split("->")
            idxs_to_vanish = {
                out_idx
                for dim, out_idx in enumerate(output_str.strip())
                if isinstance(
                    descr_to_slice.get(EinsumElementwiseAxis(dim)), INT_CLASSES
                )
            }

            return pt.einsum(
                "".join([idx for idx in new_subscript if idx not in idxs_to_vanish]),
                *new_einsum_operands,
            ).copy(redn_axis_to_redn_descr=expr.array.redn_axis_to_redn_descr)

        else:
            return _verify_is_array(super().map_basic_index(expr))


def push_einsum_indices_to_operands(expr: ArrayOrNamesTc) -> ArrayOrNamesTc:
    """
    Returns a transformed version of *expr* where any expression matching
    ``einsum(str, A1, A2, .., AN)[idx]`` is transformed as ``einsum(str,
    A1[idx1'], A2[idx2'], ...)``, where "idx1'" is chosen so that the
    transformation preserves value correctness.

    .. testsetup::

        >>> from actx_dgfem_suite.arraycontext import push_einsum_indices_to_operands

    .. doctest::

        >>> import pytato as pt
        >>> A = pt.make_placeholder("A", (10, 4), np.float64)
        >>> B = pt.make_placeholder("B", (4, 10), np.float64)
        >>> y = (A @ B)[0]
        >>> push_einsum_indices_to_operands((A@B)[0]) == A[0] @ B
        True
        >>> push_einsum_indices_to_operands((A@B)[:,0]) == A @ B[:, 0]
        True
    """
    mapper = EinsumIndexPusher()
    return cast("ArrayOrNamesTc", mapper(expr))
