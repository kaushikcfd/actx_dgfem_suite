"""
A :mod:`cupy`-based :class:`~arraycontext.ArrayContext`.

.. autoclass:: CuPyArrayContext
"""

# CuPy does not publish type information that basedpyright can consume.
# pyright: reportAny=false, reportImplicitOverride=false
# pyright: reportMissingImports=false, reportMissingTypeStubs=false
# pyright: reportMissingParameterType=false, reportUnannotatedClassAttribute=false
# pyright: reportUnknownArgumentType=false, reportUnknownLambdaType=false
# pyright: reportUnknownMemberType=false, reportUnknownParameterType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

from functools import partial, reduce
from typing import TYPE_CHECKING, cast, overload

import cupy as cp
import numpy as np
from arraycontext.container import NotAnArrayContainerError, serialize_container
from arraycontext.container.traversal import (
    rec_map_container,
    rec_map_reduce_array_container,
    rec_multimap_array_container,
    rec_multimap_reduce_array_container,
    with_array_context,
)
from arraycontext.context import ArrayContext, CSRMatrix, SparseMatrix
from arraycontext.fake_numpy import (
    BaseFakeNumpyLinalgNamespace,
    BaseFakeNumpyNamespace,
)
from arraycontext.typing import (
    Array,
    ArrayOrContainer,
    ArrayOrContainerOrScalar,
    ArrayOrContainerOrScalarT,
    ArrayOrScalar,
    ScalarLike,
    is_scalar_like,
)
from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import DTypeLike
    from pymbolic import Scalar
    from pytools.tag import Tag, ToTagSetConvertible


_EMPTY_TAG_SET: frozenset[Tag] = frozenset()


class CuPyArrayContext(ArrayContext):
    """
    A :class:`ArrayContext` that uses :class:`cupy.ndarray` instances for its
    base array class and performs all array operations eagerly on the GPU.

    .. note::

        CuPy stores a global configuration state for the currently active CUDA
        device (i.e. :func:`cupy.cuda.Device`). Callers are expected to manage
        the device and stream configuration themselves.
    """

    def __init__(self) -> None:
        super().__init__()
        self.array_types = (cp.ndarray,)

    def _get_fake_numpy_namespace(self):
        return CuPyFakeNumpyNamespace(self)

    def _rec_map_container(
        self,
        func: Callable[[Array], Array],
        array: ArrayOrContainerOrScalar,
        allowed_types: tuple[type, ...] | None = None,
        *,
        default_scalar: ScalarLike | None = None,
    ) -> ArrayOrContainerOrScalar:
        if allowed_types is None:
            allowed_types = self.array_types

        def _wrapper(ary: ArrayOrScalar) -> ArrayOrScalar:
            if isinstance(ary, allowed_types):
                return func(cast("Array", ary))
            elif is_scalar_like(ary):
                if default_scalar is None:
                    return ary
                else:
                    return np.array(ary).dtype.type(default_scalar)
            else:
                raise TypeError(
                    f"{type(self).__name__}.{func.__name__[1:]} invoked with "
                    f"an unsupported array type: got '{type(ary).__name__}', "
                    f"but expected one of {allowed_types}"
                )

        return rec_map_container(_wrapper, array)

    # {{{ ArrayContext interface

    def from_numpy(self, array):
        def _from_numpy(ary):
            return cp.asarray(ary)

        return with_array_context(
            self._rec_map_container(_from_numpy, array, allowed_types=(np.ndarray,)),
            actx=self,
        )

    def to_numpy(self, array):
        def _to_numpy(ary):
            return cp.asnumpy(ary)

        return with_array_context(
            self._rec_map_container(_to_numpy, array), actx=None
        )

    def freeze(self, array: ArrayOrContainerOrScalarT) -> ArrayOrContainerOrScalarT:
        def _freeze(ary):
            return ary

        return with_array_context(self._rec_map_container(_freeze, array), actx=None)

    def thaw(self, array: ArrayOrContainerOrScalarT) -> ArrayOrContainerOrScalarT:
        return with_array_context(array, actx=self)

    def tag(self, tags: ToTagSetConvertible, array):
        # Sorry, not capable.
        return array

    def tag_axis(self, iaxis, tags: ToTagSetConvertible, array):
        return array

    def call_loopy(self, t_unit, **kwargs):
        raise NotImplementedError(
            "Calling loopy on CuPy arrays is not supported. Maybe rewrite"
            " the loopy kernel as numpy-flavored array operations using"
            " ArrayContext.np."
        )

    def einsum(self, spec, *args, arg_names=None, tagged=()):
        return cp.einsum(spec, *args, optimize="optimal")

    @override
    def make_csr_matrix(
        self,
        shape: tuple[int, int],
        elem_values: Array,
        elem_col_indices: Array,
        row_starts: Array,
        *,
        tags: ToTagSetConvertible = _EMPTY_TAG_SET,
        axes: tuple[ToTagSetConvertible, ...] | None = None,
    ) -> CSRMatrix:
        raise NotImplementedError("Sparse matrices aren't yet supported with CuPy.")

    @override
    def sparse_matmul(
        self, x1: SparseMatrix, x2: ArrayOrContainer
    ) -> ArrayOrContainer:
        raise NotImplementedError("Sparse matrices aren't yet supported with CuPy.")

    def clone(self):
        return type(self)()

    # }}}

    # {{{ properties

    @property
    def permits_inplace_modification(self):
        return True

    @property
    def supports_nonscalar_broadcasting(self):
        return True

    @property
    def permits_advanced_indexing(self):
        return True

    # }}}


class CuPyFakeNumpyLinalgNamespace(BaseFakeNumpyLinalgNamespace):
    # Everything is implemented in the base class for now.
    pass


class CuPyFakeNumpyNamespace(BaseFakeNumpyNamespace):
    """
    A :mod:`numpy` mimic for :class:`CuPyArrayContext`.
    """

    _array_context: CuPyArrayContext

    @override
    def _get_fake_numpy_linalg_namespace(self):
        return CuPyFakeNumpyLinalgNamespace(self._array_context)

    def __getattr__(self, name: str):
        return partial(rec_multimap_array_container, getattr(cp, name))

    # {{{ array creation routines

    @override
    def zeros(self, shape: int | tuple[int, ...], dtype: DTypeLike) -> Array:
        return cast("Array", cast("object", cp.zeros(shape=shape, dtype=dtype)))

    @override
    def _full_like_array(
        self,
        ary: Array,
        fill_value: Scalar,
    ) -> Array:
        return cast("Array", cast("object", cp.full_like(ary, fill_value)))

    # }}}

    # {{{ array manipulation routines

    def ravel(self, a: ArrayOrContainerOrScalar, order="C"):
        def inner_ravel(ary: ArrayOrScalar) -> ArrayOrScalar:
            if is_scalar_like(ary):
                return ary
            else:
                assert isinstance(ary, cp.ndarray)
                return cast("Array", cast("object", cp.ravel(ary, order)))

        return rec_map_container(inner_ravel, a)

    def broadcast_to(self, array: ArrayOrContainerOrScalar, shape: tuple[int, ...]):
        def inner_bcast(ary: ArrayOrScalar) -> ArrayOrScalar:
            if is_scalar_like(ary):
                return ary
            else:
                assert isinstance(ary, cp.ndarray)
                return cast("Array", cast("object", cp.broadcast_to(ary, shape)))

        return rec_map_container(inner_bcast, array)

    def concatenate(self, arrays, axis=0):
        return rec_multimap_array_container(cp.concatenate, arrays, axis)

    def stack(self, arrays, axis=0):
        return rec_multimap_array_container(
            lambda *args: cp.stack(args, axis=axis), *arrays
        )

    # }}}

    # {{{ linear algebra

    @override
    def vdot(self, a, b, dtype=None) -> ArrayOrScalar:
        def _rec_vdot(ary1, ary2):
            common_dtype = cp.result_type(ary1, ary2)
            if dtype not in (None, common_dtype):
                raise NotImplementedError(
                    f"{type(self).__name__} cannot take dtype in vdot."
                )

            return cp.vdot(ary1, ary2)

        return cast(
            "ArrayOrScalar",
            rec_multimap_reduce_array_container(sum, _rec_vdot, a, b),
        )

    # }}}

    # {{{ logic functions

    def all(self, a, /):
        return rec_map_reduce_array_container(
            partial(reduce, cp.logical_and), cp.all, a
        )

    def any(self, a, /):
        return rec_map_reduce_array_container(
            partial(reduce, cp.logical_or), cp.any, a
        )

    @override
    def array_equal(
        self,
        a: ArrayOrContainerOrScalar,
        b: ArrayOrContainerOrScalar,
    ) -> Array:
        actx = self._array_context

        # NOTE: not all backends support `bool` properly, so use `int8` instead
        true_ary = actx.from_numpy(np.int8(True))
        false_ary = actx.from_numpy(np.int8(False))

        def rec_equal(x, y):
            if type(x) is not type(y):
                return false_ary

            try:
                serialized_x = serialize_container(x)
                serialized_y = serialize_container(y)
            except NotAnArrayContainerError:
                if x.shape != y.shape:
                    return false_ary
                else:
                    return cp.all(cp.equal(x, y))
            else:
                if len(serialized_x) != len(serialized_y):
                    return false_ary
                return reduce(
                    cp.logical_and,
                    [
                        (true_ary if kx_i == ky_i else false_ary)
                        and rec_equal(x_i, y_i)
                        for (kx_i, x_i), (ky_i, y_i) in zip(
                            serialized_x, serialized_y, strict=True
                        )
                    ],
                    true_ary,
                )

        return cast("Array", rec_equal(a, b))

    # }}}

    # {{{ mathematical functions

    @overload
    def sum(
        self,
        a: ArrayOrContainer,
        axis: int | tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
    ) -> Array: ...
    @overload
    def sum(
        self,
        a: Scalar,
        axis: int | tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
    ) -> Scalar: ...

    @override
    def sum(
        self,
        a: ArrayOrContainerOrScalar,
        axis: int | tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
    ) -> ArrayOrScalar:
        return rec_map_reduce_array_container(
            sum, partial(cp.sum, axis=axis, dtype=dtype), a
        )

    @overload
    def min(
        self,
        a: ArrayOrContainer,
        axis: int | tuple[int, ...] | None = None,
    ) -> Array: ...
    @overload
    def min(
        self,
        a: Scalar,
        axis: int | tuple[int, ...] | None = None,
    ) -> Scalar: ...

    @override
    def min(
        self,
        a: ArrayOrContainerOrScalar,
        axis: int | tuple[int, ...] | None = None,
    ) -> ArrayOrScalar:
        return rec_map_reduce_array_container(
            partial(reduce, cp.minimum), partial(cp.amin, axis=axis), a
        )

    amin = min  # pyright: ignore[reportAssignmentType, reportDeprecated]

    @overload
    def max(
        self,
        a: ArrayOrContainer,
        axis: int | tuple[int, ...] | None = None,
    ) -> Array: ...
    @overload
    def max(
        self,
        a: Scalar,
        axis: int | tuple[int, ...] | None = None,
    ) -> Scalar: ...

    @override
    def max(
        self,
        a: ArrayOrContainerOrScalar,
        axis: int | tuple[int, ...] | None = None,
    ) -> ArrayOrScalar:
        return rec_map_reduce_array_container(
            partial(reduce, cp.maximum), partial(cp.amax, axis=axis), a
        )

    amax = max  # pyright: ignore[reportDeprecated, reportAssignmentType]

    # }}}

    # {{{ sorting, searching and counting

    def where(self, criterion, then, else_):
        return rec_multimap_array_container(cp.where, criterion, then, else_)

    # }}}


# vim: foldmethod=marker
