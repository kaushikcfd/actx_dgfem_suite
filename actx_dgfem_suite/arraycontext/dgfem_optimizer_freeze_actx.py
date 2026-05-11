from __future__ import annotations

import ast
import dataclasses as dc
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytato as pt
from arraycontext import (
    ArrayContext,
    PyOpenCLArrayContext,
    PytatoPyOpenCLArrayContext,
    with_array_context,
)
from arraycontext.container.traversal import rec_keyed_map_array_container
from arraycontext.impl.pyopencl.taggable_cl_array import (
    TaggableCLArray,
    to_tagged_cl_array,
)
from arraycontext.impl.pytato.utils import (
    _ary_container_key_stringifier,
    _normalize_pt_expr,
    get_cl_axes_from_pt_axes,
)
from pytools import common_prefix
from typing_extensions import override

from actx_dgfem_suite.codegen.pytato_target import (
    ArraycontextProgram,
    generate_arraycontext_code,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import loopy as lp
    import pyopencl.array as cla
    from arraycontext.typing import (
        ArrayOrContainerOrScalarT,
        ArrayOrScalar,
    )
    from pytato.array import DataInterface
    from pytools.obj_array import ObjectArray1D


@dc.dataclass(frozen=True)
class InMemoryArraycontextProgram:
    program: ArraycontextProgram
    source_code: str
    function: Callable[..., ObjectArray1D[cla.Array]]
    output_names: tuple[str, ...]


def compile_in_memory_arraycontext_program(
    expr: pt.DictOfNamedArrays,
    *,
    actx: ArrayContext,
    function_name: str,
) -> InMemoryArraycontextProgram:
    output_names = tuple(sorted(expr))
    program = generate_arraycontext_code(
        expr,
        function_name=function_name,
        actx=actx,
        show_code=False,
    )

    module = ast.Module(
        body=[*program.import_statements, program.function_def],
        type_ignores=[],
    )
    source_code = ast.unparse(ast.fix_missing_locations(module))
    variables_after_execution: dict[str, Any] = {
        "_MODULE_SOURCE_CODE": source_code,
    }
    exec(source_code, variables_after_execution)  # noqa: S102

    return InMemoryArraycontextProgram(
        program=program,
        source_code=source_code,
        function=cast(
            "Callable[..., ObjectArray1D[cla.Array]]",
            variables_after_execution[function_name],
        ),
        output_names=output_names,
    )


def evaluate_in_memory_arraycontext_program(
    program: InMemoryArraycontextProgram,
    *,
    actx: ArrayContext,
    kwargs: Mapping[str, DataInterface],
) -> ObjectArray1D[cla.Array]:
    assert not program.program.numpy_arrays_to_store
    input_kwargs = {
        kw: kwargs[kw] for kw in program.program.argument_names if kw in kwargs
    }
    return program.function(actx, {}, **input_kwargs)


def get_codegen_freeze_function_name(
    arrays: list[pt.Array], default_name: str = "frozen_result"
) -> str:
    from pytato.tags import PrefixNamed

    name_hint_tags: list[PrefixNamed] = []
    for subary in arrays:
        name_hint_tags.extend(subary.tags_of_type(PrefixNamed))

    name_hint = common_prefix([nh.prefix for nh in name_hint_tags])
    return f"frozen_{name_hint}" if name_hint else default_name


class _PyOpenCLFreezeEvalArrayContext(PyOpenCLArrayContext):
    @override
    def transform_loopy_program(
        self, t_unit: lp.TranslationUnit
    ) -> lp.TranslationUnit:  # type: ignore[no-untyped-def]
        from actx_dgfem_suite.arraycontext.split_iteration_domains import (
            split_iteration_domain_across_work_items,
        )

        return split_iteration_domain_across_work_items(t_unit, self.queue.device)


class FreezeDGFEMExpressionArrayContext(PytatoPyOpenCLArrayContext):

    _freeze_prg_cache: dict[  # pyright: ignore[reportIncompatibleVariableOverride]
        pt.AbstractResultWithNamedArrays, InMemoryArraycontextProgram
    ]

    @cached_property
    def freeze_eval_actx(self) -> PyOpenCLArrayContext:
        return _PyOpenCLFreezeEvalArrayContext(self.queue, self.allocator)

    @override
    def compile(self, f: Callable[..., Any]) -> Callable[..., Any]:
        raise RuntimeError(
            f"{type(self).__name__}.compile() is unsupported. "
            "This array context is only meant to evaluate"
            " expressions during freeze()."
        )

    @override
    def freeze(self, array: ArrayOrContainerOrScalarT) -> ArrayOrContainerOrScalarT:
        if np.isscalar(array):
            return array

        import pyopencl.array as cla

        array_as_dict: dict[str, cla.Array | TaggableCLArray | pt.Array] = {}
        key_to_frozen_subary: dict[str, TaggableCLArray] = {}
        key_to_pt_arrays: dict[str, pt.Array] = {}

        def _record_leaf_ary_in_dict(
            key: tuple[object, ...],
            ary: ArrayOrScalar,
        ) -> ArrayOrScalar:
            key_str = "_ary" + _ary_container_key_stringifier(key)
            if not isinstance(ary, cla.Array | TaggableCLArray | pt.Array):
                raise TypeError(f"expected one of array_types, got {type(ary)}")
            array_as_dict[key_str] = ary
            return ary

        rec_keyed_map_array_container(_record_leaf_ary_in_dict, array)

        for key, subary in array_as_dict.items():
            if isinstance(subary, TaggableCLArray):
                key_to_frozen_subary[key] = subary.with_queue(None)
            elif isinstance(subary, self._frozen_array_types):
                key_to_frozen_subary[key] = to_tagged_cl_array(
                    cast("cla.Array", subary).with_queue(None)
                )
            elif isinstance(subary, pt.DataWrapper):
                key_to_frozen_subary[key] = to_tagged_cl_array(
                    cast("cla.Array", subary.data),
                    axes=get_cl_axes_from_pt_axes(subary.axes),
                    tags=subary.tags,
                )
            elif isinstance(subary, pt.Array):
                key_to_pt_arrays[key] = subary
            else:
                raise TypeError(
                    f"{type(self).__name__}.freeze invoked with an unsupported "
                    f"array type: got '{type(subary).__name__}', but expected one "
                    f"of {self.array_types}"
                )

        def _to_frozen(
            key: tuple[object, ...],
            ary: ArrayOrScalar,
        ) -> ArrayOrScalar:
            key_str = "_ary" + _ary_container_key_stringifier(key)
            return key_to_frozen_subary[key_str]

        if not key_to_pt_arrays:
            return with_array_context(  # pyright: ignore[reportAny]
                rec_keyed_map_array_container(_to_frozen, array),
                actx=None,
            )

        dag = pt.transform.deduplicate(
            pt.make_dict_of_named_arrays(key_to_pt_arrays)
        )
        normalized_expr, bound_arguments = _normalize_pt_expr(dag)

        try:
            compiled_prg = self._freeze_prg_cache[normalized_expr]
        except KeyError:
            try:
                transformed_dag, function_name = self._dag_transform_cache[
                    normalized_expr
                ]
            except KeyError:
                transformed_dag = normalized_expr
                function_name = get_codegen_freeze_function_name(
                    list(key_to_pt_arrays.values())
                )
                self._dag_transform_cache[normalized_expr] = (
                    transformed_dag,
                    function_name,
                )

            assert isinstance(transformed_dag, pt.DictOfNamedArrays)
            compiled_prg = compile_in_memory_arraycontext_program(
                transformed_dag,
                actx=self.freeze_eval_actx,
                function_name=function_name,
            )
            self._freeze_prg_cache[normalized_expr] = compiled_prg
        else:
            transformed_dag, _function_name = self._dag_transform_cache[
                normalized_expr
            ]

        result_as_np_obj_array = evaluate_in_memory_arraycontext_program(
            compiled_prg,
            actx=self.freeze_eval_actx,
            kwargs=dict(bound_arguments),
        )

        out_dict = dict(
            zip(compiled_prg.output_names, result_as_np_obj_array, strict=True)
        )

        assert len(set(out_dict) & set(key_to_frozen_subary)) == 0
        key_to_frozen_subary = {
            **key_to_frozen_subary,
            **{
                k: to_tagged_cl_array(
                    (
                        v.with_queue(None)
                        if (v.flags.c_contiguous and v.offset == 0)
                        else self.freeze_eval_actx.call_loopy(
                            self.freeze_eval_actx._get_to_numpy_noncontiguous_copy_kernel(
                                v.dtype, v.ndim
                            ),
                            inp=v,
                            **{
                                f"s{i}": stride // v.dtype.itemsize
                                for i, stride in enumerate(v.strides)
                            },
                        )["out"]
                    ),
                    axes=get_cl_axes_from_pt_axes(transformed_dag[k].expr.axes),
                    tags=transformed_dag[k].expr.tags,
                )
                for k, v in out_dict.items()
            },
        }

        return with_array_context(  # pyright: ignore[reportAny]
            rec_keyed_map_array_container(_to_frozen, array),
            actx=None,
        )
