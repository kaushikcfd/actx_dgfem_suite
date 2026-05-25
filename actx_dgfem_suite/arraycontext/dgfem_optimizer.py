from typing import cast

import loopy as lp
import numpy as np
import pytato as pt
from arraycontext import PyOpenCLArrayContext, PytatoPyOpenCLArrayContext
from arraycontext.container.traversal import (
    rec_keyed_map_array_container,
    with_array_context,
)
from arraycontext.impl.pyopencl.taggable_cl_array import (
    TaggableCLArray,
    to_tagged_cl_array,
)
from arraycontext.typing import ArrayOrContainerOrScalarT, ArrayOrScalar
from typing_extensions import override

from actx_dgfem_suite.arraycontext.batched_einsum_loop_transforms import (
    transform_batched_einsum_loop_nests,
)
from actx_dgfem_suite.arraycontext.constants_folder import (
    fold_constants_in_einsum_indirections,
)
from actx_dgfem_suite.arraycontext.deduplicate_by_value import (
    dedup_datawrappers_having_same_value,
)
from actx_dgfem_suite.arraycontext.distribute_operands_of_mass_einsum import (
    apply_distributive_law_to_mass_inverse,
)
from actx_dgfem_suite.arraycontext.evaluate_pt_exprs_using_pyopencl import (
    eagerly_evaluate_using_pyopencl,
)
from actx_dgfem_suite.arraycontext.kennedy_loop_fusion import (
    apply_kennedy_loop_fusion_for_einsum_tags,
)
from actx_dgfem_suite.arraycontext.mass_inverse_fuser import fuse_mass_inverses
from actx_dgfem_suite.arraycontext.materialization_policy import (
    make_einsum_operands_as_subst,
    materialize_for_dgfem_opt,
    propagate_einsum_axes_tags,
)
from actx_dgfem_suite.arraycontext.push_einsum_indices import (
    push_einsum_indices_to_operands,
)
from actx_dgfem_suite.arraycontext.transpose_consts_in_einsums import (
    transpose_deriv_matrix_in_grad_and_div,
    transpose_lift_matrix_in_facemass,
)


class _PyOpenCLFreezeEvalArrayContext(PyOpenCLArrayContext):
    @override
    def transform_loopy_program(
        self, t_unit: lp.TranslationUnit
    ) -> lp.TranslationUnit:
        from actx_dgfem_suite.arraycontext.split_iteration_domains import (
            split_iteration_domain_across_work_items,
        )

        return split_iteration_domain_across_work_items(t_unit, self.queue.device)


class DGFEMOptimizerArrayContext(PytatoPyOpenCLArrayContext):
    """
    An :class:`~arraycontext.ArrayContext` tuned for DG-FEM array expressions.
    It fuses the nodes in the expression DAG to the granularity of functional
    batched einsums, where none of the functional operands exhibit reductions.

    See paper (TODO) for details.
    """

    # {{{ eagerly evluate the ops in the DAG using pyopencl

    @property
    def freeze_eval_actx(self) -> PyOpenCLArrayContext:
        return _PyOpenCLFreezeEvalArrayContext(self.queue, self.allocator)

    @override
    def freeze(self, array: ArrayOrContainerOrScalarT) -> ArrayOrContainerOrScalarT:
        if np.isscalar(array):
            return array

        import pyopencl.array as cla
        from arraycontext.impl.pytato.utils import (
            _ary_container_key_stringifier,
            get_cl_axes_from_pt_axes,
        )

        array_as_dict: dict[str, cla.Array | TaggableCLArray | pt.Array] = {}
        key_to_frozen_subary: dict[str, TaggableCLArray] = {}
        key_to_pt_arrays: dict[str, pt.Array] = {}
        pyopencl_actx = self.freeze_eval_actx

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

        dag = pt.make_dict_of_named_arrays(key_to_pt_arrays)
        dag = pt.transform.deduplicate(dag)
        dag = pt.transform.deduplicate_data_wrappers(dag)
        dag = pt.unify_axes_tags(dag)

        out_dict = eagerly_evaluate_using_pyopencl(dag, actx=pyopencl_actx)

        assert len(set(out_dict) & set(key_to_frozen_subary)) == 0
        key_to_frozen_subary = {
            **key_to_frozen_subary,
            **{
                k: to_tagged_cl_array(
                    (
                        v.with_queue(None)
                        if (v.flags.c_contiguous and v.offset == 0)
                        else pyopencl_actx.call_loopy(
                            pyopencl_actx._get_to_numpy_noncontiguous_copy_kernel(
                                v.dtype, v.ndim
                            ),
                            input=v,
                            **{
                                f"s{i}": stride // v.dtype.itemsize
                                for i, stride in enumerate(v.strides)
                            },
                        )["output"]
                    ),
                    axes=get_cl_axes_from_pt_axes(dag[k].expr.axes),
                    tags=dag[k].expr.tags,
                )
                for k, v in out_dict.items()
            },
        }

        return with_array_context(  # pyright: ignore[reportAny]
            rec_keyed_map_array_container(_to_frozen, array),
            actx=None,
        )

    # }}}

    @override
    def transform_dag(
        self, dag: pt.AbstractResultWithNamedArrays
    ) -> pt.AbstractResultWithNamedArrays:
        if pt.analysis.get_num_nodes(dag) < 10:
            # FIXME: This is only for debugging purposes, remove this once
            # everything is finalized.
            return super().transform_dag(dag)

        assert isinstance(dag, pt.DictOfNamedArrays)
        dag = pt.rewrite_einsums_with_no_broadcasts(dag)
        dag = apply_distributive_law_to_mass_inverse(dag)
        dag = push_einsum_indices_to_operands(dag)
        dag = fuse_mass_inverses(dag)
        dag = materialize_for_dgfem_opt(dag)
        dag = pt.push_index_to_materialized_nodes(dag)
        dag = pt.transform.deduplicate_data_wrappers(dag)
        dag = dedup_datawrappers_having_same_value(dag, self)
        dag = fold_constants_in_einsum_indirections(dag, self)
        dag = transpose_lift_matrix_in_facemass(dag, self)
        dag = transpose_deriv_matrix_in_grad_and_div(dag, self)
        dag = pt.transform.deduplicate_data_wrappers(dag)
        dag = dedup_datawrappers_having_same_value(dag, self)
        dag = propagate_einsum_axes_tags(dag)
        dag = make_einsum_operands_as_subst(dag)

        return dag

    @override
    def transform_loopy_program(
        self, t_unit: lp.TranslationUnit
    ) -> lp.TranslationUnit:
        from actx_dgfem_suite.arraycontext.disjoint_loop_nest_barriers import (
            add_gbarrier_between_disjoint_loop_nests,
        )
        from actx_dgfem_suite.arraycontext.metadata import (
            IncomingEisumTag,
        )

        if not any(
            tv.tags_of_type(IncomingEisumTag)
            for tv in t_unit.default_entrypoint.temporary_variables.values()
        ):
            raise NotImplementedError

        # Make offsets as 0. (FIXME: move this to loopy knl invocation)
        # -----------------------------------------------------------------------
        knl = t_unit.default_entrypoint
        knl = knl.copy(
            args=tuple(
                arg.copy(offset=0)  # pyright: ignore[reportUnknownMemberType]
                for arg in knl.args
            )
        )
        t_unit = t_unit.with_kernel(knl)
        del knl

        t_unit = apply_kennedy_loop_fusion_for_einsum_tags(t_unit)
        t_unit = add_gbarrier_between_disjoint_loop_nests(t_unit)
        t_unit = transform_batched_einsum_loop_nests(t_unit, self.queue.device)
        t_unit = lp.set_options(  # pyright: ignore[reportUnknownMemberType]
            t_unit,
            build_options=["-cl-fast-relaxed-math", "-cl-mad-enable"],
        )
        return t_unit
