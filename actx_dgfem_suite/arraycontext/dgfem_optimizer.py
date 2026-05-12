import loopy as lp
import pyopencl.tools as cl_tools
import pytato as pt
from arraycontext import PytatoPyOpenCLArrayContext
from arraycontext.typing import ArrayOrContainerOrScalarT
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
from actx_dgfem_suite.arraycontext.dgfem_optimizer_freeze_actx import (
    FreezeDGFEMExpressionArrayContext,
)
from actx_dgfem_suite.arraycontext.distribute_operands_of_mass_einsum import (
    apply_distributive_law_to_mass_inverse,
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


class DGFEMOptimizerArrayContext(PytatoPyOpenCLArrayContext):
    """
    An :class:`~arraycontext.ArrayContext` tuned for DG-FEM array expressions.
    It fuses the nodes in the expression DAG to the granularity of functional
    batched einsums, where none of the functional operands exhibit reductions.

    See paper (TODO) for details.
    """

    @property
    def freeze_actx(self) -> FreezeDGFEMExpressionArrayContext:
        return FreezeDGFEMExpressionArrayContext(
            self.queue, cl_tools.MemoryPool(cl_tools.ImmediateAllocator(self.queue))
        )

    # {{{ use freeze_actx to interpret the ops in the DAG

    @override
    def freeze(self, array: ArrayOrContainerOrScalarT) -> ArrayOrContainerOrScalarT:
        return self.freeze_actx.freeze(array)

    # }}}

    @override
    def transform_dag(
        self, dag: pt.AbstractResultWithNamedArrays
    ) -> pt.AbstractResultWithNamedArrays:
        if pt.analysis.get_num_nodes(dag) < 10:
            # FIXME: This is only for debugging purposes, remove this once
            # everything is finalized.
            return self.freeze_actx.transform_dag(dag)

        assert isinstance(dag, pt.DictOfNamedArrays)
        dag = pt.rewrite_einsums_with_no_broadcasts(dag)
        dag = apply_distributive_law_to_mass_inverse(dag)
        dag = push_einsum_indices_to_operands(dag)
        dag = fuse_mass_inverses(dag)
        dag = materialize_for_dgfem_opt(dag)
        dag = pt.push_index_to_materialized_nodes(dag)
        dag = pt.transform.deduplicate_data_wrappers(dag)
        dag = dedup_datawrappers_having_same_value(dag, self.freeze_actx)
        dag = fold_constants_in_einsum_indirections(dag, self.freeze_actx)
        dag = transpose_lift_matrix_in_facemass(dag, self.freeze_actx)
        dag = transpose_deriv_matrix_in_grad_and_div(dag, self.freeze_actx)
        dag = pt.transform.deduplicate_data_wrappers(dag)
        dag = dedup_datawrappers_having_same_value(dag, self.freeze_actx)
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
