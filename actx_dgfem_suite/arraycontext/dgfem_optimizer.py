from functools import cached_property

import loopy as lp
import pytato as pt
from arraycontext import PytatoPyOpenCLArrayContext
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
from actx_dgfem_suite.arraycontext.kennedy_loop_fusion import (
    apply_kennedy_loop_fusion_for_einsum_tags,
)
from actx_dgfem_suite.arraycontext.mass_inverse_fuser import fuse_mass_inverses
from actx_dgfem_suite.arraycontext.materialization_policy import (
    make_einsum_operands_as_subst,
    materialize_for_dgfem_opt,
    propagate_einsum_axes_tags,
)
from actx_dgfem_suite.arraycontext.no_fusion_actx import (
    NoFusionPytatoPyOpenCLActx,
)
from actx_dgfem_suite.arraycontext.push_einsum_indices import (
    push_einsum_indices_to_operands,
)


class DGFEMOptimizerArrayContext(PytatoPyOpenCLArrayContext):
    """
    An :class:`~arraycontext.ArrayContext` tuned for DG-FEM array expressions.
    It fuses the nodes in the expression DAG to the granularity of functional
    batched einsums, where none of the functional operands exhibit reductions.

    See paper (TODO) for details.
    """

    @cached_property
    def comptime_actx(self) -> PytatoPyOpenCLArrayContext:
        return NoFusionPytatoPyOpenCLActx(self.queue, self.allocator)

    @override
    def transform_dag(
        self, dag: pt.AbstractResultWithNamedArrays
    ) -> pt.AbstractResultWithNamedArrays:

        if pt.analysis.get_num_nodes(dag) < 10:
            # FIXME: This is only for debugging purposes, remove this once
            # everything is finalized.
            return self.comptime_actx.transform_dag(dag)

        assert isinstance(dag, pt.DictOfNamedArrays)
        dag = apply_distributive_law_to_mass_inverse(dag)
        dag = push_einsum_indices_to_operands(dag)
        dag = fuse_mass_inverses(dag)
        dag = materialize_for_dgfem_opt(dag)
        dag = pt.rewrite_einsums_with_no_broadcasts(dag)
        dag = pt.push_index_to_materialized_nodes(dag)
        dag = fold_constants_in_einsum_indirections(dag, self.comptime_actx)
        dag = pt.transform.deduplicate_data_wrappers(dag)
        dag = dedup_datawrappers_having_same_value(dag, self.comptime_actx)
        dag = propagate_einsum_axes_tags(dag)
        return make_einsum_operands_as_subst(dag)

    @override
    def transform_loopy_program(
        self, t_unit: lp.TranslationUnit
    ) -> lp.TranslationUnit:
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
