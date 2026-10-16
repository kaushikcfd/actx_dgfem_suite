from __future__ import annotations

from feinsum.diagnostics import EinsumTunitMatchError

__doc__ = """
A binary for running DG-FEM benchmarks for an array of arraycontexts. Call as
``python -m actx_dgfem_suite.compare_with_sota_array_libs -h`` for a detailed
description on how to run the benchmarks. """

import gc
from time import time

import loopy as lp
import pyopencl as cl
import pyopencl.array as cl_array
import pyopencl.tools as cl_tools
import pytato as pt
from tabulate import tabulate
from typing_extensions import override

from actx_dgfem_suite.arraycontext._loop_nest_utils import get_loop_nest
from actx_dgfem_suite.arraycontext.constants_folder import (
    fold_constants_in_einsum_indirections,
)
from actx_dgfem_suite.arraycontext.deduplicate_by_value import (
    dedup_datawrappers_having_same_value,
)
from actx_dgfem_suite.arraycontext.dgfem_optimizer import DGFEMOptimizerArrayContext
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
)
from actx_dgfem_suite.arraycontext.push_einsum_indices import (
    push_einsum_indices_to_operands,
)
from actx_dgfem_suite.arraycontext.split_iteration_domains import (
    split_iteration_domain_across_work_items,
    split_loop_nest_across_work_items,
)
from actx_dgfem_suite.arraycontext.transpose_consts_in_einsums import (
    transpose_deriv_matrix_in_grad_and_div,
    transpose_lift_matrix_in_facemass,
)
from actx_dgfem_suite.measure import finish_command_queue
from actx_dgfem_suite.perf_analysis import get_float64_flops
from actx_dgfem_suite.rhs_builder import get_rhs


def transform_batched_einsum_loop_nests(
    t_unit: lp.TranslationUnit, cl_device: cl.Device
) -> lp.TranslationUnit:
    import feinsum as fnsm
    import loopy.match as lp_match
    from feinsum.diagnostics import NoFactInDatabaseError

    from actx_dgfem_suite.arraycontext.batched_einsum_loop_transforms import (
        merge_domains_for_potential_loop_nests,
    )

    t_unit = merge_domains_for_potential_loop_nests(t_unit)

    # use feinsum to transform batched einsums or fallback to libparanumal transform
    for loop_nest in sorted(
        {
            insn.within_inames
            for insn in t_unit.default_entrypoint.instructions
            if not isinstance(insn, lp.BarrierInstruction)
        },
        key=sorted,
    ):
        within = lp_match.And(tuple(lp_match.Iname(iname) for iname in loop_nest))
        batched_einsum, _ = fnsm.get_a_matched_einsum(
            t_unit,
            insn_match=within,
            long_dim_length=36,
        )
        try:
            transform = fnsm.retrieve(batched_einsum, cl_device)
        except NoFactInDatabaseError:
            kernel = t_unit.default_entrypoint
            representative_insn = next(
                insn
                for insn in kernel.instructions
                if insn.within_inames == loop_nest
            )
            kernel = split_loop_nest_across_work_items(
                kernel,
                get_loop_nest(kernel, representative_insn),
                cl_device,
            )
            t_unit = t_unit.with_kernel(kernel)
        else:
            t_unit = transform(t_unit, insn_match=within)

    return t_unit


class PhaseAblationArrayContext(DGFEMOptimizerArrayContext):
    """
    An array context that applies the same transformation phases
    as :class:DGFEMOptimizerArrayContext, but avoids the phase
    :attr:`phase_to_skip`. If :attr:`phase_to_avoid` is None, no phase
    is skipped and we get the baseline arraycontext.
    """

    phase_to_skip: int | None

    def __init__(
        self,
        phase_to_skip: int | None,
        queue: cl.CommandQueue,
        allocator: cl_array.Allocator,
    ):
        super().__init__(queue, allocator)
        self.phase_to_skip = phase_to_skip

    @override
    def clone(self) -> PhaseAblationArrayContext:
        return PhaseAblationArrayContext(
            self.phase_to_skip, self.queue, self.allocator
        )

    @override
    def transform_dag(
        self, dag: pt.AbstractResultWithNamedArrays
    ) -> pt.AbstractResultWithNamedArrays:
        assert isinstance(dag, pt.DictOfNamedArrays)
        if all(isinstance(ary.expr, pt.DataWrapper) for ary in dag.values()):
            # compile() also calls this hook on thawed, frozen inputs to infer
            # placeholder metadata. These already-concrete arrays need no
            # DG-FEM transformations (and no materialization).
            return dag

        if self.phase_to_skip != 1:
            dag = pt.rewrite_einsums_with_no_broadcasts(dag)
            dag = apply_distributive_law_to_mass_inverse(dag)
            dag = push_einsum_indices_to_operands(dag)
            dag = fuse_mass_inverses(dag)

        if self.phase_to_skip != 2:
            dag = materialize_for_dgfem_opt(dag, use_tiebreaks=False)
        else:
            # choose a naive materialization policy.
            dag = pt.transform.materialize_with_mpms(dag)

        if self.phase_to_skip != 3:
            dag = pt.push_index_to_materialized_nodes(dag)

        dag = pt.transform.deduplicate_data_wrappers(dag)
        dag = dedup_datawrappers_having_same_value(dag, self)
        dag = fold_constants_in_einsum_indirections(dag, self)
        dag = transpose_lift_matrix_in_facemass(dag, self)
        dag = transpose_deriv_matrix_in_grad_and_div(dag, self)
        dag = pt.transform.deduplicate_data_wrappers(dag)
        dag = dedup_datawrappers_having_same_value(dag, self)
        dag = make_einsum_operands_as_subst(dag)

        return dag

    @override
    def transform_loopy_program(
        self, t_unit: lp.TranslationUnit
    ) -> lp.TranslationUnit:
        from actx_dgfem_suite.arraycontext.disjoint_loop_nest_barriers import (
            add_gbarrier_between_disjoint_loop_nests,
        )

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

        if self.phase_to_skip != 4:
            import feinsum as fnsm
            import loopy.match as lp_match

            try:
                # Only fuse programs that fit the einsum heuristic. In
                # particular, unmatched reductions must not be treated as
                # elementwise operations by the fusion fallback.
                for insn in t_unit.default_entrypoint.instructions:
                    if not isinstance(insn, lp.BarrierInstruction):
                        fnsm.get_a_matched_einsum(
                            t_unit, insn_match=lp_match.Id(insn.id)
                        )
            except EinsumTunitMatchError:
                pass
            else:
                t_unit = apply_kennedy_loop_fusion_for_einsum_tags(t_unit)

        t_unit = add_gbarrier_between_disjoint_loop_nests(t_unit)

        if self.phase_to_skip != 5:
            try:
                t_unit = transform_batched_einsum_loop_nests(
                    t_unit, self.queue.device
                )
            except EinsumTunitMatchError:
                # maybe going with a simple parallelization strategy here...
                t_unit = split_iteration_domain_across_work_items(
                    t_unit, self.queue.device
                )
        else:
            # maybe going with a simple parallelization strategy here...
            t_unit = split_iteration_domain_across_work_items(
                t_unit, self.queue.device
            )

        t_unit = lp.set_options(  # pyright: ignore[reportUnknownMemberType]
            t_unit,
            build_options=["-cl-fast-relaxed-math", "-cl-mad-enable"],
        )
        return t_unit


def _get_flop_rate(
    cl_ctx: cl.Context,
    equation: str,
    dim: int,
    degree: int,
    phase_to_skip: int | None,
) -> float:
    gc.collect()

    cq = cl.CommandQueue(cl_ctx)
    alloc = cl_tools.MemoryPool(cl_tools.ImmediateAllocator(cq))
    actx = PhaseAblationArrayContext(phase_to_skip, cq, alloc)
    rhs, args = get_rhs(equation, actx, dim, degree)  # pyright: ignore[reportAny]
    compiled_rhs = actx.compile(rhs)  # pyright: ignore[reportAny]

    # {{{ warmup rounds

    i_warmup = 0
    t_warmup = 0.0

    while i_warmup < 5 and t_warmup < 0.5:
        finish_command_queue(actx)
        t_start = time()
        compiled_rhs(*args)
        finish_command_queue(actx)
        t_end = time()
        t_warmup += t_end - t_start
        i_warmup += 1

    # }}}

    # {{{ timing rounds

    i_timing = 0
    t_rhs = 0.0

    while i_timing < 50 and t_rhs < 4:
        finish_command_queue(actx)
        t_start = time()
        for _ in range(3):
            compiled_rhs(*args)
        finish_command_queue(actx)
        t_end = time()
        t_rhs += t_end - t_start
        i_timing += 3

    # }}}

    flops = get_float64_flops(equation, dim, degree)
    return (flops * i_timing) / t_rhs


def main(cl_ctx: cl.Context) -> None:
    device_name = cl_ctx.devices[0].name
    if device_name == "NVIDIA TITAN V":
        equation = "wave"
    elif device_name == "NVIDIA H200 NVL":
        equation = "large_wave"
    else:
        equation = "tiny_wave"

    phase_ids = range(1, 6)
    table = [["Function space", *(f"Q_{phase_id}" for phase_id in phase_ids)]]

    for degree in range(1, 5):
        flop_rates = [
            _get_flop_rate(
                cl_ctx,
                equation=equation,
                dim=3,
                degree=degree,
                phase_to_skip=phase_id,
            )
            for phase_id in phase_ids
        ]
        gc.collect()
        table.append(
            [
                f"P_{degree}^{{DG}}",
                *(f"{flop_rate * 1e-9:.1f}" for flop_rate in flop_rates),
            ]
        )

    print("GFLOP/s for the 3D wave equation:")
    print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))


if __name__ == "__main__":
    cl_ctx = cl.create_some_context()
    main(cl_ctx)
