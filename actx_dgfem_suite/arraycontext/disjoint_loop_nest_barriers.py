import dataclasses as dc
import itertools

import loopy as lp
import loopy.match as lp_match
from typing_extensions import override

from actx_dgfem_suite.arraycontext._loop_nest_utils import LoopNest, get_loop_nest


@dc.dataclass(frozen=True)
class InsnIds(lp_match.MatchExpressionBase):
    insn_ids_to_match: frozenset[str]

    @override
    def __call__(self, kernel: lp.LoopKernel, matchable: lp_match.Matchable) -> bool:
        return matchable.id in self.insn_ids_to_match


def _get_call_kernel_insn_ids(kernel: lp.LoopKernel) -> tuple[frozenset[str], ...]:
    """
    Returns a sequence of collection of instruction ids where each entry in the
    sequence corresponds to the instructions in a call-kernel to launch.

    In this heuristic we simply draw kernel boundaries such that instruction
    belonging to disjoint loop-nest pairs are executed in different call kernels.

    .. note::

        We require that every statement in *kernel* is nested within a perfect loop
        nest.
    """
    from pytools.graph import compute_topological_order

    loop_nest_dep_graph: dict[LoopNest, set[LoopNest]] = {
        get_loop_nest(kernel, insn): set() for insn in kernel.instructions
    }

    for insn in kernel.instructions:
        insn_loop_nest = get_loop_nest(kernel, insn)
        for dep_id in insn.depends_on:
            dep_loop_nest = get_loop_nest(kernel, kernel.id_to_insn[dep_id])
            if insn_loop_nest != dep_loop_nest:
                loop_nest_dep_graph[dep_loop_nest].add(insn_loop_nest)

    toposorted_loop_nests: list[LoopNest] = compute_topological_order(
        loop_nest_dep_graph, lambda loop_nest: sorted(loop_nest.inames)
    )

    return tuple(loop_nest.insns_in_loop_nest for loop_nest in toposorted_loop_nests)


def add_gbarrier_between_disjoint_loop_nests(
    t_unit: lp.TranslationUnit,
) -> lp.TranslationUnit:
    kernel = t_unit.default_entrypoint
    ing = kernel.get_instruction_id_generator()

    call_kernel_insn_ids = _get_call_kernel_insn_ids(kernel)
    gbarrier_ids: list[str] = []

    for ibarrier, (insns_before, insns_after) in enumerate(
        itertools.pairwise(call_kernel_insn_ids)
    ):
        id_based_on = ing(f"_actx_gbarrier_{ibarrier}")
        kernel = lp.add_barrier(
            kernel,
            insn_before=InsnIds(insns_before),
            insn_after=InsnIds(insns_after),
            id_based_on=id_based_on,
            within_inames=frozenset(),
        )
        assert id_based_on in kernel.id_to_insn
        gbarrier_ids.append(id_based_on)

    for pred_gbarrier, succ_gbarrier in itertools.pairwise(gbarrier_ids):
        kernel = lp.add_dependency(kernel, lp_match.Id(succ_gbarrier), pred_gbarrier)

    return t_unit.with_kernel(kernel)
