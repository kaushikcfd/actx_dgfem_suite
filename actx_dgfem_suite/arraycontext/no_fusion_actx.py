import dataclasses as dc
import itertools
from collections.abc import Mapping
from typing import cast

import loopy as lp
import loopy.match as lp_match
import pyopencl as cl
import pytato as pt
from arraycontext import PytatoPyOpenCLArrayContext
from typing_extensions import override

# {{{ _LoopNest class definition


@dc.dataclass(frozen=True, eq=True)
class _LoopNest:
    inames: frozenset[str]
    insns_in_loop_nest: frozenset[str]


def _is_a_perfect_loop_nest(kernel: lp.LoopKernel, inames: frozenset[str]) -> bool:
    try:
        template_iname = next(iter(inames))
    except StopIteration:
        return True
    else:
        insn_ids_in_template_iname = kernel.iname_to_insns()[template_iname]
        return all(
            kernel.iname_to_insns()[iname] == insn_ids_in_template_iname
            for iname in inames
        )


def _get_loop_nest(kernel: lp.LoopKernel, insn: lp.InstructionBase) -> _LoopNest:
    assert _is_a_perfect_loop_nest(kernel, insn.within_inames)
    if insn.within_inames:
        any_iname_in_nest, *_other_inames = insn.within_inames
        return _LoopNest(
            insn.within_inames, frozenset(kernel.iname_to_insns()[any_iname_in_nest])
        )
    else:
        if insn.reduction_inames():
            # TODO: Avoid O(N^2) complexity (typically there aren't long
            # kernels with "reduce-to-scalar" operations, but this might bite
            # in the future)
            insn_ids = frozenset(
                {
                    insn_.id
                    for insn_ in kernel.instructions
                    if insn_.reduction_inames() == insn.reduction_inames()
                }
            )
            return _LoopNest(frozenset(), insn_ids)
        else:
            # we treat a loop nest with 0-depth in a special manner by putting
            # each such instruction into a separate loop nest.
            return _LoopNest(frozenset(), frozenset([insn.id]))


# }}}


# {{{ split_iteration_domain_across_work_items


def get_iname_length(kernel: lp.LoopKernel, iname: str) -> float | int:
    from loopy.isl_helpers import static_max_of_pw_aff

    max_domain_size = static_max_of_pw_aff(
        kernel.get_iname_bounds(iname).size, constants_only=False
    ).max_val()
    if max_domain_size.is_infty():
        import math

        return math.inf
    else:
        return max_domain_size.to_python()


def _get_iname_pos_from_loop_nest(
    kernel: lp.LoopKernel, loop_nest: _LoopNest
) -> Mapping[str, int]:
    import pymbolic.primitives as prim

    iname_orders: set[tuple[str, ...]] = set()

    for insn_id in loop_nest.insns_in_loop_nest:
        insn = kernel.id_to_insn[insn_id]
        if isinstance(insn, lp.Assignment):
            if isinstance(insn.assignee, prim.Subscript):
                iname_orders.add(
                    tuple(
                        cast("prim.Variable", idx).name
                        for idx in insn.assignee.index_tuple
                    )
                )
        elif isinstance(insn, lp.CallInstruction):
            # must be a callable kernel, don't touch.
            pass
        elif isinstance(insn, (lp.BarrierInstruction, lp.NoOpInstruction)):
            pass
        else:
            raise NotImplementedError(type(insn))

    if len(iname_orders) != 1:
        raise RuntimeError(
            "split_iteration_domain failed by receiving a"
            " kernel not belonging to the expected grammar or"
            " kernels."
        )

    (iname_order,) = iname_orders
    return {iname: i for i, iname in enumerate(iname_order)}


def _split_loop_nest_across_work_items(
    kernel: lp.LoopKernel,
    loop_nest: _LoopNest,
    iname_to_length: Mapping[str, float | int],
    cl_device: cl.Device,
) -> lp.LoopKernel:

    ngroups = cl_device.max_compute_units * 4  # '4' to overfill the device
    l_one_size = 4
    l_zero_size = 16

    if len(loop_nest.inames) == 0:
        pass
    elif len(loop_nest.inames) == 1:
        (iname,) = loop_nest.inames
        kernel = lp.split_iname(kernel, iname, ngroups * l_zero_size * l_one_size)
        kernel = lp.split_iname(
            kernel, f"{iname}_inner", l_zero_size, inner_tag="l.0"
        )
        kernel = lp.split_iname(
            kernel,
            f"{iname}_inner_outer",
            l_one_size,
            inner_tag="l.1",
            outer_tag="g.0",
        )
    else:
        iname_pos_in_assignee = _get_iname_pos_from_loop_nest(kernel, loop_nest)

        # Pick the loop with largest loop count. In case of ties, look at the
        # iname position in the assignee and pick the iname indexing over
        # leading axis for the work-group hardware iname.
        sorted_inames = sorted(
            loop_nest.inames,
            key=lambda iname: (
                iname_to_length[iname],
                -iname_pos_in_assignee[iname],
            ),
        )
        smaller_loop, bigger_loop = sorted_inames[-2], sorted_inames[-1]

        kernel = lp.split_iname(kernel, f"{bigger_loop}", l_one_size * ngroups)
        kernel = lp.split_iname(
            kernel,
            f"{bigger_loop}_inner",
            l_one_size,
            inner_tag="l.1",
            outer_tag="g.0",
        )
        kernel = lp.split_iname(kernel, smaller_loop, l_zero_size, inner_tag="l.0")

    return kernel


@lp.for_each_kernel
def split_iteration_domain_across_work_items(
    kernel: lp.LoopKernel,
    cl_device: cl.Device,
) -> lp.LoopKernel:

    insn_id_to_loop_nest: Mapping[str, _LoopNest] = {
        insn.id: _get_loop_nest(kernel, insn) for insn in kernel.instructions
    }
    iname_to_length = {
        iname: get_iname_length(kernel, iname) for iname in kernel.all_inames()
    }

    all_loop_nests = frozenset(insn_id_to_loop_nest.values())

    for loop_nest in all_loop_nests:
        kernel = _split_loop_nest_across_work_items(
            kernel, loop_nest, iname_to_length, cl_device
        )

    return kernel


# }}}

# {{{ add_gbarrier_between_disjoint_loop_nests


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

    loop_nest_dep_graph: dict[_LoopNest, set[_LoopNest]] = {
        _get_loop_nest(kernel, insn): set() for insn in kernel.instructions
    }

    for insn in kernel.instructions:
        insn_loop_nest = _get_loop_nest(kernel, insn)
        for dep_id in insn.depends_on:
            dep_loop_nest = _get_loop_nest(kernel, kernel.id_to_insn[dep_id])
            if insn_loop_nest != dep_loop_nest:
                loop_nest_dep_graph[dep_loop_nest].add(insn_loop_nest)

    # TODO: pass 'key' to compute_topological_order to ensure deterministic result
    toposorted_loop_nests: list[_LoopNest] = compute_topological_order(
        loop_nest_dep_graph
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


# }}}


class NoFusionPytatoPyOpenCLActx(PytatoPyOpenCLArrayContext):
    @override
    def transform_dag(
        self, dag: pt.AbstractResultWithNamedArrays
    ) -> pt.AbstractResultWithNamedArrays:
        dag = pt.transform.deduplicate(dag)
        dag = pt.transform.deduplicate_data_wrappers(dag)

        return pt.transform.map_and_copy(
            dag,
            lambda expr: (
                expr.tagged(pt.tags.ImplStored())
                if isinstance(expr, pt.Array)
                else expr
            ),
        )

    @override
    def transform_loopy_program(
        self, t_unit: lp.TranslationUnit
    ) -> lp.TranslationUnit:
        t_unit = split_iteration_domain_across_work_items(t_unit, self.queue.device)
        return add_gbarrier_between_disjoint_loop_nests(t_unit)
