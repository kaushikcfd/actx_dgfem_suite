from collections.abc import Mapping
from typing import cast

import loopy as lp
import pyopencl as cl

from actx_dgfem_suite.arraycontext._loop_nest_utils import LoopNest, get_loop_nest


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
    kernel: lp.LoopKernel, loop_nest: LoopNest
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
        elif isinstance(
            insn, (lp.CallInstruction, lp.BarrierInstruction, lp.NoOpInstruction)
        ):
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
    loop_nest: LoopNest,
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

        kernel = lp.split_iname(kernel, bigger_loop, l_one_size * ngroups)
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
    insn_id_to_loop_nest: Mapping[str, LoopNest] = {
        insn.id: get_loop_nest(kernel, insn) for insn in kernel.instructions
    }
    iname_to_length = {
        iname: get_iname_length(kernel, iname) for iname in kernel.all_inames()
    }

    all_loop_nests = frozenset(insn_id_to_loop_nest.values())

    for loop_nest in sorted(all_loop_nests, key=lambda k: sorted(k.inames)):
        kernel = _split_loop_nest_across_work_items(
            kernel, loop_nest, iname_to_length, cl_device
        )

    return kernel
