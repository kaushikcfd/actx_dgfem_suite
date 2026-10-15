__doc__ = """
A binary for running DG-FEM benchmarks for an array of arraycontexts. Call as
``python -m actx_dgfem_suite.compare_with_sota_array_libs -h`` for a detailed description on how to run
the benchmarks.
"""

import gc
from time import time

import loopy as lp
import pyopencl as cl
import pyopencl.array as cl_array
import pyopencl.tools as cl_tools
import pytato as pt
from arraycontext.impl.pytato import PytatoPyOpenCLArrayContext
from tabulate import tabulate
from typing_extensions import override

from actx_dgfem_suite.measure import finish_command_queue
from actx_dgfem_suite.perf_analysis import get_float64_flops
from actx_dgfem_suite.rhs_builder import get_rhs


class PhaseAblationArrayContext(PytatoPyOpenCLArrayContext):
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
    def transform_dag(
        self, dag: pt.AbstractResultWithNamedArrays
    ) -> pt.AbstractResultWithNamedArrays:
        return super().transform_dag(dag)

    @override
    def transform_loopy_program(
        self, t_unit: lp.TranslationUnit
    ) -> lp.TranslationUnit:
        return super().transform_loopy_program(t_unit)


def _get_flop_rate(
    cl_ctx: cl.Context,
    equation: str,
    dim: int,
    degree: int,
    phase_to_skip: int | None,
) -> float:
    cq = cl.CommandQueue(cl_ctx)

    alloc = cl_tools.MemoryPool(cl_tools.ImmediateAllocator(cq))
    actx = PhaseAblationArrayContext(phase_to_skip, cq, alloc)
    rhs, args = get_rhs(equation, actx, dim, degree)  # pyright: ignore[reportAny]
    compiled_rhs = actx.compile(rhs)  # pyright: ignore[reportAny]

    # {{{ warmup rounds

    i_warmup = 0
    t_warmup = 0.0

    while i_warmup < 5 and t_warmup < 2:
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
    phase_ids = range(1, 6)
    table = [["Function space", *(f"Q_{phase_id}" for phase_id in phase_ids)]]

    for degree in range(1, 5):
        flop_rates = [
            _get_flop_rate(
                cl_ctx,
                equation="wave",
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
