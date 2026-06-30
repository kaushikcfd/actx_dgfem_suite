__copyright__ = """
Copyright (C) 2020 Andreas Kloeckner (for implementing the wave eqn solver)
Copyright (C) 2021 University of Illinois Board of Trustees
Copyright (C) 2023 Kaushik Kulkarni
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from time import time
from typing import cast

import numpy as np
import pyopencl.tools as cl_tools
from tabulate import tabulate

from actx_dgfem_suite.arraycontext import DGFEMOptimizerArrayContext
from actx_dgfem_suite.equations.wave import get_wave_rhs
from actx_dgfem_suite.measure import finish_command_queue, instantiate_actx_t
from actx_dgfem_suite.utils import (
    get_ndof_for_regular_rect_mesh,
    get_nel_1d_for_regular_rect_mesh,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def stringify_dofs_per_s(dofs_per_s: float) -> str:
    if np.isnan(dofs_per_s):
        return "N/A"
    else:
        return f"{dofs_per_s * 1e-9:.3f}"


def _get_ndofs_list() -> Sequence[int]:
    actx = instantiate_actx_t(DGFEMOptimizerArrayContext)
    if actx.queue.device.name == "NVIDIA TITAN V":
        return [500_000, 1_000_000, 2_000_000, 4_000_000, 6_000_000, 7_000_000]
    if actx.queue.device.name == "NVIDIA H200 NVL":
        return [
            1_000_000,
            5_000_000,
            10_000_000,
            15_000_000,
            20_000_000,
            25_000_000,
            27_000_000,
        ]
    del actx
    raise RuntimeError("Only Titan V, H200 NVL supported.")


def main(
    dim: int,
    degrees: Sequence[int],
):
    ndofs_list = _get_ndofs_list()
    dof_throughput = np.empty([len(degrees), len(ndofs_list)])

    for i_degree, degree in enumerate(degrees):
        for i_ndofs, ndofs in enumerate(ndofs_list):
            import gc

            gc.collect()
            actx = instantiate_actx_t(DGFEMOptimizerArrayContext)
            rhs_clbl, (rhs_args,) = get_wave_rhs(
                actx=actx,
                dim=dim,
                order=degree,
                ndofs=int(ndofs),
            )
            compiled_rhs_clbl = actx.compile(
                rhs_clbl  # pyright: ignore[reportArgumentType]
            )

            # {{{ warmup rounds

            i_warmup = 0
            t_warmup = 0

            compiled_rhs_clbl(rhs_args)
            # Pop `f` to get rid of any associate memoized cl arrays
            vars(compiled_rhs_clbl).pop("f", None)
            del rhs_clbl

            assert isinstance(actx, DGFEMOptimizerArrayContext)
            assert actx.allocator is not None

            if isinstance(actx.allocator, cl_tools.MemoryPool):
                gc.collect()
                actx.queue.finish()
                actx.allocator.free_held()

                memoize_cache = getattr(actx, "_pytools_memoize_in_dict", None)
                if isinstance(memoize_cache, dict):
                    memoize_cache.clear()

                keyed_memoize_cache = getattr(
                    actx, "_pytools_keyed_memoize_in_dict", None
                )
                if isinstance(keyed_memoize_cache, dict):
                    keyed_memoize_cache.clear()
                gc.collect()

            while i_warmup < 5 and t_warmup < 2:
                finish_command_queue(actx)
                t_start = time()
                compiled_rhs_clbl(rhs_args)
                finish_command_queue(actx)
                t_end = time()
                t_warmup += t_end - t_start
                i_warmup += 1

            # }}}

            # {{{ timing rounds

            i_timing = 0
            t_rhs = 0

            while i_timing < 50 and t_rhs < 4:
                finish_command_queue(actx)
                t_start = time()

                for _ in range(3):
                    compiled_rhs_clbl(rhs_args)

                finish_command_queue(actx)
                t_end = time()

                t_rhs += t_end - t_start
                i_timing += 3

            # }}}

            # Multiplying by "(dim + 1)" to account for DOFs for all fields
            actual_ndofs = get_ndof_for_regular_rect_mesh(
                dim,
                degree,
                get_nel_1d_for_regular_rect_mesh(dim, degree, int(ndofs)),
            )

            print("Actual NDOFS = ", actual_ndofs * 1e-6)
            dof_throughput[i_degree, i_ndofs] = ((actual_ndofs) * (dim + 1)) / (
                t_rhs / i_timing
            )
            print(f"Avg time = {(1e3 * t_rhs / i_timing)}ms")

            del rhs_args
            del compiled_rhs_clbl
            del actx
            gc.collect()

    print(f"GDOFS/s for {dim}D-wave for:")
    table: list[list[str]] = []
    for i_degree, degree in enumerate(degrees):
        table.append(
            [
                f"P{degree}",
                *[
                    stringify_dofs_per_s(
                        cast("float", dof_throughput[i_degree, i_ndofs])
                    )
                    for i_ndofs, _ in enumerate(ndofs_list)
                ],
            ]
        )
    print(
        tabulate(
            table,
            tablefmt="fancy_grid",
            headers=[""] + [f"{ndofs / 1e6}M" for ndofs in ndofs_list],
        )
    )


@dataclass(frozen=True)
class CLIArgs:
    dim: int
    degrees: str


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python wave_dof_scaling.py",
        description="Obtain DOF-throughput scaling for different problem sizes of a"
        "wave equation solver.",
    )

    parser.add_argument(
        "--dim",
        metavar="D",
        type=int,
        help=(
            "An integer representing the"
            " topological dimensions to run the problems on"
            " (for ex. 3 to run 3D versions of the"
            " problem)"
        ),
        required=True,
    )
    parser.add_argument(
        "--degrees",
        metavar="G",
        type=str,
        help=(
            "comma separated integers representing the"
            " polynomial degree of the discretizing function"
            " spaces to run the problems on (for ex. 1,2,3"
            " to run using P1,P2,P3 function spaces)"
        ),
        required=True,
    )

    args = CLIArgs(**vars(parser.parse_args()))  # pyright: ignore[reportAny]
    main(
        dim=args.dim,
        degrees=[int(k.strip()) for k in args.degrees.split(",")],
    )
