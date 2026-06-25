__doc__ = """
A binary for running DG-FEM benchmarks for an array of arraycontexts. Call as
``python -m actx_dgfem_suite.run -h`` for a detailed description on how to run
the benchmarks.
"""

import argparse
import dataclasses as dc
import gc
from collections.abc import Sequence
from time import time

import loopy as lp
import numpy as np
import numpy.typing as npt
from arraycontext import (
    ArrayContext,
    EagerJAXArrayContext,
    NumpyArrayContext,
    PytatoJAXArrayContext,
    rec_multimap_array_container,
)
from bidict import bidict
from meshmode.array_context import (
    PyOpenCLArrayContext as BasePyOpenCLArrayContext,
)
from tabulate import tabulate
from typing_extensions import override

from actx_dgfem_suite.arraycontext import DGFEMOptimizerArrayContext
from actx_dgfem_suite.measure import finish_command_queue, instantiate_actx_t
from actx_dgfem_suite.perf_analysis import get_float64_flops, get_roofline_flop_rate
from actx_dgfem_suite.rhs_builder import get_rhs


class PyOpenCLArrayContext(BasePyOpenCLArrayContext):
    @override
    def transform_loopy_program(
        self, t_unit: lp.TranslationUnit
    ) -> lp.TranslationUnit:
        from actx_dgfem_suite.arraycontext.split_iteration_domains import (
            split_iteration_domain_across_work_items,
        )

        return split_iteration_domain_across_work_items(t_unit, self.queue.device)


def _get_actx_t_priority(actx_t: type[ArrayContext]) -> int:
    if issubclass(actx_t, PytatoJAXArrayContext):
        return 10
    else:
        return 1


def stringify_flops(flops: float) -> str:
    if np.isnan(flops):
        return "N/A"
    else:
        return f"{flops * 1e-9:.1f}"


def _get_flop_rate(
    actx_t: type[ArrayContext],
    equation: str,
    dim: int,
    degree: int,
    verify: bool,
) -> float:

    # {{{ target actx pass + optional correctness check

    actx = instantiate_actx_t(actx_t)
    rhs, args = get_rhs(equation, actx, dim, degree)  # pyright: ignore[reportAny]
    compiled_rhs = actx.compile(rhs)  # pyright: ignore[reportAny]

    output = compiled_rhs(*args)  # pyright: ignore[reportAny]
    if verify:
        numpy_actx = NumpyArrayContext()
        ref_rhs, ref_args = get_rhs(  # pyright: ignore[reportAny]
            equation, numpy_actx, dim, degree
        )
        ref_output = ref_rhs(*ref_args)  # pyright: ignore[reportAny]
        np_ref_output = numpy_actx.to_numpy(ref_output)  # pyright: ignore[reportAny]
        del numpy_actx, ref_rhs, ref_args, ref_output
        gc.collect()

        rec_multimap_array_container(
            lambda x, y: np.testing.assert_allclose(  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
                x,  # pyright: ignore[reportUnknownArgumentType]
                y,  # pyright: ignore[reportUnknownArgumentType]
                rtol=1e-5,
                atol=1e-5,
            ),
            np_ref_output,
            actx.to_numpy(output),  # pyright: ignore[reportAny]
        )
    del output

    # }}}

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


def main(
    equations: Sequence[str],
    dims: Sequence[int],
    degrees: Sequence[int],
    actx_ts: Sequence[type[ArrayContext]],
    verify: bool,
):
    flop_rate: npt.NDArray[np.float64] = np.empty(
        [len(actx_ts), len(dims), len(equations), len(degrees)]
    )
    roofline_flop_rate: npt.NDArray[np.float64] = np.empty(
        [len(dims), len(equations), len(degrees)]
    )

    for idim, dim in enumerate(dims):
        for iequation, equation in enumerate(equations):
            for idegree, degree in enumerate(degrees):
                roofline_flop_rate[idim, iequation, idegree] = (
                    get_roofline_flop_rate(equation, dim, degree)
                )

    # sorting `actx_ts` to run JAX related operations at the end as they only
    # free the device memory atexit
    for iactx_t, actx_t in sorted(
        enumerate(actx_ts), key=lambda k: _get_actx_t_priority(k[1])
    ):
        for idim, dim in enumerate(dims):
            for iequation, equation in enumerate(equations):
                for idegree, degree in enumerate(degrees):
                    flop_rate[iactx_t, idim, iequation, idegree] = _get_flop_rate(
                        actx_t, equation, dim, degree, verify=verify
                    )
                    gc.collect()

    for idim, dim in enumerate(dims):
        for iequation, equation in enumerate(equations):
            print(f"GFLOPS/s for {dim}D-{equation}:")
            table = [
                [
                    "",
                    *[
                        _NAME_TO_ACTX_CLASS.inv[
                            actx_t
                        ]  # pyright: ignore[reportArgumentType]
                        for actx_t in actx_ts
                    ],
                    "Roofline",
                ]
            ]
            for idegree, degree in enumerate(degrees):
                table.append(
                    [
                        f"P{degree}",
                        *[
                            stringify_flops(
                                flop_rate[
                                    iactx_t, idim, iequation, idegree
                                ]  # pyright: ignore[reportAny]
                            )
                            for iactx_t, _ in enumerate(actx_ts)
                        ],
                        stringify_flops(
                            roofline_flop_rate[
                                idim, iequation, idegree
                            ]  # pyright: ignore[reportAny]
                        ),
                    ]
                )
            print(tabulate(table, tablefmt="fancy_grid"))


_NAME_TO_ACTX_CLASS = bidict(
    {
        "pyopencl": PyOpenCLArrayContext,
        "jax:nojit": EagerJAXArrayContext,
        "jax:jit": PytatoJAXArrayContext,
        "pytato:dgfem_opt": DGFEMOptimizerArrayContext,
    }
)


@dc.dataclass(frozen=True)
class CLIArgs:
    equations: str
    dims: str
    degrees: str
    actxs: str
    no_verify: bool


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python -m actx_dgfem_suite.run",
        description="Run DG-FEM benchmarks for arraycontexts",
    )

    parser.add_argument(
        "--equations",
        metavar="E",
        type=str,
        help=(
            "comma separated strings representing which"
            " equations to time (for ex. 'wave,euler,maxwell')."
            " Prefix with 'tiny_' to use 1K DOFs per field"
            " (for ex. 'tiny_wave') or 'large_' to use 20M DOFs"
            " per field (for ex. 'large_wave')."
            " The default (unprefixed) uses 4M DOFs per field."
        ),
        required=True,
    )
    parser.add_argument(
        "--dims",
        metavar="D",
        type=str,
        help=(
            "comma separated integers representing the"
            " topological dimensions to run the problems on"
            " (for ex. 2,3 to run 2D and 3D versions of the"
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
    parser.add_argument(
        "--actxs",
        metavar="A",
        type=str,
        help=(
            "comma separated array context names"
            " (for ex. 'pyopencl,jax:jit,pytato:dgfem_opt')"
        ),
        required=True,
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        default=False,
        help="skips value correctness checks",
    )

    args = CLIArgs(**vars(parser.parse_args()))  # pyright: ignore[reportAny]
    main(
        equations=[k.strip() for k in args.equations.split(",")],
        dims=[int(k.strip()) for k in args.dims.split(",")],
        degrees=[int(k.strip()) for k in args.degrees.split(",")],
        actx_ts=[_NAME_TO_ACTX_CLASS[k] for k in args.actxs.split(",")],
        verify=not args.no_verify,
    )
