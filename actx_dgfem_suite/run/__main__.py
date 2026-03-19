__doc__ = """
A binary for running DG-FEM benchmarks for an array of arraycontexts. Call as
``python run.py -h`` for a detailed description on how to run the benchmarks.
"""

import argparse
import dataclasses as dc
import datetime
from collections.abc import Sequence

import loopy as lp
import numpy as np
import numpy.typing as npt
import pytz
from arraycontext import ArrayContext, EagerJAXArrayContext, PytatoJAXArrayContext
from bidict import bidict
from meshmode.array_context import (
    PyOpenCLArrayContext as BasePyOpenCLArrayContext,
)
from tabulate import tabulate
from typing_extensions import override

from actx_dgfem_suite.arraycontext import DGFEMOptimizerArrayContext
from actx_dgfem_suite.measure import get_flop_rate
from actx_dgfem_suite.perf_analysis import get_roofline_flop_rate


class PyOpenCLArrayContext(BasePyOpenCLArrayContext):
    @override
    def transform_loopy_program(
        self, t_unit: lp.TranslationUnit
    ) -> lp.TranslationUnit:
        from actx_dgfem_suite.arraycontext.no_fusion_actx import (
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


def main(
    equations: Sequence[str],
    dims: Sequence[int],
    degrees: Sequence[int],
    actx_ts: Sequence[type[ArrayContext]],
):
    flop_rate: npt.NDArray[np.float64] = np.empty(
        [len(actx_ts), len(dims), len(equations), len(degrees)]
    )
    roofline_flop_rate: npt.NDArray[np.float64] = np.empty(
        [len(dims), len(equations), len(degrees)]
    )

    # sorting `actx_ts` to run JAX related operations at the end as they only
    # free the device memory atexit
    for iactx_t, actx_t in sorted(
        enumerate(actx_ts), key=lambda k: _get_actx_t_priority(k[1])
    ):
        for idim, dim in enumerate(dims):
            for iequation, equation in enumerate(equations):
                for idegree, degree in enumerate(degrees):
                    flop_rate[iactx_t, idim, iequation, idegree] = get_flop_rate(
                        actx_t, equation, dim, degree
                    )

    for idim, dim in enumerate(dims):
        for iequation, equation in enumerate(equations):
            for idegree, degree in enumerate(degrees):
                roofline_flop_rate[idim, iequation, idegree] = (
                    get_roofline_flop_rate(equation, dim, degree)
                )
    filename = datetime.datetime.now(pytz.timezone("America/Chicago")).strftime(
        "archive/case_%Y_%m_%d_%H%M.npz"
    )

    np.savez(
        filename,
        equations=equations,
        degrees=degrees,
        dims=dims,
        actx_ts=actx_ts,  # pyright: ignore[reportArgumentType]
        flop_rate=flop_rate,
        roofline_flop_rate=roofline_flop_rate,
    )

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Run DG-FEM benchmarks for arraycontexts",
    )

    parser.add_argument(
        "--equations",
        metavar="E",
        type=str,
        help=(
            "comma separated strings representing which"
            " equations to time (for ex. 'wave,euler')"
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
        metavar="G",
        type=str,
        help=(
            "comma separated integers representing the"
            " polynomial degree of the discretizing function"
            " spaces to run the problems on (for ex."
            " 'pyopencl,jax:jit,pytato:dgfem_opt')"
        ),
        required=True,
    )

    args = CLIArgs(**vars(parser.parse_args()))  # pyright: ignore[reportAny]
    main(
        equations=[k.strip() for k in args.equations.split(",")],
        dims=[int(k.strip()) for k in args.dims.split(",")],
        degrees=[int(k.strip()) for k in args.degrees.split(",")],
        actx_ts=[_NAME_TO_ACTX_CLASS[k] for k in args.actxs.split(",")],
    )
