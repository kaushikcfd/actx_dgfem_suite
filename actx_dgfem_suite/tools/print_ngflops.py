__doc__ = """
A binary for printing roofline for DG-FEM operators.Call as
``./print_roofline.py -h`` for a detailed description on how to run the benchmarks.
"""

__copyright__ = """
Copyright (C) 2026 Kaushik Kulkarni
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
from collections.abc import Sequence
from typing import cast

import numpy as np
import numpy.typing as npt
from tabulate import tabulate

from actx_dgfem_suite.perf_analysis import get_float64_flops


def stringify_flops(flops: float) -> str:
    if np.isnan(flops):
        return "N/A"
    else:
        return f"{flops * 1e-9:.1f}"


def main(
    equations: Sequence[str],
    dims: Sequence[int],
    degrees: Sequence[int],
):
    f64_flops: npt.NDArray[np.float64] = np.empty(
        [len(dims), len(equations), len(degrees)]
    )

    for idim, dim in enumerate(dims):
        for iequation, equation in enumerate(equations):
            for idegree, degree in enumerate(degrees):
                f64_flops[idim, iequation, idegree] = get_float64_flops(
                    equation, dim, degree
                )

    for idim, dim in enumerate(dims):
        for iequation, equation in enumerate(equations):
            print(f"Roofline GFLOPS/s for {dim}D-{equation}:")
            table: list[list[str]] = []
            for idegree, degree in enumerate(degrees):
                table.append(
                    [
                        f"P{degree}",
                        stringify_flops(
                            f64_flops[
                                idim, iequation, idegree
                            ]  # pyright: ignore[reportAny]
                        ),
                    ]
                )
            print(tabulate(table, tablefmt="fancy_grid", headers=["", "Roofline"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="print_ngflops.py",
        description="Pretty print #GFLOPS for various DG-FEM operators",
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

    args = parser.parse_args()
    main(
        equations=[k.strip() for k in cast("str", args.equations).split(",")],
        dims=[int(k.strip()) for k in cast("str", args.dims).split(",")],
        degrees=[int(k.strip()) for k in cast("str", args.degrees).split(",")],
    )
