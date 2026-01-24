__doc__ = """
A binary for printing roofline for DG-FEM operators.Call as
``./print_roofline.py -h`` for a detailed description on how to run the benchmarks.
"""

import argparse
import numpy as np

from actx_dgfem_suite.perf_analysis import get_float64_flops
from typing import Sequence
from tabulate import tabulate


def stringify_flops(flops: float) -> str:
    if np.isnan(flops):
        return "N/A"
    else:
        return f"{flops*1e-9:.1f}"


def main(equations: Sequence[str],
         dims: Sequence[int],
         degrees: Sequence[int],
         ):
    f64_flops = np.empty([len(dims), len(equations), len(degrees)])

    for idim, dim in enumerate(dims):
        for iequation, equation in enumerate(equations):
            for idegree, degree in enumerate(degrees):
                f64_flops[idim, iequation, idegree] = (
                    get_float64_flops(equation, dim, degree)
                )

    for idim, dim in enumerate(dims):
        for iequation, equation in enumerate(equations):
            print(f"Roofline GFLOPS/s for {dim}D-{equation}:")
            table = []
            for idegree, degree in enumerate(degrees):
                table.append(
                    [f"P{degree}",
                     stringify_flops(f64_flops[idim, iequation, idegree])
                     ]
                )
            print(tabulate(table, tablefmt="fancy_grid", headers=["",
                                                                  "Roofline"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="print_ngflops.py",
        description="Pretty print #GFLOPS for various DG-FEM operators",
    )

    parser.add_argument("--equations", metavar="E", type=str,
                        help=("comma separated strings representing which"
                              " equations to time (for ex. 'wave,euler')"),
                        required=True,
                        )
    parser.add_argument("--dims", metavar="D", type=str,
                        help=("comma separated integers representing the"
                              " topological dimensions to run the problems on"
                              " (for ex. 2,3 to run 2D and 3D versions of the"
                              " problem)"),
                        required=True,
                        )
    parser.add_argument("--degrees", metavar="G", type=str,
                        help=("comma separated integers representing the"
                              " polynomial degree of the discretizing function"
                              " spaces to run the problems on (for ex. 1,2,3"
                              " to run using P1,P2,P3 function spaces)"),
                        required=True,
                        )

    args = parser.parse_args()
    main(equations=[k.strip() for k in args.equations.split(",")],
         dims=[int(k.strip()) for k in args.dims.split(",")],
         degrees=[int(k.strip()) for k in args.degrees.split(",")])
