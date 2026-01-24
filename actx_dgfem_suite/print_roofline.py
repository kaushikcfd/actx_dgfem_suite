__doc__ = """
A binary for printing roofline for DG-FEM operators.Call as
``./print_roofline.py -h`` for a detailed description on how to run the benchmarks.
"""

import argparse
from collections.abc import Sequence

import numpy as np
from tabulate import tabulate

from actx_dgfem_suite.perf_analysis import get_roofline_flop_rate


def stringify_flops(flops: float) -> str:
    if np.isnan(flops):
        return "N/A"
    else:
        return f"{flops * 1e-9:.1f}"


def main(
    equations: Sequence[str],
    dims: Sequence[int],
    degrees: Sequence[int],
    device: str | None,
):
    roofline_flop_rate = np.empty([len(dims), len(equations), len(degrees)])

    for idim, dim in enumerate(dims):
        for iequation, equation in enumerate(equations):
            for idegree, degree in enumerate(degrees):
                roofline_flop_rate[idim, iequation, idegree] = (
                    get_roofline_flop_rate(equation, dim, degree, device_name=device)
                )

    for idim, dim in enumerate(dims):
        for iequation, equation in enumerate(equations):
            print(f"Roofline GFLOPS/s for {dim}D-{equation}:")
            table = []
            for idegree, degree in enumerate(degrees):
                table.append(
                    [
                        f"P{degree}",
                        stringify_flops(
                            roofline_flop_rate[idim, iequation, idegree]
                        ),
                    ]
                )
            print(tabulate(table, tablefmt="fancy_grid", headers=["", "Roofline"]))


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
        "--device",
        metavar="V",
        type=str,
        help="Name of the device to get the roofline for (for"
        " ex. 'NVIDIA TITAN V')",
        required=False,
    )

    args = parser.parse_args()
    main(
        equations=[k.strip() for k in args.equations.split(",")],
        dims=[int(k.strip()) for k in args.dims.split(",")],
        degrees=[int(k.strip()) for k in args.degrees.split(",")],
        device=args.device,
    )
