__doc__ = """
A binary for running DG-FEM benchmarks for an array of arraycontexts. Call as
``python run.py -h`` for a detailed description on how to run the benchmarks.
"""

import argparse
import dataclasses as dc
from collections.abc import Sequence

import pyopencl as cl
import pyopencl.tools as cl_tools

from actx_dgfem_suite import utils
from actx_dgfem_suite.codegen import SuiteGeneratingArraycontext


def get_actx(equation: str, dim: int, degree: int) -> SuiteGeneratingArraycontext:

    cl_ctx = cl.create_some_context()
    cq = cl.CommandQueue(cl_ctx)
    allocator = cl_tools.MemoryPool(cl_tools.ImmediateAllocator(cq))

    return SuiteGeneratingArraycontext(
        cq,
        allocator,
        main_file_path=utils.get_benchmark_main_file_path(equation, dim, degree),
        datawrappers_path=utils.get_benchmark_literals_path(equation, dim, degree),
        pickled_ref_input_args_path=utils.get_benchmark_ref_input_arguments_path(
            equation, dim, degree
        ),
        pickled_ref_output_path=utils.get_benchmark_ref_output_path(
            equation, dim, degree
        ),
    )


def main(
    equations: Sequence[str],
    dims: Sequence[int],
    degrees: Sequence[int],
):
    for dim in dims:
        for equation in equations:
            for degree in degrees:
                if equation == "wave":
                    from actx_dgfem_suite.suite_generators.wave import main as driver

                    actx = get_actx(equation, dim, degree)
                    driver(dim=dim, order=degree, actx=actx, ndofs=3_000_000)
                elif equation == "tiny_wave":
                    from actx_dgfem_suite.suite_generators.wave import main as driver

                    actx = get_actx(equation, dim, degree)
                    driver(dim=dim, order=degree, actx=actx, ndofs=1_000)
                elif equation == "euler":
                    from actx_dgfem_suite.suite_generators.euler import (
                        main as driver,
                    )

                    actx = get_actx(equation, dim, degree)
                    driver(dim=dim, order=degree, actx=actx, ndofs=3_000_000)
                elif equation == "tiny_euler":
                    from actx_dgfem_suite.suite_generators.euler import (
                        main as driver,
                    )

                    actx = get_actx(equation, dim, degree)
                    driver(dim=dim, order=degree, actx=actx, ndofs=1_000)
                elif equation == "maxwell":
                    from actx_dgfem_suite.suite_generators.maxwell import (
                        main as driver,
                    )

                    actx = get_actx(equation, dim, degree)
                    driver(dim=dim, order=degree, actx=actx, ndofs=3_000_000)
                elif equation == "tiny_maxwell":
                    from actx_dgfem_suite.suite_generators.maxwell import (
                        main as driver,
                    )

                    actx = get_actx(equation, dim, degree)
                    driver(dim=dim, order=degree, actx=actx, ndofs=1_000)
                elif equation == "cns_without_chem":
                    from actx_dgfem_suite.suite_generators.cns import main as driver

                    actx = get_actx(equation, dim, degree)
                    driver(dim=dim, order=degree, actx=actx, single_gas_only=True)
                elif equation == "cns_with_chem":
                    from actx_dgfem_suite.suite_generators.cns import main as driver

                    actx = get_actx(equation, dim, degree)
                    driver(dim=dim, order=degree, actx=actx, single_gas_only=False)
                else:
                    raise NotImplementedError(equation, dim, degree)

                print(75 * "-")
                print(f"Done generating {equation}_{dim}D_P{degree}")
                print(75 * "-")


@dc.dataclass(frozen=True)
class CLIArgs:
    equations: str
    dims: str
    degrees: str


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate DG-FEM benchmarks suite",
    )

    parser.add_argument(
        "--equations",
        metavar="E",
        type=str,
        help=(
            "comma separated strings representing which"
            " equations to time. Available options:"
            "wave, euler, maxwell, tiny_wave, tiny_euler, tiny_maxwell,"
            " cns_with_chem and cns_without_chem.)"
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

    args = CLIArgs(**vars(parser.parse_args()))  # pyright: ignore[reportAny]
    main(
        equations=[k.strip() for k in args.equations.split(",")],
        dims=[int(k.strip()) for k in args.dims.split(",")],
        degrees=[int(k.strip()) for k in args.degrees.split(",")],
    )
