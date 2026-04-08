"""
Utilities for performance evaluation of a DG-FEM benchmark.

.. autofunction:: get_flop_rate
"""

from time import time
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from arraycontext import (
    ArrayContext,
    EagerJAXArrayContext,
    PyOpenCLArrayContext,
    PytatoJAXArrayContext,
    PytatoPyOpenCLArrayContext,
    rec_multimap_array_container,
)
from meshmode.dof_array import array_context_for_pickling

from actx_dgfem_suite.perf_analysis import get_float64_flops
from actx_dgfem_suite.utils import (
    get_benchmark_ref_input_arguments_path,
    get_benchmark_ref_output_path,
    get_benchmark_rhs_invoker,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _instantiate_actx_t(actx_t: type[ArrayContext]) -> ArrayContext:
    import gc

    gc.collect()

    if issubclass(actx_t, (PyOpenCLArrayContext, PytatoPyOpenCLArrayContext)):
        import pyopencl as cl
        import pyopencl.tools as cl_tools

        ctx = cl.create_some_context()
        cq = cl.CommandQueue(ctx)
        allocator = cl_tools.MemoryPool(cl_tools.ImmediateAllocator(cq))
        return actx_t(cq, allocator)
    elif issubclass(actx_t, (EagerJAXArrayContext, PytatoJAXArrayContext)):
        import os

        if os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] != "false":
            raise RuntimeError(
                "environment variable 'XLA_PYTHON_CLIENT_PREALLOCATE'"
                " is not set 'false'. This is required so that"
                " backends other than JAX can allocate buffers on the"
                " device."
            )

        import jax

        jax.config.update(  # pyright: ignore[reportUnknownMemberType]
            "jax_enable_x64", True
        )
        return actx_t()
    else:
        raise NotImplementedError(actx_t)


def finish_command_queue(actx: ArrayContext) -> None:
    if isinstance(actx, (PytatoPyOpenCLArrayContext, PyOpenCLArrayContext)):
        actx.queue.finish()
    elif isinstance(actx, (EagerJAXArrayContext, PytatoJAXArrayContext)):
        # actx.compile would have called block_until_ready.
        pass
    else:
        raise NotImplementedError(type(actx))


def get_flop_rate(
    actx_t: type[ArrayContext], equation: str, dim: int, degree: int
) -> float:
    """
    Runs the benchmarks corresponding to *equation*, *dim*, *degree* using an
    instance of *actx_t* and returns the FLOP-through as "Total number of
    Floating Point Operations per second".
    """
    import pickle

    from actx_dgfem_suite.utils import is_dataclass_array_container

    rhs_invoker = get_benchmark_rhs_invoker(equation, dim, degree)
    actx = _instantiate_actx_t(actx_t)
    rhs_clbl: Callable[..., Any] = rhs_invoker(actx)  # pyright: ignore[reportAny]

    with (
        open(
            get_benchmark_ref_input_arguments_path(equation, dim, degree), "rb"
        ) as fp,
        array_context_for_pickling(actx),
    ):
        loaded = cast("tuple[tuple[Any, ...], dict[str, Any]]", pickle.load(fp))
        np_args, np_kwargs = loaded

    with (
        open(get_benchmark_ref_output_path(equation, dim, degree), "rb") as fp,
        array_context_for_pickling(actx),
    ):
        ref_output: Any = pickle.load(fp)  # pyright: ignore[reportAny]

    if all(
        (
            is_dataclass_array_container(arg)  # pyright: ignore[reportAny]
            or (
                isinstance(arg, np.ndarray)
                and arg.dtype == "O"
                and all(
                    is_dataclass_array_container(el)  # pyright: ignore[reportAny]
                    for el in arg  # pyright: ignore[reportAny]
                )
            )
            or np.isscalar(arg)
        )
        for arg in np_args  # pyright: ignore[reportAny]
    ) and all(
        is_dataclass_array_container(arg)  # pyright: ignore[reportAny]
        or np.isscalar(arg)  # pyright: ignore[reportAny]
        for arg in np_kwargs.values()  # pyright: ignore[reportAny]
    ):
        args, kwargs = np_args, np_kwargs
    elif any(
        is_dataclass_array_container(arg)  # pyright: ignore[reportAny]
        for arg in np_args  # pyright: ignore[reportAny]
    ) or any(
        is_dataclass_array_container(arg)  # pyright: ignore[reportAny]
        for arg in np_kwargs.values()  # pyright: ignore[reportAny]
    ):
        raise NotImplementedError("Pickling not implemented for input" " types.")
    else:
        args, kwargs = (
            tuple(
                actx.from_numpy(arg) for arg in np_args  # pyright: ignore[reportAny]
            ),
            {
                kw: actx.from_numpy(arg)  # pyright: ignore[reportAny]
                for kw, arg in np_kwargs.items()  # pyright: ignore[reportAny]
            },
        )

    if is_dataclass_array_container(ref_output):  # pyright: ignore[reportAny]
        np_ref_output = actx.to_numpy(ref_output)  # pyright: ignore[reportAny]
    else:
        np_ref_output = ref_output  # pyright: ignore[reportAny]

    # {{{ verify correctness for actx_t

    output = rhs_clbl(*args, **kwargs)  # pyright: ignore[reportAny]
    rec_multimap_array_container(
        lambda x, y: np.testing.assert_allclose(  # pyright: ignore[reportUnknownLambdaType,reportUnknownArgumentType]
            x,  # pyright: ignore[reportUnknownArgumentType]
            y,  # pyright: ignore[reportUnknownArgumentType]
            rtol=1e-7,
            atol=1e-7,
        ),
        np_ref_output,
        actx.to_numpy(output),  # pyright: ignore[reportAny]
    )

    # }}}

    # {{{ warmup rounds

    i_warmup = 0
    t_warmup = 0

    while i_warmup < 20 and t_warmup < 2:
        t_start = time()
        rhs_clbl(*args, **kwargs)
        t_end = time()
        t_warmup += t_end - t_start
        i_warmup += 1

    # }}}

    # {{{ warmup rounds

    i_timing = 0
    t_rhs = 0

    while i_timing < 50 and t_rhs < 5:

        finish_command_queue(actx)
        t_start = time()
        for _ in range(5):
            rhs_clbl(*args, **kwargs)
        finish_command_queue(actx)
        t_end = time()

        t_rhs += t_end - t_start
        i_timing += 5

    # }}}

    flops = get_float64_flops(equation, dim, degree)

    return (flops * i_timing) / t_rhs
