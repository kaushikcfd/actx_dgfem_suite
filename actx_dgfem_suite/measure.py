"""
Utilities for performance evaluation of a DG-FEM benchmark.
"""

import gc

from arraycontext import (
    ArrayContext,
    EagerJAXArrayContext,
    NumpyArrayContext,
    PyOpenCLArrayContext,
    PytatoJAXArrayContext,
    PytatoPyOpenCLArrayContext,
)


def _is_cupy_actx_t(actx_t: type[ArrayContext]) -> bool:
    # Check the MRO by name to avoid introducing a dependency on the CuPy module.
    return any(
        base.__module__ == "actx_dgfem_suite.arraycontext.cupy_actx"
        and base.__name__ == "CuPyArrayContext"
        for base in actx_t.__mro__
    )


def instantiate_actx_t[ActxT: (ArrayContext)](actx_t: type[ActxT]) -> ActxT:
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
    elif issubclass(actx_t, NumpyArrayContext) or _is_cupy_actx_t(actx_t):
        return actx_t()
    else:
        raise NotImplementedError(actx_t)


def finish_command_queue(actx: ArrayContext) -> None:
    if isinstance(actx, (PytatoPyOpenCLArrayContext, PyOpenCLArrayContext)):
        actx.queue.finish()
    elif isinstance(actx, (EagerJAXArrayContext, PytatoJAXArrayContext)):
        # actx.compile would have called block_until_ready.
        pass
    elif isinstance(actx, NumpyArrayContext):
        pass
    elif _is_cupy_actx_t(type(actx)):
        actx.get_current_stream().synchronize()  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
    else:
        raise NotImplementedError(type(actx))
