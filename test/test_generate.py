from actx_dgfem_suite.codegen import SuiteGeneratingArraycontext
from pyopencl.tools import pytest_generate_tests_for_pyopencl \
    as pytest_generate_tests  # noqa: F401
import pyopencl as cl
import pyopencl.tools as cl_tools

import tempfile


def _get_suite_generating_actx(ctx):
    cq = cl.CommandQueue(ctx)
    allocator = cl_tools.MemoryPool(cl_tools.ImmediateAllocator(cq))
    tempdir = tempfile.mkdtemp()

    return SuiteGeneratingArraycontext(
        cq, allocator,
        main_file_path=f"{tempdir}/main.py",
        datawrappers_path=f"{tempdir}/datawrappers.npz",
        pickled_ref_input_args_path=f"{tempdir}/ref_input_args.npz",
        pickled_ref_output_path=f"{tempdir}/ref_output.npz",
    )


def test_array_returning_function(ctx_factory):
    cl_ctx = ctx_factory()

    def f(x):
        return 2*x

    actx = _get_suite_generating_actx(cl_ctx)

    a = actx.zeros(10, "float64")
    actx.compile(f)(a+42)  # internally asserts that the result is correct

    a = actx.zeros(10, "float32")
    actx.compile(f)(actx.thaw(actx.freeze(a+1729)))


def test_array_container_returning_function(ctx_factory):
    cl_ctx = ctx_factory()

    def f(x):
        from pytools.obj_array import make_obj_array
        return make_obj_array([2*x, 3*x, x**2])

    actx = _get_suite_generating_actx(cl_ctx)

    a = actx.zeros(10, "float64")
    actx.compile(f)(a+42)  # internally asserts that the result is correct

    a = actx.zeros(10, "float32")
    actx.compile(f)(actx.thaw(actx.freeze(a+1729)))
