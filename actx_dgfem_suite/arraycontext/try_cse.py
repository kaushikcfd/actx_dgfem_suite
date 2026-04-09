from time import time

import feinsum as fnsm
import loopy as lp
import numpy as np
import pyopencl as cl
import pyopencl.array as cla
import pyopencl.tools as cl_tools

from actx_dgfem_suite.arraycontext.euler_facemass import (
    get_euler_facemass_kernel,
    get_params_for_euler_facemass,
)


def get_footprint_nbytes(t_unit: lp.TranslationUnit) -> int:
    from typing import Any, cast

    from loopy.schedule import CallKernel
    from loopy.schedule.tools import get_subkernel_arg_info
    from pymbolic.typing import Integer
    from pytools import product

    t_unit = lp.linearize(lp.preprocess_program(t_unit))
    knl = t_unit.default_entrypoint
    assert knl.linearization is not None
    subkernel_names = tuple(
        sched_item.kernel_name
        for sched_item in knl.linearization
        if isinstance(sched_item, CallKernel)
    )
    footprint_bytes: Integer = 0
    for subknl_name in subkernel_names:
        subknl_arg_info = get_subkernel_arg_info(knl, subknl_name)
        for tv_name in subknl_arg_info.passed_temporaries:
            tv = knl.temporary_variables[tv_name]
            assert isinstance(
                tv.nbytes, Integer
            ), "Only int shape supported for now."
            footprint_bytes += tv.nbytes
        for arg_name in subknl_arg_info.passed_arg_names:
            arg = knl.arg_dict[arg_name]
            if isinstance(arg, lp.ArrayArg):
                assert arg.shape is not None
                assert arg.dtype is not None
                arg_nbytes = (
                    cast("int", product(cast("tuple[Any, ...]", arg.shape)))
                    * arg.dtype.itemsize
                )
            else:
                assert arg.dtype is not None
                arg_nbytes = arg.dtype.itemsize
            footprint_bytes += cast("Integer", arg_nbytes)

    return cast("int", footprint_bytes)


def get_time_for_idealized_5facemass_knl(cq: cl.CommandQueue):
    from numpy.random import default_rng
    t_unit = lp.make_kernel(
        "{[i,f,j,e]: 0<=f<4 and 0<=i<20 and 0<=j<10 and 0<=e<162000}",
        """
        _LIFT(_0, _1, _2) := M[_0, _1, _2]
        _sgeo(_0, _1) := J[_0, _1]
        _flux_0(_0, _1, _2) := u_0[_0, _1, _2]
        _flux_1(_0, _1, _2) := u_1[_0, _1, _2]
        _flux_2(_0, _1, _2) := u_2[_0, _1, _2]
        _flux_3(_0, _1, _2) := u_3[_0, _1, _2]
        _flux_4(_0, _1, _2) := u_4[_0, _1, _2]
        out_0[e, i] = out_0[e, i] + sum([f, j], _LIFT(i, f, j)
                                               * _sgeo(f, e)
                                               * _flux_0(f, e, j))
        out_1[e, i] = out_1[e, i] + sum([f, j], _LIFT(i, f, j)
                                                * _sgeo(f, e)
                                                * _flux_1(f, e, j))
        out_2[e, i] = out_2[e, i] + sum([f, j], _LIFT(i, f, j)
                                                * _sgeo(f, e)
                                                * _flux_2(f, e, j))
        out_3[e, i] = out_3[e, i] + sum([f, j], _LIFT(i, f, j)
                                                * _sgeo(f, e)
                                                * _flux_3(f, e, j))
        out_4[e, i] = out_4[e, i] + sum([f, j], _LIFT(i, f, j)
                                                * _sgeo(f, e)
                                                * _flux_4(f, e, j))
        """,
        [
            lp.GlobalArg(
                "M,J,out_0,out_1,out_2,out_3,out_4,"
                "u_0,u_1,u_2,u_3,u_4",
                dtype="float64",
                shape=lp.auto,
            ),
            ...,
        ],
        lang_version=(2018, 2),
    )
    ref_t_unit = t_unit
    rng = default_rng(42)
    params_np = {
        "M": rng.random((20, 4, 10)),
        "J": rng.random((4, 162_000)),
        **{f"u_{i}": rng.random((4, 162_000, 10)) for i in range(5)},
    }
    params_cl = {
        **{
            k: cla.to_device(cq, v)
            for k, v in params_np.items()
            if k in t_unit.default_entrypoint.arg_dict
        },
        **{f"out_{i}": cla.zeros(cq, (162000, 20), np.float64) for i in range(5)},
    }

    transform = fnsm.retrieve(
        fnsm.get_a_matched_einsum(t_unit)[0],
        cq.device,
        consider_query=lambda q: q.transform_id != "ifj_fe_fej_to_ei.py",
    )
    t_unit = transform(t_unit)
    alloc = cl_tools.MemoryPool(cl_tools.ImmediateAllocator(cq))

    t_unit = lp.set_options(  # pyright: ignore[reportUnknownMemberType]
        t_unit,
        build_options=["-cl-fast-relaxed-math", "-cl-mad-enable"],
    )
    t_unit = lp.set_options(t_unit, no_numpy=True, return_dict=True)
    t_unit_executor = t_unit.executor(
        cq, entrypoint=None, allocator=alloc, **params_cl
    )

    # warmup
    for _ in range(10):
        t_unit_executor(cq, allocator=alloc, **params_cl)

    cq.finish()
    t_start = time()
    for _ in range(20):
        t_unit_executor(cq, allocator=alloc, **params_cl)
    cq.finish()
    t_end = time()
    print(
        f"[Idealized] Footprint = {get_footprint_nbytes(ref_t_unit) * 1e-6:.1f} MB"
    )
    print(f"[Idealized] Total time = {(t_end - t_start) * 50:.1f} ms")


def get_time_for_3dp3_euler_facemass_knl(cq: cl.CommandQueue):
    t_unit = get_euler_facemass_kernel()
    ref_t_unit = t_unit
    params_np = get_params_for_euler_facemass()
    params_cl = {
        **{
            k: cla.to_device(cq, v)
            for k, v in params_np.items()
            if k in t_unit.default_entrypoint.arg_dict
        },
        **{f"out_{i}": cla.empty(cq, (162000, 20), np.float64) for i in range(5)},
    }

    transform = fnsm.retrieve(
        fnsm.get_a_matched_einsum(t_unit)[0],
        cq.device,
        consider_query=lambda q: q.transform_id != "ifj_fe_fej_to_ei.py",
    )
    t_unit = transform(t_unit)
    alloc = cl_tools.MemoryPool(cl_tools.ImmediateAllocator(cq))

    t_unit = lp.set_options(  # pyright: ignore[reportUnknownMemberType]
        t_unit,
        build_options=["-cl-fast-relaxed-math", "-cl-mad-enable"],
    )
    t_unit = lp.set_options(t_unit, no_numpy=True, return_dict=True)
    t_unit_executor = t_unit.executor(
        cq, entrypoint=None, allocator=alloc, **params_cl
    )

    # warmup
    for _ in range(10):
        t_unit_executor(cq, allocator=alloc, **params_cl)

    cq.finish()
    t_start = time()
    for _ in range(20):
        t_unit_executor(cq, allocator=alloc, **params_cl)
    cq.finish()
    t_end = time()
    print(
        f"[Actual Euler] Footprint = {get_footprint_nbytes(ref_t_unit) * 1e-6:.1f}"
        " MB"
    )
    print(f"[Actual Euler] Total time = {(t_end - t_start) * 50:.1f} ms")


if __name__ == "__main__":
    ctx = cl.create_some_context()
    cq = cl.CommandQueue(ctx)
    get_time_for_idealized_5facemass_knl(cq)
    get_time_for_3dp3_euler_facemass_knl(cq)
