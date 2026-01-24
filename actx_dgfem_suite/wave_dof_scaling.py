__copyright__ = """
Copyright (C) 2020 Andreas Kloeckner (for implementing the wave eqn solver)
Copyright (C) 2021 University of Illinois Board of Trustees
Copyright (C) 2023 Kaushik Kulkarni
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
import loopy as lp
import numpy as np
from time import time

from typing import Type, Sequence
from bidict import bidict
from meshmode.array_context import (
    BatchedEinsumPytatoPyOpenCLArrayContext,
    PyOpenCLArrayContext as BasePyOpenCLArrayContext,
)
from arraycontext import ArrayContext, PytatoJAXArrayContext, EagerJAXArrayContext
from tabulate import tabulate

from arraycontext import (
    with_container_arithmetic,
    dataclass_array_container
)

from dataclasses import dataclass

from pytools.obj_array import flat_obj_array, make_obj_array

from meshmode.dof_array import DOFArray
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa

from grudge.dof_desc import as_dofdesc, DOFDesc, DISCR_TAG_BASE, DISCR_TAG_QUAD
from grudge.trace_pair import TracePair
from grudge.discretization import DiscretizationCollection

import grudge.op as op

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class PyOpenCLArrayContext(BasePyOpenCLArrayContext):
    def transform_loopy_program(self,
                                t_unit: lp.TranslationUnit
                                ) -> lp.TranslationUnit:
        from meshmode.arraycontext_extras.split_actx.utils import (
            split_iteration_domain_across_work_items)
        t_unit = split_iteration_domain_across_work_items(t_unit, self.queue.device)
        return t_unit


def _get_actx_t_priority(actx_t):
    if issubclass(actx_t, PytatoJAXArrayContext):
        return 10
    else:
        return 1


def stringify_dofs_per_s(dofs_per_s: float) -> str:
    if np.isnan(dofs_per_s):
        return "N/A"
    else:
        return f"{dofs_per_s*1e-9:.3f}"


def get_nunit_dofs(*, dim: int, degree: int) -> int:
    if dim == 3:
        if degree == 1:
            return 4
        elif degree == 2:
            return 10
        elif degree == 3:
            return 20
        elif degree == 4:
            return 35
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


def get_actual_scalar_ndofs(*, ndofs: int, degree: int, dim: int) -> int:
    from math import ceil
    nunit_dofs = get_nunit_dofs(dim=dim, degree=degree)
    nel_1d = ceil(((ndofs/nunit_dofs)/6)**(1/3))
    return 6 * (nel_1d ** 3) * nunit_dofs


# {{{ wave eqution bits

@with_container_arithmetic(bcast_obj_array=True, rel_comparison=True,
        _cls_has_array_context_attr=True)
@dataclass_array_container
@dataclass(frozen=True)
class WaveState:
    u: DOFArray
    v: np.ndarray  # [object array]

    def __post_init__(self):
        assert isinstance(self.v, np.ndarray) and self.v.dtype.char == "O"

    @property
    def array_context(self):
        return self.u.array_context


def wave_flux(actx, dcoll, c, w_tpair):
    u = w_tpair.u
    v = w_tpair.v
    dd = w_tpair.dd

    normal = actx.thaw(dcoll.normal(dd))

    flux_weak = WaveState(
        u=v.avg @ normal,
        v=u.avg * normal
    )

    # upwind
    v_jump = v.diff @ normal
    flux_weak += WaveState(
        u=0.5 * u.diff,
        v=0.5 * v_jump * normal,
    )

    return op.project(dcoll, dd, dd.with_dtag("all_faces"), c*flux_weak)


class _WaveStateTag:
    pass


def wave_operator(actx, dcoll, c, w, quad_tag=None):
    dd_base = as_dofdesc("vol")
    dd_vol = DOFDesc("vol", quad_tag)
    dd_faces = DOFDesc("all_faces", quad_tag)
    dd_btag = as_dofdesc(BTAG_ALL).with_discr_tag(quad_tag)

    def interp_to_surf_quad(utpair):
        local_dd = utpair.dd
        local_dd_quad = local_dd.with_discr_tag(quad_tag)
        return TracePair(
            local_dd_quad,
            interior=op.project(dcoll, local_dd, local_dd_quad, utpair.int),
            exterior=op.project(dcoll, local_dd, local_dd_quad, utpair.ext)
        )

    w_quad = op.project(dcoll, dd_base, dd_vol, w)
    u = w_quad.u
    v = w_quad.v

    dir_w = op.project(dcoll, dd_base, dd_btag, w)
    dir_u = dir_w.u
    dir_v = dir_w.v
    dir_bval = WaveState(u=dir_u, v=dir_v)
    dir_bc = WaveState(u=-dir_u, v=dir_v)

    return (
        op.inverse_mass(
            dcoll,
            WaveState(
                u=-c*op.weak_local_div(dcoll, dd_vol, v),
                v=-c*op.weak_local_grad(dcoll, dd_vol, u)
            )
            + op.face_mass(
                dcoll,
                dd_faces,
                wave_flux(
                    actx,
                    dcoll, c=c,
                    w_tpair=op.bdry_trace_pair(dcoll,
                                               dd_btag,
                                               interior=dir_bval,
                                               exterior=dir_bc)
                ) + sum(
                    wave_flux(actx, dcoll, c=c, w_tpair=interp_to_surf_quad(tpair))
                    for tpair in op.interior_trace_pairs(dcoll, w,
                        comm_tag=_WaveStateTag)
                )
            )
        )
    )


def bump(actx, dcoll, t=0):
    source_center = np.array([0.2, 0.35, 0.1])[:dcoll.dim]
    source_width = 0.05
    source_omega = 3

    nodes = actx.thaw(dcoll.nodes())
    center_dist = flat_obj_array([
        nodes[i] - source_center[i]
        for i in range(dcoll.dim)
        ])

    return (
        np.cos(source_omega*t)
        * actx.np.exp(
            -np.dot(center_dist, center_dist)
            / source_width**2))


def get_wave_rhs(*, actx, dim, order, ndofs, use_nonaffine_mesh=False):
    from math import ceil

    nunit_dofs = get_nunit_dofs(dim=dim, degree=order)

    nel_1d = ceil(((ndofs/nunit_dofs)/6)**(1/3))
    assert (6 * (nel_1d ** 3) * nunit_dofs) == ndofs

    if use_nonaffine_mesh:
        from meshmode.mesh.generation import generate_warped_rect_mesh
        # FIXME: *generate_warped_rect_mesh* in meshmode warps a
        # rectangle domain with hard-coded vertices at (-0.5,)*dim
        # and (0.5,)*dim. Should extend the function interface to allow
        # for specifying the end points directly.
        mesh = generate_warped_rect_mesh(dim=dim, order=order,
                                         nelements_side=nel_1d)
    else:
        from meshmode.mesh.generation import generate_regular_rect_mesh
        mesh = generate_regular_rect_mesh(
                a=(-0.5,)*dim,
                b=(0.5,)*dim,
                nelements_per_axis=(nel_1d,)*dim)

    logger.info("%d elements", mesh.nelements)

    from meshmode.discretization.poly_element import \
            QuadratureSimplexGroupFactory, \
            default_simplex_group_factory
    dcoll = DiscretizationCollection(
        actx, mesh,
        discr_tag_to_group_factory={
            DISCR_TAG_BASE: default_simplex_group_factory(base_dim=dim, order=order),
            # High-order quadrature to integrate inner products of polynomials
            # on warped geometry exactly (metric terms are also polynomial)
            DISCR_TAG_QUAD: QuadratureSimplexGroupFactory(3*order),
        })

    quad_tag = None
    fields = WaveState(
        u=bump(actx, dcoll),
        v=make_obj_array([dcoll.zeros(actx) for i in range(dcoll.dim)])
    )

    c = 1

    # FIXME: Sketchy, empirically determined fudge factor
    # 5/4 to account for larger LSRK45 stability region
    dt = 1e-3

    def rhs(t, w):
        return wave_operator(actx, dcoll, c=c, w=w, quad_tag=quad_tag)

    logger.info("dt = %g", dt)

    t = np.float64(0.5)
    fields = actx.thaw(actx.freeze(fields))

    return lambda x: rhs(t, x), fields

# }}}


def main(dim: int,
         degrees: Sequence[int],
         actx_t: Type[ArrayContext],
         ):
    from actx_dgfem_suite.measure import _instantiate_actx_t

    ndofs_list = [0.5e6, 1e6, 2e6, 3e6, 4e6, 6e6]
    dof_throughput = np.empty([len(degrees), len(ndofs_list)])

    for i_degree, degree in enumerate(degrees):
        for i_ndofs, ndofs in enumerate(ndofs_list):
            actx = _instantiate_actx_t(actx_t)
            actual_scalar_ndofs = get_actual_scalar_ndofs(ndofs=ndofs,
                                                          degree=degree,
                                                          dim=dim)
            rhs_clbl, rhs_args = get_wave_rhs(actx=actx,
                                              dim=dim,
                                              order=degree,
                                              ndofs=actual_scalar_ndofs)
            compiled_rhs_clbl = actx.compile(rhs_clbl)

            # {{{ warmup rounds

            i_warmup = 0
            t_warmup = 0

            while i_warmup < 20 and t_warmup < 2:
                t_start = time()
                compiled_rhs_clbl(rhs_args)
                t_end = time()
                t_warmup += (t_end - t_start)
                i_warmup += 1

            # }}}

            # {{{ timing rounds

            i_timing = 0
            t_rhs = 0

            while i_timing < 100 and t_rhs < 5:

                t_start = time()
                for _ in range(40):
                    compiled_rhs_clbl(rhs_args)
                t_end = time()

                t_rhs += (t_end - t_start)
                i_timing += 40

            # }}}

            # Multiplying by "(dim + 1)" to account for DOFs for all fields
            dof_throughput[i_degree, i_ndofs] = (actual_scalar_ndofs
                                                 * (dim + 1)
                                                 * i_timing) / t_rhs

            del actx
            import gc
            gc.collect()

    print(f"GDOFS/s for {dim}D-wave for {_NAME_TO_ACTX_CLASS.inv[actx_t]}:")
    table = []
    for i_degree, degree in enumerate(degrees):
        table.append(
            [f"P{degree}",
             *[stringify_dofs_per_s(dof_throughput[i_degree, i_ndofs])
               for i_ndofs, _ in enumerate(ndofs_list)],
             ]
        )
    print(tabulate(table,
                   tablefmt="fancy_grid",
                   headers=[""] + [str(ndofs) for ndofs in ndofs_list]))


_NAME_TO_ACTX_CLASS = bidict({
    "pyopencl": PyOpenCLArrayContext,
    "jax:nojit": EagerJAXArrayContext,
    "jax:jit": PytatoJAXArrayContext,
    "pytato:batched_einsum": BatchedEinsumPytatoPyOpenCLArrayContext,
})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=".py",
        description="Obtain DOF-throughput scaling for different problem sizes of a"
                    "wave equation solver.",
    )

    parser.add_argument("--dim", metavar="D", type=int,
                        help=("An integer representing the"
                              " topological dimensions to run the problems on"
                              " (for ex. 3 to run 3D versions of the"
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
    parser.add_argument("--actx", metavar="G", type=str,
                        help=("strings denoting which array context to use for the"
                              " scaling test  (for ex."
                              " 'pytato:batched_einsum')"),
                        required=True,
                        )

    args = parser.parse_args()
    main(dim=args.dim,
         degrees=[int(k.strip()) for k in args.degrees.split(",")],
         actx_t=_NAME_TO_ACTX_CLASS[args.actx])
