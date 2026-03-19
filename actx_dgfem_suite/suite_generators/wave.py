__copyright__ = """
Copyright (C) 2020 Andreas Kloeckner
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


import logging
from dataclasses import dataclass
from typing import Callable, ClassVar, cast

import grudge.geometry as geom
import numpy as np
from arraycontext import (
    Array,
    ArrayContext,
    dataclass_array_container,
    with_container_arithmetic,
)
from grudge import op
from grudge.discretization import DiscretizationCollection
from grudge.dof_desc import (
    DISCR_TAG_BASE,
    DISCR_TAG_QUAD,
    DiscretizationTag,
    DOFDesc,
    as_dofdesc,
)
from grudge.trace_pair import TracePair
from meshmode.dof_array import DOFArray
from meshmode.mesh import BTAG_ALL
from pytools.obj_array import flat, new_1d

from actx_dgfem_suite.utils import get_nel_1d_for_regular_rect_mesh

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# {{{ wave equation bits


@with_container_arithmetic(
    bcasts_across_obj_array=True,
    rel_comparison=True,
)
@dataclass_array_container
@dataclass(frozen=True)
class WaveState:
    u: DOFArray
    v: np.ndarray  # [object array]

    # NOTE: disable numpy doing any array math
    __array_ufunc__: ClassVar[None] = None

    def __post_init__(self):
        assert isinstance(self.v, np.ndarray) and self.v.dtype.char == "O"


def wave_flux(
    actx: ArrayContext,
    dcoll: DiscretizationCollection,
    c: float,
    w_tpair: TracePair[WaveState],  # pyright: ignore[reportInvalidTypeArguments]
) -> WaveState:
    u = w_tpair.u
    v = w_tpair.v
    dd = w_tpair.dd

    normal = geom.normal(actx, dcoll, dd)
    flux_weak = WaveState(
        u=v.avg @ normal,  # pyright: ignore[reportOperatorIssue, reportArgumentType]
        v=u.avg * normal,  # pyright: ignore[reportOperatorIssue, reportArgumentType]
    )

    # upwind
    v_jump = (  # pyright: ignore[reportUnknownVariableType, reportOperatorIssue]
        v.diff @ normal
    )
    flux_weak += (  # pyright: ignore[reportUnknownVariableType, reportOperatorIssue]
        WaveState(
            u=0.5 * u.diff,  # pyright: ignore[reportArgumentType]
            v=0.5
            * v_jump
            * normal,  # pyright: ignore[reportOperatorIssue, reportArgumentType]
        )
    )

    return op.project(  # pyright: ignore[reportUnknownVariableType]
        dcoll,
        dd,
        dd.with_domain_tag("all_faces"),  # pyright: ignore[reportArgumentType]
        c * flux_weak,  # pyright: ignore[reportUnknownArgumentType]
    )


class _WaveStateTag:
    pass


def wave_operator(
    actx: ArrayContext,
    dcoll: DiscretizationCollection,
    c: float,
    w: WaveState,
    quad_tag: DiscretizationTag | None = None,
) -> WaveState:
    dd_base = as_dofdesc("vol")
    dd_vol = DOFDesc("vol", quad_tag)
    dd_faces = DOFDesc("all_faces", quad_tag)
    dd_btag = as_dofdesc(BTAG_ALL).with_discr_tag(
        quad_tag  # pyright: ignore[reportArgumentType]
    )

    def interp_to_surf_quad(
        utpair: TracePair[WaveState],  # pyright: ignore[reportInvalidTypeArguments]
    ) -> TracePair[WaveState]:  # pyright: ignore[reportInvalidTypeArguments]
        local_dd = utpair.dd
        local_dd_quad = local_dd.with_discr_tag(
            quad_tag  # pyright: ignore[reportArgumentType]
        )
        return TracePair(  # pyright: ignore[reportUnknownVariableType]
            local_dd_quad,
            interior=op.project(dcoll, local_dd, local_dd_quad, utpair.int),
            exterior=op.project(dcoll, local_dd, local_dd_quad, utpair.ext),
        )

    w_quad = op.project(dcoll, dd_base, dd_vol, w)
    u = w_quad.u
    v = w_quad.v

    dir_w = op.project(dcoll, dd_base, dd_btag, w)
    dir_u = dir_w.u
    dir_v = dir_w.v
    dir_bval = WaveState(u=dir_u, v=dir_v)
    dir_bc = WaveState(u=-dir_u, v=dir_v)

    return op.inverse_mass(  # pyright: ignore[reportReturnType]
        dcoll,
        WaveState(  # pyright: ignore[reportUnknownArgumentType, reportOperatorIssue]
            u=-c  # pyright: ignore[reportOperatorIssue, reportArgumentType]
            * op.weak_local_div(
                dcoll, dd_vol, v  # pyright: ignore[reportArgumentType]
            ),
            v=-c
            * op.weak_local_grad(
                dcoll, dd_vol, u
            ),  # pyright: ignore[reportOperatorIssue, reportArgumentType]
        )
        + op.face_mass(
            dcoll,
            dd_faces,
            wave_flux(  # pyright: ignore[reportUnknownArgumentType]
                actx,
                dcoll,
                c=c,
                w_tpair=op.bdry_trace_pair(  # pyright: ignore[reportUnknownArgumentType]
                    dcoll,
                    dd_btag,
                    interior=dir_bval,  # pyright: ignore[reportArgumentType]
                    exterior=dir_bc,  # pyright: ignore[reportArgumentType]
                ),
            )
            + sum(  # pyright: ignore[reportCallIssue]
                wave_flux(
                    actx,
                    dcoll,
                    c=c,
                    w_tpair=interp_to_surf_quad(
                        tpair  # pyright: ignore[reportUnknownArgumentType, reportArgumentType]
                    ),
                )
                for tpair in op.interior_trace_pairs(  # pyright: ignore[reportUnknownVariableType]
                    dcoll,
                    w,  # pyright: ignore[reportArgumentType]
                    comm_tag=_WaveStateTag,
                )
            ),
        ),
    )


# }}}


def estimate_rk4_timestep(
    actx: ArrayContext, dcoll: DiscretizationCollection, c: float
) -> Array:
    from grudge.dt_utils import characteristic_lengthscales

    local_dts = characteristic_lengthscales(actx, dcoll) / c

    return op.nodal_min(dcoll, "vol", local_dts)


def bump(
    actx: ArrayContext, dcoll: DiscretizationCollection, t: float = 0
) -> DOFArray:
    source_center = np.array([0.2, 0.35, 0.1])[: dcoll.dim]
    source_width = 0.05
    source_omega = 3

    nodes = actx.thaw(dcoll.nodes())
    center_dist = flat([nodes[i] - source_center[i] for i in range(dcoll.dim)])

    return cast(
        "DOFArray",
        np.cos(source_omega * t)
        * actx.np.exp(
            -np.dot(  # pyright: ignore[reportAny]
                center_dist,  # pyright: ignore[reportArgumentType]
                center_dist,  # pyright: ignore[reportArgumentType]
            )
            / source_width**2
        ),
    )


def main(dim: int, order: int, actx: ArrayContext, ndofs: int) -> None:
    nel_1d = get_nel_1d_for_regular_rect_mesh(dim, order, ndofs)

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    logger.info("%d elements", mesh.nelements)

    from meshmode.discretization.poly_element import (
        QuadratureSimplexGroupFactory,
        default_simplex_group_factory,
    )

    dcoll = DiscretizationCollection(
        actx,
        mesh,
        discr_tag_to_group_factory={
            DISCR_TAG_BASE: default_simplex_group_factory(base_dim=dim, order=order),
            # High-order quadrature to integrate inner products of polynomials
            # on warped geometry exactly (metric terms are also polynomial)
            DISCR_TAG_QUAD: QuadratureSimplexGroupFactory(3 * order),
        },
    )

    fields = WaveState(
        u=bump(actx, dcoll),
        v=new_1d(
            [dcoll.zeros(actx) for _ in range(dcoll.dim)]
        ),  # pyright: ignore[reportArgumentType]
    )

    c = 1

    # FIXME: Sketchy, empirically determined fudge factor
    # 5/4 to account for larger LSRK45 stability region
    dt = actx.to_numpy(0.45 * estimate_rk4_timestep(actx, dcoll, c)) * 5 / 4

    def rhs(w: WaveState) -> WaveState:
        return wave_operator(actx, dcoll, c=c, w=w, quad_tag=None)

    compiled_rhs = cast(
        "Callable[[WaveState], WaveState]",
        actx.compile(rhs),  # pyright: ignore[reportArgumentType]
    )

    logger.info("dt = %g", dt)

    fields = actx.thaw(actx.freeze(fields))
    compiled_rhs(fields)


# vim: foldmethod=marker
