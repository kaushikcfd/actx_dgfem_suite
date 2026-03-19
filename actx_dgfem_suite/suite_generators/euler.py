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

import numpy as np
import numpy.typing as npt
from arraycontext import ArrayContext
from grudge import DiscretizationCollection
from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD
from grudge.models.euler import ConservedEulerField, EulerOperator, InviscidWallBC
from meshmode.discretization.poly_element import (
    QuadratureSimplexGroupFactory,
    default_simplex_group_factory,
)
from meshmode.dof_array import DOFArray as _DOFArray
from meshmode.mesh import BTAG_ALL
from meshmode.mesh.generation import generate_regular_rect_mesh
from pytools.obj_array import ObjectArray, new_1d

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

GLOBAL_NDOFS = 3e6


def _get_nel_1d(dim: int, order: int) -> int:
    from math import cbrt, ceil

    if dim == 3:
        if order == 1:
            nel_1d = ceil(cbrt((GLOBAL_NDOFS / 4) / 12))
        elif order == 2:
            nel_1d = ceil(cbrt((GLOBAL_NDOFS / 10) / 12))
        elif order == 3:
            nel_1d = ceil(cbrt((GLOBAL_NDOFS / 20) / 12))
        elif order == 4:
            nel_1d = ceil(cbrt((GLOBAL_NDOFS / 35) / 12))
        else:
            raise NotImplementedError(order)
    elif dim == 2:
        if order in {1, 2, 3, 4}:
            nel_1d = 1000
        else:
            raise NotImplementedError(order)
    else:
        raise NotImplementedError

    return int(nel_1d)


def gaussian_profile(
    x_vec: ObjectArray[tuple[int], _DOFArray],
    t: float = 0,
    rho0: float = 1.0,
    rhoamp: float = 1.0,
    p0: float = 1.0,
    gamma: float = 1.4,
    center: npt.NDArray[np.float64] | None = None,
    velocity: npt.NDArray[np.float64] | None = None,
) -> ConservedEulerField:

    dim = len(x_vec)
    if center is None:
        center = np.zeros(shape=(dim,))
    if velocity is None:
        velocity = np.zeros(shape=(dim,))

    lump_loc = center + t * velocity

    # coordinates relative to lump center
    rel_center = new_1d([x_vec[i] - lump_loc[i] for i in range(dim)])
    actx = x_vec[0].array_context
    assert actx is not None
    r = actx.np.sqrt(  # pyright: ignore[reportAny]
        np.dot(
            rel_center, rel_center  # pyright: ignore[reportArgumentType, reportAny]
        )
    )
    expterm = rhoamp * actx.np.exp(1 - r**2)  # pyright: ignore[reportAny]

    mass = expterm + rho0  # pyright: ignore[reportAny]
    mom = velocity * mass  # pyright: ignore[reportAny]
    energy = (p0 / (gamma - 1.0)) + np.dot(  # pyright: ignore[reportAny]
        mom, mom  # pyright: ignore[reportAny]
    ) / (2.0 * mass)

    return ConservedEulerField(
        mass=mass, energy=energy, momentum=mom  # pyright: ignore[reportAny]
    )


def make_pulse(
    amplitude: float,
    r0: npt.NDArray[np.float64],
    w: float,
    r: ObjectArray[tuple[int], _DOFArray],
) -> _DOFArray:
    dim = len(r)
    r_0 = np.zeros(dim)
    r_0 = r_0 + r0
    rel_center = new_1d([r[i] - r_0[i] for i in range(dim)])
    actx = r[0].array_context
    assert actx is not None
    rms2 = w * w
    r2 = (  # pyright: ignore[reportAny]
        np.dot(rel_center, rel_center) / rms2  # pyright: ignore[reportArgumentType]
    )
    return actx.np.exp(-0.5 * r2)  # pyright: ignore[reportAny]


def acoustic_pulse_condition(
    x_vec: ObjectArray[tuple[int], _DOFArray], t: float = 0
) -> ConservedEulerField:
    dim = len(x_vec)
    vel = np.zeros(shape=(dim,))
    orig = np.zeros(shape=(dim,))
    uniform_gaussian = gaussian_profile(
        x_vec, t=t, center=orig, velocity=vel, rhoamp=0.0
    )

    amplitude = 1.0
    width = 0.1
    pulse = make_pulse(amplitude, orig, width, x_vec)

    return ConservedEulerField(
        mass=uniform_gaussian.mass,
        energy=uniform_gaussian.energy + pulse,
        momentum=uniform_gaussian.momentum,
    )


def main(
    dim: int, order: int, actx: ArrayContext, *, overintegration: bool = False
) -> None:

    nel_1d = _get_nel_1d(dim, order)

    # eos-related parameters
    gamma = 1.4

    # {{{ discretization

    box_ll = -0.5
    box_ur = 0.5
    mesh = generate_regular_rect_mesh(
        a=(box_ll,) * dim, b=(box_ur,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    if overintegration:
        quad_tag = DISCR_TAG_QUAD
    else:
        quad_tag = None

    dcoll = DiscretizationCollection(
        actx,
        mesh,
        discr_tag_to_group_factory={
            DISCR_TAG_BASE: default_simplex_group_factory(
                base_dim=mesh.dim, order=order
            ),
            DISCR_TAG_QUAD: QuadratureSimplexGroupFactory(2 * order),
        },
    )

    # }}}

    # {{{ Euler operator

    euler_operator = EulerOperator(
        dcoll,
        bdry_conditions={BTAG_ALL: InviscidWallBC()},
        flux_type="lf",
        gamma=gamma,
        quadrature_tag=quad_tag,
    )

    def rhs(t: float, q: ConservedEulerField) -> ConservedEulerField:
        return euler_operator.operator(  # pyright: ignore[reportUnknownMemberType, reportReturnType]
            actx, t, q
        )

    compiled_rhs = actx.compile(rhs)  # pyright: ignore[reportArgumentType]

    from grudge.dt_utils import h_min_from_volume

    cfl = 0.125
    cn = 0.5 * (order + 1) ** 2
    dt = cfl * actx.to_numpy(h_min_from_volume(dcoll)) / cn

    fields = acoustic_pulse_condition(actx.thaw(dcoll.nodes()))

    logger.info("Timestep size: %g", dt)

    # }}}

    t = np.float64(0.5)
    fields = actx.thaw(  # pyright: ignore[reportUnknownVariableType]
        actx.freeze(  # pyright: ignore[reportUnknownArgumentType]
            fields  # pyright: ignore[reportArgumentType]
        )
    )
    compiled_rhs(t, fields)  # pyright: ignore[reportUnknownArgumentType]


# vim: foldmethod=marker
