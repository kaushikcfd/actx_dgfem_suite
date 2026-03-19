__copyright__ = """
Copyright (C) 2015 Andreas Kloeckner
Copyright (C) 2021 University of Illinois Board of Trustees
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
from arraycontext import ArrayContext
from grudge.discretization import make_discretization_collection
from grudge.models.em import MaxwellOperator, Vector, get_rectangular_cavity_mode
from meshmode.dof_array import DOFArray as _DOFArray
from meshmode.mesh.generation import generate_regular_rect_mesh
from pytools.obj_array import ObjectArray

from actx_dgfem_suite.utils import get_nel_1d_for_regular_rect_mesh

logger = logging.getLogger(__name__)


def main(actx: ArrayContext, dim: int, order: int, ndofs: int):
    nel_1d = get_nel_1d_for_regular_rect_mesh(dim, order, ndofs)
    mesh = generate_regular_rect_mesh(
        a=(0.0,) * dim, b=(1.0,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    dcoll = make_discretization_collection(actx, mesh, order=order)

    epsilon = 1
    mu = 1

    maxwell_operator = MaxwellOperator(
        dcoll, epsilon, mu, flux_type=0.5, dimensions=dim
    )

    def cavity_mode(x: ObjectArray[tuple[int], _DOFArray], t: float):
        if dim == 3:
            return get_rectangular_cavity_mode(actx, x, t, 1, (1, 2, 2))
        else:
            return get_rectangular_cavity_mode(actx, x, t, 1, (2, 3))

    fields = cavity_mode(actx.thaw(dcoll.nodes()), 0)

    maxwell_operator.check_bc_coverage(mesh)

    def rhs(t: float, w: Vector) -> Vector:
        return maxwell_operator.operator(t, w)

    compiled_rhs = actx.compile(rhs)

    fields = actx.freeze_thaw(fields)
    compiled_rhs(np.float64(0.5), fields)
