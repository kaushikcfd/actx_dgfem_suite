__copyright__ = """
Copyright (C) 2026 Kaushik Kulkarni
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

import pytato as pt
from pytato.transform import (
    ArrayOrNamesTc,
    CopyMapper,
)
from pytools import memoize_method
from typing_extensions import override


class MassInverseFuser(CopyMapper):
    @memoize_method
    def memoized_mult(self, x1: pt.Array, x2: pt.Array) -> pt.Array:
        return x1 * x2

    @memoize_method
    def memoized_einsum(self, subscripts: str, *operands: pt.Array) -> pt.Array:
        return pt.einsum(subscripts, *operands)

    @override
    def map_einsum(self, expr: pt.Einsum) -> pt.Array:
        if (
            pt.analysis.is_einsum_similar_to_subscript(expr, "e,ik,ek->ei")
            and isinstance(expr.args[2], pt.Einsum)
            and pt.analysis.is_einsum_similar_to_subscript(
                expr.args[2], "kfj,fe,fej->ek"
            )
        ):
            # Fuse mass inverse and face mass
            JMinv = expr.args[0]
            DMinv = expr.args[1]
            Dface, Jface, uface = expr.args[2].args

            return pt.einsum(
                "ifj,fe,fej->ei",
                self.memoized_einsum("ik,kfj->ifj", DMinv, Dface),
                self.memoized_mult(JMinv, Jface),
                uface,
            )
        elif (
            pt.analysis.is_einsum_similar_to_subscript(expr, "e,ik,ek->ei")
            and isinstance(expr.args[2], pt.Einsum)
            and pt.analysis.is_einsum_similar_to_subscript(
                expr.args[2], "xre,rkj,xej->ek"
            )
        ):
            # Fuse mass inverse and divergence.
            JMinv = expr.args[0]
            DMinv = expr.args[1]
            Jdiv, Ddiv, udiv = expr.args[2].args

            return pt.einsum(
                "xre,rij,xej->ei",
                self.memoized_mult(JMinv, Jdiv),
                self.memoized_einsum("ik,rkj->rij", DMinv, Ddiv),
                udiv,
            )
        elif (
            pt.analysis.is_einsum_similar_to_subscript(expr, "e,ik,ek->ei")
            and isinstance(expr.args[2], pt.Einsum)
            and pt.analysis.is_einsum_similar_to_subscript(
                expr.args[2], "re,rkj,ej->ek"
            )
        ):
            # Fuse mass inverse and a component of grad.
            JMinv = expr.args[0]
            DMinv = expr.args[1]
            Jgrad, Dgrad, ugrad = expr.args[2].args

            return pt.einsum(
                "re,rij,ej->ei",
                self.memoized_mult(JMinv, Jgrad),
                self.memoized_einsum("ik,rkj->rij", DMinv, Dgrad),
                ugrad,
            )
        else:
            return super().map_einsum(expr)


def fuse_mass_inverses(expr: ArrayOrNamesTc) -> ArrayOrNamesTc:
    r"""
    Fuses the composition inverse mass matrix operator and a finite-element
    operator into a single einsum. Specifically, this rewrites the einsums
    corresponding to the composition
    :math:`M^{-1}\left(\operatorname{FEMOp}\left(u\right)\right)` as a single
    einsum. Here, :math:`\operatorname{FEMOp}` denotes a finite-element operator
    implemented as an einsum, corresponding to either the face mass, divergence,
    or gradient operator.
    """
    mapper = MassInverseFuser()
    return mapper(expr)
