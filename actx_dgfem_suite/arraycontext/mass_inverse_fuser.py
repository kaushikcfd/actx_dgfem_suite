from typing import TYPE_CHECKING, cast

import pytato as pt
from pytato.array import NormalizedSlice, ShapeComponent
from pytato.scalar_expr import INT_CLASSES
from pytato.transform import (
    ArrayOrNamesTc,
    CopyMapper,
    _verify_is_array,
)

if TYPE_CHECKING:
    from pymbolic.typing import Integer


class MassInverseFuser(CopyMapper):
    def map_einsum(self, expr: pt.Einsum) -> pt.Einsum:
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
                pt.einsum("ik,kfj->ifj", DMinv, Dface),
                JMinv * Jface,
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
                JMinv * Jdiv,
                pt.einsum("ik,rkj->rij", DMinv, Ddiv),
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
                JMinv * Jgrad,
                pt.einsum("ik,rkj->rij", DMinv, Dgrad),
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
    return cast("ArrayOrNamesTc", mapper(expr))
