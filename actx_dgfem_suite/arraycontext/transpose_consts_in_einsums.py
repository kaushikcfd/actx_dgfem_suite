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
from arraycontext import PytatoPyOpenCLArrayContext


def _transpose_lift_matrix_rec(
    expr: pt.transform.ArrayOrNames, actx: PytatoPyOpenCLArrayContext
) -> pt.transform.ArrayOrNames:
    if isinstance(expr, pt.Einsum) and pt.analysis.is_einsum_similar_to_subscript(
        expr, "ifj,fe,fej->ei"
    ):
        old_lift = expr.args[0]
        assert isinstance(old_lift, pt.Array)
        new_lift = actx.freeze_thaw(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
            pt.transpose(old_lift, [2, 1, 0])
        )
        assert isinstance(new_lift, pt.DataWrapper)
        return pt.einsum(
            "jfi,fe,fej->ei", new_lift, expr.args[1], expr.args[2]
        ).replace_if_different(
            tags=expr.tags,
            axes=expr.axes,
            redn_axis_to_redn_descr=expr.redn_axis_to_redn_descr,
        )
    else:
        return expr


def transpose_lift_matrix_in_facemass(
    expr: pt.transform.ArrayOrNamesTc, comptime_actx: PytatoPyOpenCLArrayContext
) -> pt.transform.ArrayOrNamesTc:

    return pt.transform.map_and_copy(
        expr,
        lambda expr: _transpose_lift_matrix_rec(expr, actx=comptime_actx),
    )


def _transpose_deriv_matrix_rec(
    expr: pt.transform.ArrayOrNames, actx: PytatoPyOpenCLArrayContext
) -> pt.transform.ArrayOrNames:
    if isinstance(expr, pt.Einsum) and pt.analysis.is_einsum_similar_to_subscript(
        expr, "re,rij,ej->ei"
    ):
        old_deriv = expr.args[1]
        assert isinstance(old_deriv, pt.Array)
        new_deriv = actx.freeze_thaw(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
            pt.transpose(old_deriv, [0, 2, 1])
        )
        assert isinstance(new_deriv, pt.DataWrapper)
        return pt.einsum(
            "re,rji,ej->ei", expr.args[0], new_deriv, expr.args[2]
        ).replace_if_different(
            tags=expr.tags,
            axes=expr.axes,
            redn_axis_to_redn_descr=expr.redn_axis_to_redn_descr,
        )
    elif isinstance(expr, pt.Einsum) and pt.analysis.is_einsum_similar_to_subscript(
        expr, "xre,rij,xej->ei"
    ):
        old_deriv = expr.args[1]
        assert isinstance(old_deriv, pt.Array)
        new_deriv = actx.freeze_thaw(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
            pt.transpose(old_deriv, [0, 2, 1])
        )
        assert isinstance(new_deriv, pt.DataWrapper)
        return pt.einsum(
            "xre,rji,xej->ei", expr.args[0], new_deriv, expr.args[2]
        ).replace_if_different(
            tags=expr.tags,
            axes=expr.axes,
            redn_axis_to_redn_descr=expr.redn_axis_to_redn_descr,
        )
    else:
        return expr


def transpose_deriv_matrix_in_grad_and_div(
    expr: pt.transform.ArrayOrNamesTc, comptime_actx: PytatoPyOpenCLArrayContext
) -> pt.transform.ArrayOrNamesTc:

    return pt.transform.map_and_copy(
        expr,
        lambda expr: _transpose_deriv_matrix_rec(expr, actx=comptime_actx),
    )
