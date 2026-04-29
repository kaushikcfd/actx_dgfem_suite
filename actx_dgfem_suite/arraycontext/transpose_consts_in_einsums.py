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
