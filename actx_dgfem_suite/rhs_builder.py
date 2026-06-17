"""
Dispatch helper that builds (rhs, call_args) for each supported equation.
"""

from typing import Any

from arraycontext import ArrayContext

_NDOFS_FULL = 3_000_000
_NDOFS_TINY = 1_000


def get_rhs(
    equation: str, actx: ArrayContext, dim: int, degree: int
) -> tuple[Any, tuple[Any, ...]]:
    """
    Return *(rhs_callable, call_args)* for *equation*.

    *call_args* is a tuple suitable for ``rhs_callable(*call_args)``.
    """
    ndofs = _NDOFS_TINY if equation.startswith("tiny_") else _NDOFS_FULL
    base = equation.removeprefix("tiny_")

    if base == "wave":
        from actx_dgfem_suite.equations.wave import get_wave_rhs

        return get_wave_rhs(actx=actx, dim=dim, order=degree, ndofs=ndofs)
    elif base == "euler":
        from actx_dgfem_suite.equations.euler import get_euler_rhs

        return get_euler_rhs(actx=actx, dim=dim, order=degree, ndofs=ndofs)
    elif base == "maxwell":
        from actx_dgfem_suite.equations.maxwell import get_maxwell_rhs

        return get_maxwell_rhs(actx=actx, dim=dim, order=degree, ndofs=ndofs)
    else:
        raise NotImplementedError(f"equation '{equation}' not supported")
