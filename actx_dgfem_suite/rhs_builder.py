"""
Dispatch helper that builds (rhs, call_args) for each supported equation.
"""

from typing import Any

from arraycontext import ArrayContext

_NDOFS_FULL = 4_000_000
_NDOFS_TINY = 1_000
_NDOFS_LARGE = 20_000_000


def get_rhs(
    equation: str, actx: ArrayContext, dim: int, degree: int
) -> tuple[Any, tuple[Any, ...]]:
    """
    Return *(rhs_callable, call_args)* for *equation*.

    *call_args* is a tuple suitable for ``rhs_callable(*call_args)``.
    """
    if equation.startswith("tiny_"):
        ndofs = _NDOFS_TINY
        base = equation.removeprefix("tiny_")
    elif equation.startswith("large_"):
        ndofs = _NDOFS_LARGE
        base = equation.removeprefix("large_")
    else:
        ndofs = _NDOFS_FULL
        base = equation

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
