def get_actx_dgfem_suite_path() -> str:
    """
    Returns the absolute path for the install location of :mod:`actx_dgfem_suite`.
    """
    import importlib.util
    import os

    spec = importlib.util.find_spec("actx_dgfem_suite")
    assert spec is not None and spec.origin is not None
    module_path = os.path.abspath(os.path.join(spec.origin, os.path.pardir))
    assert os.path.isdir(module_path), module_path
    return os.path.abspath(module_path)


def get_nel_1d_for_regular_rect_mesh(dim: int, order: int, ndofs: int) -> int:
    from math import cbrt, ceil, sqrt

    if dim == 3:
        if order == 1:
            nel_1d = ceil(cbrt((ndofs / 4) / 6))
        elif order == 2:
            nel_1d = ceil(cbrt((ndofs / 10) / 6))
        elif order == 3:
            nel_1d = ceil(cbrt((ndofs / 20) / 6))
        elif order == 4:
            nel_1d = ceil(cbrt((ndofs / 35) / 6))
        else:
            raise NotImplementedError(order)
    elif dim == 2:
        if order == 1:
            nel_1d = ceil(sqrt((ndofs / 3) / 2))
        elif order == 2:
            nel_1d = ceil(sqrt((ndofs / 6) / 2))
        elif order == 3:
            nel_1d = ceil(sqrt((ndofs / 10) / 2))
        elif order == 4:
            nel_1d = ceil(sqrt((ndofs / 15) / 2))
        else:
            raise NotImplementedError(order)
    else:
        raise NotImplementedError

    return int(nel_1d)


def get_ndof_for_regular_rect_mesh(dim: int, order: int, nel_1d: int) -> int:
    if dim == 3:
        nel = 6 * nel_1d * nel_1d * nel_1d
        if order == 1:
            ndofs = nel * 6
        elif order == 2:
            ndofs = nel * 10
        elif order == 3:
            ndofs = nel * 20
        elif order == 4:
            ndofs = nel * 35
        else:
            raise NotImplementedError(order)
    elif dim == 2:
        nel = 2 * nel_1d * nel_1d
        if order == 1:
            ndofs = nel * 3
        elif order == 2:
            ndofs = nel * 6
        elif order == 3:
            ndofs = nel * 10
        elif order == 4:
            ndofs = nel * 15
        else:
            raise NotImplementedError(order)
    else:
        raise NotImplementedError

    return ndofs
