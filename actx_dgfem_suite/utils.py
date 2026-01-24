from dataclasses import is_dataclass

from arraycontext import is_array_container_type


def get_actx_dgfem_suite_path() -> str:
    """
    Returns the absolute path for the install location of :mod:`actx_dgfem_suite`.
    """
    import importlib.util
    import os

    module_path = os.path.abspath(
        os.path.join(
            importlib.util.find_spec("actx_dgfem_suite").origin, os.path.pardir
        )
    )
    assert os.path.isdir(module_path), module_path
    return os.path.abspath(module_path)


def _get_benchmark_directory(equation: str, dim: int, degree: int) -> str:
    import os

    dir_path = os.path.join(
        get_actx_dgfem_suite_path(), "suite", f"{equation}_{dim}D_P{degree}"
    )

    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    assert os.path.isdir(dir_path)
    return dir_path


def get_benchmark_main_file_path(equation: str, dim: int, degree: int) -> str:
    import os

    return os.path.join(_get_benchmark_directory(equation, dim, degree), "main.py")


def get_benchmark_literals_path(equation: str, dim: int, degree: int) -> str:
    import os

    return os.path.join(
        _get_benchmark_directory(equation, dim, degree), "literals.npz"
    )


def get_benchmark_ref_input_arguments_path(
    equation: str, dim: int, degree: int
) -> str:
    import os

    return os.path.join(
        _get_benchmark_directory(equation, dim, degree), "ref_input_args.pkl"
    )


def get_benchmark_ref_output_path(equation: str, dim: int, degree: int) -> str:
    import os

    return os.path.join(
        _get_benchmark_directory(equation, dim, degree), "ref_outputs.pkl"
    )


def get_benchmark_rhs_invoker(equation: str, dim: int, degree: int):
    import importlib.util
    import os

    spec = importlib.util.spec_from_file_location(
        "main",
        os.path.join(_get_benchmark_directory(equation, dim, degree), "main.py"),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot find benchmark for equation '{equation}'" f"-{dim}D-P{degree}"
        )

    benchmark_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark_module)
    rhs_invoker = benchmark_module.RHSInvoker

    return rhs_invoker


def is_dataclass_array_container(ary) -> bool:
    from meshmode.dof_array import DOFArray

    return (
        is_array_container_type(ary.__class__) and is_dataclass(ary)
    ) or isinstance(ary, DOFArray)
