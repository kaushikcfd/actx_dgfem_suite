import tempfile

from actx_dgfem_suite.codegen import SuiteGeneratingArraycontext


def _get_suite_generating_actx():
    tempdir = tempfile.mkdtemp()

    return SuiteGeneratingArraycontext(
        main_file_path=f"{tempdir}/main.py",
        datawrappers_path=f"{tempdir}/datawrappers.npz",
        pickled_ref_input_args_path=f"{tempdir}/ref_input_args.npz",
        pickled_ref_output_path=f"{tempdir}/ref_output.npz",
    )


def test_array_returning_function():

    def f(x):
        return 2 * x

    actx = _get_suite_generating_actx()

    a = actx.np.zeros(10, "float64")
    actx.compile(f)(
        actx.thaw(actx.freeze(a + 42))
    )  # internally asserts that the result is correct


def test_array_container_returning_function():

    def f(x):
        from pytools.obj_array import new_1d

        return new_1d([2 * x, 3 * x, x**2])

    actx = _get_suite_generating_actx()

    a = actx.np.zeros(10, "float64")
    actx.compile(f)(
        actx.thaw(actx.freeze(a + 42))
    )  # internally asserts that the result is correct
