from importlib import import_module

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
CuPyArrayContext = import_module(
    "actx_dgfem_suite.arraycontext.cupy_actx"
).CuPyArrayContext


def test_cupy_array_context_round_trip():
    actx = CuPyArrayContext()
    host_ary = np.arange(6, dtype=np.float64).reshape(2, 3)

    device_ary = actx.from_numpy(host_ary)

    assert isinstance(device_ary, cp.ndarray)
    np.testing.assert_array_equal(actx.to_numpy(device_ary), host_ary)


def test_cupy_array_context_fake_numpy():
    actx = CuPyArrayContext()
    ary = actx.from_numpy(np.array([[1, 2], [3, 4]], dtype=np.int64))

    np.testing.assert_array_equal(
        actx.to_numpy(actx.np.ravel(ary)), np.array([1, 2, 3, 4])
    )
    np.testing.assert_array_equal(
        actx.to_numpy(actx.np.broadcast_to(ary[0], (2, 2))),
        np.array([[1, 2], [1, 2]]),
    )
    assert actx.to_numpy(actx.np.sum(ary)) == 10
    assert bool(actx.to_numpy(actx.np.array_equal(ary, ary.copy())))


def test_cupy_array_context_stack_and_where():
    actx = CuPyArrayContext()
    ary = actx.from_numpy(np.array([1, 2, 3]))

    stacked = actx.np.stack([ary, 2 * ary])
    selected = actx.np.where(ary > 1, ary, 0 * ary)

    np.testing.assert_array_equal(
        actx.to_numpy(stacked), np.array([[1, 2, 3], [2, 4, 6]])
    )
    np.testing.assert_array_equal(actx.to_numpy(selected), np.array([0, 2, 3]))
