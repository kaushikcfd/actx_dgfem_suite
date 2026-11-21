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

import loopy as lp
import pyopencl as cl
import pytest
from loopy.kernel.tools import get_global_barrier_order

from actx_dgfem_suite.arraycontext import dgfem_optimizer
from actx_dgfem_suite.arraycontext.batched_einsum_loop_transforms import (
    transform_batched_einsum_loop_nests,
)
from actx_dgfem_suite.measure import instantiate_actx_t
from actx_dgfem_suite.rhs_builder import get_rhs


@pytest.mark.parametrize(
    ("equation", "expected_barriers"),
    [("tiny_wave", 2), ("tiny_maxwell", 1), ("tiny_euler", 1)],
)
def test_rhs_global_barriers_after_fusion(
    equation: str, expected_barriers: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    actx = instantiate_actx_t(dgfem_optimizer.DGFEMOptimizerArrayContext)
    rhs, args = get_rhs(equation, actx, dim=2, degree=2)

    # Device-specific scheduling can add further barriers inside a fused nest.
    # Inspect its input to count the barriers between fused nests.
    barrier_counts: list[int] = []

    def inspect_and_schedule(
        t_unit: lp.TranslationUnit, cl_device: cl.Device
    ) -> lp.TranslationUnit:
        barriers = get_global_barrier_order(t_unit.default_entrypoint)
        barrier_counts.append(len(barriers))
        return transform_batched_einsum_loop_nests(t_unit, cl_device)

    monkeypatch.setattr(
        dgfem_optimizer, "transform_batched_einsum_loop_nests", inspect_and_schedule
    )
    actx.compile(rhs)(*args)
    actx.queue.finish()

    assert barrier_counts == [expected_barriers]
