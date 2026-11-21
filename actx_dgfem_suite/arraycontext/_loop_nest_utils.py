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

import dataclasses as dc

import loopy as lp


@dc.dataclass(frozen=True, eq=True)
class LoopNest:
    inames: frozenset[str]
    insns_in_loop_nest: frozenset[str]


def is_a_perfect_loop_nest(kernel: lp.LoopKernel, inames: frozenset[str]) -> bool:
    try:
        template_iname = next(iter(inames))
    except StopIteration:
        return True
    else:
        insn_ids_in_template_iname = kernel.iname_to_insns()[template_iname]
        return all(
            kernel.iname_to_insns()[iname] == insn_ids_in_template_iname
            for iname in inames
        )


def get_loop_nest(kernel: lp.LoopKernel, insn: lp.InstructionBase) -> LoopNest:
    assert is_a_perfect_loop_nest(kernel, insn.within_inames)
    if insn.within_inames:
        any_iname_in_nest, *_other_inames = insn.within_inames
        return LoopNest(
            insn.within_inames, frozenset(kernel.iname_to_insns()[any_iname_in_nest])
        )
    else:
        if insn.reduction_inames():
            # TODO: Avoid O(N^2) complexity (typically there aren't long
            # kernels with "reduce-to-scalar" operations, but this might bite
            # in the future)
            insn_ids = frozenset(
                {
                    insn_.id
                    for insn_ in kernel.instructions
                    if insn_.reduction_inames() == insn.reduction_inames()
                }
            )
            return LoopNest(frozenset(), insn_ids)
        else:
            # We treat a loop nest with 0-depth in a special manner by putting
            # each such instruction into a separate loop nest.
            return LoopNest(frozenset(), frozenset([insn.id]))
