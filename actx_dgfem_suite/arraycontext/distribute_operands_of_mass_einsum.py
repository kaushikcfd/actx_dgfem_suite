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


def apply_distributive_law_to_mass_inverse(
    expr: pt.transform.ArrayOrNamesTc,
) -> pt.transform.ArrayOrNamesTc:
    from pytato.transform.einsum_distributive_law import (
        DoDistribute,
        DoNotDistribute,
        EinsumDistributiveLawDescriptor,
        apply_distributive_property_to_einsums,
    )

    def how_to_distribute(expr: pt.Einsum) -> EinsumDistributiveLawDescriptor:
        if pt.analysis.is_einsum_similar_to_subscript(expr, "e,ij,ej->ei"):
            return DoDistribute(ioperand=2)
        else:
            return DoNotDistribute()

    return apply_distributive_property_to_einsums(expr, how_to_distribute)
