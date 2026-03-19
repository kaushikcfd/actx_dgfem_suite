import pytato as pt


def apply_distributive_law_to_mass_inverse(
    expr: pt.DictOfNamedArrays,
) -> pt.AbstractResultWithNamedArrays:
    from pytato.transform.einsum_distributive_law import (
        DoDistribute,
        DoNotDistribute,
        EinsumDistributiveLawDescriptor,
        apply_distributive_property_to_einsums,
    )

    def how_to_distribute(expr: pt.Array) -> EinsumDistributiveLawDescriptor:
        assert isinstance(expr, pt.Einsum)
        if pt.analysis.is_einsum_similar_to_subscript(expr, "e,ij,ej->ei"):
            return DoDistribute(ioperand=2)
        else:
            return DoNotDistribute()

    return apply_distributive_property_to_einsums(expr, how_to_distribute)
