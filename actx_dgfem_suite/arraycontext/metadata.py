from __future__ import annotations

import feinsum as fnsm
from pytools.tag import Tag, tag_dataclass


@tag_dataclass
class IncomingEisumTag(Tag):
    """
    Records the incoming einsum for a subexpression in the computational graph.
    """

    einsum: fnsm.BatchedEinsum

    def __post_init__(self) -> None:
        assert self.einsum.b == 1


@tag_dataclass
class EinsumAxisTag(Tag):
    """
    Tag that is attached to an ouptut axis for an array that which is of the
    form ``y <- f(y1, ... yk, einsum)``, where ``f`` is a composition of
    elementwise operations.
    """

    ensm: fnsm.BatchedEinsum
    index: str

    def __post_init__(self):
        assert fnsm.canonicalize_einsum(self.ensm) == self.ensm
        assert self.index in self.ensm.all_indices

    @staticmethod
    def from_non_canon_form(ensm: fnsm.BatchedEinsum, index: str) -> EinsumAxisTag:
        from feinsum.canonicalization import (
            get_substitution_mapping_between_isomorphic_batched_einsums,
        )

        assert index in ensm.all_indices
        canon_ensm = fnsm.canonicalize_einsum(ensm)
        subst_map = get_substitution_mapping_between_isomorphic_batched_einsums(
            ensm, canon_ensm
        )
        return EinsumAxisTag(canon_ensm, subst_map[index])
