from __future__ import annotations

import string

import feinsum as fnsm
import loopy as lp
import loopy.match as lp_match
import numpy as np
from feinsum.diagnostics import EinsumTunitMatchError
from feinsum.einsum import SizeParam
from pytools.tag import Tag, tag_dataclass


@tag_dataclass
class EinsumAxisTag(Tag):
    ensm: fnsm.BatchedEinsum
    index: str

    @staticmethod
    def from_non_canon_form(ensm: fnsm.BatchedEinsum, index: str) -> EinsumAxisTag:
        from feinsum.canonicalization import (
            get_substitution_mapping_between_isomorphic_batched_einsums,
        )

        canon_ensm = fnsm.canonicalize_einsum(ensm)
        subst_map = get_substitution_mapping_between_isomorphic_batched_einsums(
            ensm, canon_ensm
        )
        return EinsumAxisTag(canon_ensm, subst_map[index])


def _get_fusion_order_key(tag: EinsumAxisTag) -> tuple[int, str, str]:
    return (
        int(tag.index not in tag.ensm.out_idx_set),
        tag.ensm.get_subscripts(),
        tag.index,
    )


def apply_kennedy_loop_fusion_for_einsum_tags(
    t_unit: lp.TranslationUnit,
) -> lp.TranslationUnit:
    kernel = t_unit.default_entrypoint
    einsum_axis_tag_to_iname: dict[EinsumAxisTag, set[str]] = {}
    for insn in kernel.instructions:
        if isinstance(insn, lp.BarrierInstruction):
            continue
        try:
            einsum, subst_map = fnsm.get_a_matched_einsum(
                t_unit, insn_match=lp_match.Id(insn.id)
            )
            inames_to_tag = insn.within_inames | insn.reduction_inames()
        except EinsumTunitMatchError:
            # Elementwise instructions may have no einsum operands to match.
            # Treat their output axes as an identity einsum (e.g. ij->ij).
            inames = tuple(sorted(insn.within_inames))
            indices = string.ascii_lowercase[: len(inames)]
            sizes: list[int | SizeParam] = []
            for iname, index in zip(inames, indices, strict=True):
                size = kernel.get_iname_bounds(iname).size
                sizes.append(
                    size
                    if isinstance(size, int) and size < 500
                    else SizeParam(index.upper())
                )
            einsum = fnsm.einsum(
                f"{indices}->{indices}",
                fnsm.array("arg", sizes, np.float64),
            )
            subst_map = dict(zip(inames, indices, strict=True))
            inames_to_tag = insn.within_inames

        for iname in inames_to_tag:
            tag = EinsumAxisTag.from_non_canon_form(einsum, subst_map[iname])
            einsum_axis_tag_to_iname.setdefault(tag, set()).add(iname)

    if not einsum_axis_tag_to_iname:
        raise NotImplementedError

    for tag, inames in sorted(
        einsum_axis_tag_to_iname.items(), key=lambda x: _get_fusion_order_key(x[0])
    ):
        kernel = lp.rename_inames_in_batch(
            kernel,
            lp.get_kennedy_unweighted_fusion_candidates(
                kernel,
                inames,
                prefix=tag.index,
            ),
        )
    return t_unit.with_kernel(kernel)
