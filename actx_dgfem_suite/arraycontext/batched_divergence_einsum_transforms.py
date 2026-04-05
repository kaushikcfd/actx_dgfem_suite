import itertools
from typing import Any

import feinsum as fnsm
import loopy as lp
import loopy.match as lp_match
import numpy as np
from pymbolic.typing import Integer


def _get_divergenrce_op_params(
    t_unit: lp.TranslationUnit,
    kernel_name: str | None = None,
    insn_match: Any | None = None,
) -> tuple[int, int, int, int, np.dtype[Any]]:
    from feinsum.einsum import SizeParam

    batched_einsum, _ = fnsm.get_a_matched_einsum(
        t_unit,
        kernel_name=kernel_name,
        insn_match=insn_match,
        long_dim_length=36,
    )
    einsum = fnsm.einsum(batched_einsum.get_subscripts(), *batched_einsum.args[0])
    assert einsum.ndim == 2
    assert isinstance(einsum.shape[0], SizeParam)
    assert len(einsum.args[0]) == 3
    assert all(arg.ndim == 3 for arg in einsum.args[0])
    (mat,) = [
        arg
        for arg in einsum.args[0]
        if not any(isinstance(s, SizeParam) for s in arg.shape)
    ]
    nr, ni, nj = mat.shape
    nx = nr

    ref_einsum = fnsm.einsum(
        "xre,rij,xej->ei",
        fnsm.array("J", (nx, nr, "Ne"), mat.dtype),
        fnsm.array("M", (nr, ni, nj), mat.dtype),
        fnsm.array("u", (nx, "Ne", nj), mat.dtype),
    )
    assert fnsm.canonicalize_einsum(einsum) == fnsm.canonicalize_einsum(ref_einsum)
    assert (
        isinstance(nx, Integer)
        and isinstance(nr, Integer)
        and isinstance(ni, Integer)
        and isinstance(nj, Integer)
    )
    return (int(nx), int(nr), int(ni), int(nj), mat.dtype)


def transform_single_divergence_einsum(
    t_unit: lp.TranslationUnit,
    insn_match: Any | None = None,
    kernel_name: str | None = None,
) -> lp.TranslationUnit:
    import pymbolic.primitives as prim
    from feinsum.loopy_utils import (
        extract_multiplicative_terms_in_sum_reduction_as_subst,
        hoist_invariant_multiplicative_terms_in_sum_reduction,
    )
    from loopy.symbolic import Reduction

    within = lp_match.parse_match(insn_match)
    nx, nr, ni, nj, dtype = _get_divergenrce_op_params(t_unit, kernel_name, within)
    ref_einsum = fnsm.einsum(
        "xre,rij,xej->ei",
        fnsm.array("J", (nx, nr, "Ne"), dtype),
        fnsm.array("M", (nr, ni, nj), dtype),
        fnsm.array("u", (nx, "Ne", nj), dtype),
    )

    subst_map = fnsm.identify_as_einsum(
        t_unit,
        ref_einsum,
        kernel_name=kernel_name,
        insn_match=insn_match,
        long_dim_length=36,
    )

    i_iname = subst_map["i"]
    e_iname = subst_map["e"]
    j_iname = subst_map["j"]
    r_iname = subst_map["r"]
    x_iname = subst_map["x"]
    u_var = subst_map["u"]

    kernel_name = kernel_name or t_unit.default_entrypoint.name
    vng = t_unit[kernel_name].get_var_name_generator()
    ing = t_unit[kernel_name].get_instruction_id_generator()
    e_inner_iname = vng(e_iname + "_inner")
    e_outer_iname = vng(e_iname + "_outer")

    # work groups of 4 elements (matching p_NblockV=4)
    t_unit = lp.split_iname(
        t_unit,
        e_iname,
        4,
        outer_tag="g.0",
        inner_tag="l.1",
        inner_iname=e_inner_iname,
        outer_iname=e_outer_iname,
        within=within,
        slabs=(0, 1),
    )

    # one thread per output node; unroll the two small loops (size 3 each)
    t_unit = lp.tag_inames(t_unit, {i_iname: "l.0", x_iname: "unr", r_iname: "unr"})

    # Precompute u[x, e, j] into LOCAL memory
    iprcmpt_x, iprcmpt_e, iprcmpt_j = (
        vng("iprcmpt_x"),
        vng("iprcmpt_e"),
        vng("iprcmpt_j"),
    )
    t_unit = lp.precompute(  # pyright: ignore[reportUnknownMemberType]
        t_unit,
        u_var,
        sweep_inames=[x_iname, e_inner_iname, j_iname],
        precompute_inames=[iprcmpt_x, iprcmpt_e, iprcmpt_j],
        temporary_address_space=lp.AddressSpace.LOCAL,
        within=within,
    )
    t_unit = lp.tag_inames(
        t_unit, {iprcmpt_x: "unr", iprcmpt_e: "l.1", iprcmpt_j: "l.0"}
    )

    # Extract terms corresponding to "D * u" into a subst.
    t_unit = lp.split_reduction_outward(t_unit, {x_iname, r_iname}, within=within)

    knl = t_unit[kernel_name]
    knl = hoist_invariant_multiplicative_terms_in_sum_reduction(
        knl, j_iname, within=within
    )
    du_subst_name = vng("_tmp_Du")
    knl = extract_multiplicative_terms_in_sum_reduction_as_subst(
        knl,
        within=within,
        subst_name=du_subst_name,
        arguments=[prim.Variable(x_iname), prim.Variable(r_iname)],
        terms_filter=lambda t: isinstance(t, Reduction),
    )
    t_unit = t_unit.with_kernel(knl)

    # Step 4: precompute _tmp_Du into a private tmp[x,r] array (9 unrolled scalars).
    # i and e_inner/e_outer are outer — each work-item has its own copy.
    prcmpt_x, prcmpt_r = vng("_tmp_Du_x"), vng("_tmp_Du_r")
    prcmpt_Du_id = ing("pcmpt_tmp_Du_id")
    t_unit = lp.precompute(  # pyright: ignore[reportUnknownMemberType]
        t_unit,
        du_subst_name,
        sweep_inames=[x_iname, r_iname],
        precompute_outer_inames=frozenset({e_outer_iname, e_inner_iname, i_iname}),
        precompute_inames=[prcmpt_x, prcmpt_r],
        temporary_address_space=lp.AddressSpace.PRIVATE,
        compute_insn_id=prcmpt_Du_id,
        default_tag="unr",
        within=within,
    )
    t_unit = lp.realize_reduction(t_unit, insn_id_filter=prcmpt_Du_id)
    (acc_name,) = (
        t_unit[kernel_name].id_to_insn[prcmpt_Du_id].read_dependency_names()
        - t_unit[kernel_name].all_inames()
    )
    (acc_j_init_id,) = [
        insn.id
        for insn in t_unit[kernel_name].instructions
        if (
            acc_name in insn.write_dependency_names()
            and acc_name not in insn.read_dependency_names()
        )
    ]
    (acc_j_assign_id,) = [
        insn.id
        for insn in t_unit[kernel_name].instructions
        if (
            acc_name in insn.read_dependency_names()
            and acc_name not in insn.write_dependency_names()
        )
    ]
    t_unit = lp.privatize_temporaries_with_inames(
        t_unit, frozenset([prcmpt_r, prcmpt_x]), acc_name
    )
    t_unit = lp.duplicate_inames(
        t_unit,
        [prcmpt_x, prcmpt_r],
        within=lp_match.Id(acc_j_init_id),
        tags={
            prcmpt_x: "unr",
            prcmpt_r: "unr",
        },
    )
    t_unit = lp.duplicate_inames(
        t_unit,
        [prcmpt_x, prcmpt_r],
        within=lp_match.Id(acc_j_assign_id),
        tags={
            prcmpt_x: "unr",
            prcmpt_r: "unr",
        },
    )
    t_unit = lp.prioritize_loops(t_unit, (j_iname, prcmpt_x, prcmpt_r))

    return t_unit


def transform(
    t_unit: lp.TranslationUnit,
    insn_match: Any | None = None,
    kernel_name: str | None = None,
) -> lp.TranslationUnit:
    from feinsum.loopy_utils import decouple_domain

    kernel_name = kernel_name or t_unit.default_entrypoint.name
    within = lp_match.parse_match(insn_match)
    insn_ids_matched = tuple(
        insn.id
        for insn in t_unit[kernel_name].instructions
        if within(t_unit[kernel_name], insn)
    )
    # treat each divergences as separate.
    for prev_insn_id, next_insn_id in itertools.pairwise(insn_ids_matched):
        t_unit = lp.add_barrier(
            t_unit,
            lp_match.Id(prev_insn_id),
            lp_match.Id(next_insn_id),
            id_based_on="gbarrier_in_batched_div",
            within_inames=frozenset(),
        )
        next_insn = t_unit[kernel_name].id_to_insn[next_insn_id]
        t_unit = lp.duplicate_inames(
            t_unit,
            next_insn.within_inames | next_insn.reduction_inames(),
            within=lp_match.Id(next_insn_id),
        )
        next_insn = t_unit[kernel_name].id_to_insn[next_insn_id]
        t_unit = decouple_domain(
            t_unit, next_insn.within_inames | next_insn.reduction_inames(), set()
        )
    for insn_id in insn_ids_matched:
        t_unit = transform_single_divergence_einsum(
            t_unit, lp_match.Id(insn_id), kernel_name
        )

    return t_unit
