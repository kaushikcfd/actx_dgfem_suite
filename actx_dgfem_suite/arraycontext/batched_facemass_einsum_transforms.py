import itertools
from typing import Any

import feinsum as fnsm
import loopy as lp
import loopy.match as lp_match
import numpy as np
from pymbolic.typing import Integer


def _get_facemass_op_params(
    t_unit: lp.TranslationUnit,
    kernel_name: str | None = None,
    insn_match: Any | None = None,
) -> tuple[int, int, int, int, np.dtype[Any]]:
    from feinsum.einsum import INT_CLASSES, SizeParam

    batched_einsum, _ = fnsm.get_a_matched_einsum(
        t_unit,
        kernel_name=kernel_name,
        insn_match=insn_match,
        long_dim_length=36,
    )
    einsum = fnsm.einsum(batched_einsum.get_subscripts(), *batched_einsum.args[0])
    assert einsum.ndim == 2
    assert isinstance(einsum.shape[0], SizeParam)
    ni = einsum.shape[1]
    assert isinstance(ni, INT_CLASSES)
    assert len(einsum.args[0]) == 3

    assert (
        len([arg for arg in einsum.args[0] if arg.ndim == 2]) == 1
        and len([arg for arg in einsum.args[0] if arg.ndim == 3]) == 2
    )
    (jac,) = [arg for arg in einsum.args[0] if arg.ndim == 2]
    nf = jac.shape[0]
    nj = next(iter(arg for arg in einsum.args[0] if arg.ndim == 3)).shape[2]

    ref_einsum = fnsm.einsum(
        "ifj,fe,fej->ei",
        fnsm.array("M", (ni, nf, nj), jac.dtype),
        fnsm.array("J", (nf, "Ne"), jac.dtype),
        fnsm.array("u", (nf, "Ne", nj), jac.dtype),
    )

    assert fnsm.canonicalize_einsum(einsum) == fnsm.canonicalize_einsum(ref_einsum)
    assert (
        isinstance(nf, Integer)
        and isinstance(ni, Integer)
        and isinstance(nj, Integer)
    )
    return (batched_einsum.b, int(nf), int(ni), int(nj), jac.dtype)


def _get_read_field_variables(
    t_unit: lp.TranslationUnit, kernel_name: str | None, insn_id: str
) -> frozenset[str]:
    from loopy.symbolic import get_dependencies

    kernel_name = kernel_name or t_unit.default_entrypoint.name
    _, nf, ni, nj, dtype = _get_facemass_op_params(
        t_unit, kernel_name, lp_match.Id(insn_id)
    )
    ref_einsum = fnsm.einsum(
        "ifj,fe,fej->ei",
        fnsm.array("M", (ni, nf, nj), dtype),
        fnsm.array("J", (nf, "Ne"), dtype),
        fnsm.array("u", (nf, "Ne", nj), dtype),
    )
    sigma = fnsm.identify_as_einsum(
        t_unit,
        ref_einsum,
        kernel_name=kernel_name,
        insn_match=lp_match.Id(insn_id),
        long_dim_length=36,
    )
    expr = t_unit[kernel_name].substitutions[sigma["u"]].expression
    all_read_deps: set[str] = set()
    all_args = frozenset(t_unit[kernel_name].arg_dict) | frozenset(
        t_unit[kernel_name].temporary_variables
    )
    for dep in get_dependencies(expr):
        if (
            dep in all_args
            and not dep.startswith("normal")
            and isinstance(
                dep_dtype := t_unit[kernel_name].get_var_descriptor(dep).dtype,
                lp.LoopyType,
            )
            and dep_dtype.numpy_dtype == dtype
            and isinstance(
                dep_shape := t_unit[kernel_name].get_var_descriptor(dep).shape, tuple
            )
            and len(dep_shape) == 2
        ):
            all_read_deps.add(dep)

    return frozenset(all_read_deps)


def fe_out(i: int) -> str:
    if i == 0:
        return "_fe_out"

    return f"_fe_out_{i - 1}"


def transform_single_field_facemass_einsum(
    t_unit: lp.TranslationUnit,
    insn_match: Any | None = None,
    kernel_name: str | None = None,
) -> lp.TranslationUnit:
    import pymbolic.primitives as prim
    from feinsum.loopy_utils import (
        extract_multiplicative_terms_in_sum_reduction_as_subst,
    )

    within = lp_match.parse_match(insn_match)
    b, nf, ni, nj, dtype = _get_facemass_op_params(t_unit, kernel_name, within)

    ref_einsum = fnsm.batched_einsum(
        "ifj,fe,fej->ei",
        [
            [
                fnsm.array("M", (ni, nf, nj), dtype),
                fnsm.array("J", (nf, "Ne"), dtype),
                fnsm.array(f"u_{i}", (nf, "Ne", nj), dtype),
            ]
            for i in range(b)
        ],
    )

    sigma = fnsm.identify_as_einsum(
        t_unit,
        ref_einsum,
        kernel_name=kernel_name,
        insn_match=insn_match,
        long_dim_length=36,
    )

    i_iname = sigma["i"]
    e_iname = sigma["e"]
    j_iname = sigma["j"]
    f_iname = sigma["f"]
    M_subst = sigma["M"]
    outs = tuple(sigma[fe_out(i)] for i in range(b))

    kernel_name = kernel_name or t_unit.default_entrypoint.name
    vng = t_unit[kernel_name].get_var_name_generator()

    # Step 1: Split e -> (e_outer/g.0, e_inner/l.1); tag i -> l.0
    e_inner_iname = vng(e_iname + "_inner")
    e_outer_iname = vng(e_iname + "_outer")
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
    t_unit = lp.tag_inames(t_unit, {i_iname: "l.0"})

    # Step 2: For each batch k, extract J(f,e)*u_k(f,e,j) as subst _tmp_Mu_k.
    # Terms that don't depend on i_iname are J and u_k; M/LIFT depends on i.
    # All b instructions keep the SAME f,j inames (no duplication) so that
    # Steps 3 and 4 can exploit loop fusion across the b accumulations.
    from loopy.symbolic import get_dependencies
    knl = t_unit[kernel_name]
    mu_subst_names = tuple(vng(f"_tmp_Mu_{ib}") for ib in range(b))

    for ib in range(b):
        knl = extract_multiplicative_terms_in_sum_reduction_as_subst(
            knl,
            within=lp_match.Writes(outs[ib]),
            subst_name=mu_subst_names[ib],
            arguments=[
                prim.Variable(e_inner_iname),
                prim.Variable(f_iname),
                prim.Variable(j_iname),
            ],
            terms_filter=lambda t: i_iname not in get_dependencies(t),
        )

    t_unit = t_unit.with_kernel(knl)

    # Step 3: Precompute each _tmp_Mu_k into LOCAL memory [e_inner, f, j].
    # Shape: (p_NblockS, nf, nj)
    # Shared precompute_outer_inames across all b precomputes enables loop fusion
    iprcmpt_e, iprcmpt_f, iprcmpt_j = (
        vng("iprcmpt_e"),
        vng("iprcmpt_f"),
        vng("iprcmpt_j"),
    )
    for ib in range(b):
        t_unit = lp.precompute(  # pyright: ignore[reportUnknownMemberType]
            t_unit,
            mu_subst_names[ib],
            sweep_inames=[e_inner_iname, f_iname, j_iname],
            precompute_inames=[iprcmpt_e, iprcmpt_f, iprcmpt_j],
            precompute_outer_inames=frozenset({e_outer_iname}),
            temporary_address_space=lp.AddressSpace.LOCAL,
        )
    t_unit = lp.tag_inames(
        t_unit, {iprcmpt_e: "l.1", iprcmpt_f: "unr", iprcmpt_j: "l.0"}
    )

    # Step 4: Precompute M[i,f,j] as a scalar temp per (f,j) step.
    # With f,j in precompute_outer_inames (empty sweep), loopy creates one
    # scalar _tmp_M computed once per (f,j) iteration, reused across all b
    # accumulations in the fused loop
    t_unit = lp.precompute(  # pyright: ignore[reportUnknownMemberType]
        t_unit,
        M_subst,
        sweep_inames=[],
        precompute_outer_inames=frozenset(
            {e_outer_iname, e_inner_iname, i_iname, f_iname, j_iname}
        ),
        temporary_address_space=lp.AddressSpace.PRIVATE,
        within=within,
    )

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
    field_names_to_insn_ids: dict[frozenset[str], set[str]] = {}
    for insn_id in insn_ids_matched:
        field_names_to_insn_ids.setdefault(
            _get_read_field_variables(t_unit, kernel_name, insn_id), set()
        ).add(insn_id)

    insn_id_groups = tuple(
        frozenset(insn_ids)
        for _, insn_ids in sorted(
            field_names_to_insn_ids.items(), key=lambda kxv: sorted(kxv[0])
        )
    )
    for prev_insn_ids, next_insn_ids in itertools.pairwise(insn_id_groups):
        prev_within = lp_match.Or(tuple(lp_match.Id(id_) for id_ in prev_insn_ids))
        next_within = lp_match.Or(tuple(lp_match.Id(id_) for id_ in next_insn_ids))
        t_unit = lp.add_barrier(
            t_unit,
            prev_within,
            next_within,
            id_based_on="gbarrier_in_batched_facemass",
            within_inames=frozenset(),
        )
        a_next_insn = t_unit[kernel_name].id_to_insn[next(iter(next_insn_ids))]
        t_unit = lp.duplicate_inames(
            t_unit,
            a_next_insn.within_inames | a_next_insn.reduction_inames(),
            within=next_within,
        )
        a_next_insn = t_unit[kernel_name].id_to_insn[next(iter(next_insn_ids))]
        t_unit = decouple_domain(
            t_unit, a_next_insn.within_inames | a_next_insn.reduction_inames(), set()
        )
    for insn_ids in insn_id_groups:
        within = lp_match.Or(tuple(lp_match.Id(id_) for id_ in insn_ids))
        t_unit = transform_single_field_facemass_einsum(t_unit, within, kernel_name)

    return t_unit
