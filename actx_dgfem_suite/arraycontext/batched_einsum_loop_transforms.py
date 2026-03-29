import feinsum as fnsm
import loopy as lp
import loopy.match as lp_match
import pyopencl as cl
from feinsum.diagnostics import NoFactInDatabaseError


def _is_facemass_einsum(batched_einsum: fnsm.BatchedEinsum) -> bool:
    from feinsum.einsum import INT_CLASSES, SizeParam

    einsum = fnsm.einsum(batched_einsum.get_subscripts(), *batched_einsum.args[0])
    if einsum.ndim != 2:
        return False
    if not isinstance(einsum.shape[0], SizeParam):
        return False
    ni = einsum.shape[1]
    if not isinstance(ni, INT_CLASSES):
        return False
    if len(einsum.args[0]) != 3:
        return False

    if (
        len([arg for arg in einsum.args[0] if arg.ndim == 2]) != 1
        or len([arg for arg in einsum.args[0] if arg.ndim == 3]) != 2
    ):
        return False
    (jac,) = [arg for arg in einsum.args[0] if arg.ndim == 2]
    nf = jac.shape[0]
    nj = next(iter(arg for arg in einsum.args[0] if arg.ndim == 3)).shape[2]

    return fnsm.canonicalize_einsum(einsum) == fnsm.canonicalize_einsum(
        fnsm.einsum(
            "ifj,fe,fej->ei",
            fnsm.array("M", (ni, nf, nj), jac.dtype),
            fnsm.array("J", (nf, "Ne"), jac.dtype),
            fnsm.array("u", (nf, "Ne", nj), jac.dtype),
        )
    )


def _is_derivative_einsum(batched_einsum: fnsm.BatchedEinsum) -> bool:
    from feinsum.einsum import INT_CLASSES, SizeParam

    einsum = fnsm.einsum(batched_einsum.get_subscripts(), *batched_einsum.args[0])
    if einsum.ndim != 2:
        return False
    if not isinstance(einsum.shape[0], SizeParam):
        return False
    ni = einsum.shape[1]
    if not isinstance(ni, INT_CLASSES):
        return False
    if len(einsum.args[0]) != 3:
        return False

    if (
        len([arg for arg in einsum.args[0] if arg.ndim == 2]) != 2
        or len([arg for arg in einsum.args[0] if arg.ndim == 3]) != 1
    ):
        return False
    (mat,) = [arg for arg in einsum.args[0] if arg.ndim == 3]
    nr = mat.shape[0]
    nj = mat.shape[2]

    return fnsm.canonicalize_einsum(einsum) == fnsm.canonicalize_einsum(
        fnsm.einsum(
            "re,rij,ej->ei",
            fnsm.array("J", (nr, "Ne"), mat.dtype),
            fnsm.array("M", (nr, ni, nj), mat.dtype),
            fnsm.array("u", ("Ne", nj), mat.dtype),
        )
    )


def _is_divergence_einsum(batched_einsum: fnsm.BatchedEinsum) -> bool:
    from feinsum.einsum import SizeParam

    einsum = fnsm.einsum(batched_einsum.get_subscripts(), *batched_einsum.args[0])
    if einsum.ndim != 2:
        return False
    if not isinstance(einsum.shape[0], SizeParam):
        return False
    if len(einsum.args[0]) != 3:
        return False
    if len([arg for arg in einsum.args[0] if arg.ndim == 3]) != 3:
        return False
    (mat,) = [
        arg
        for arg in einsum.args[0]
        if not any(isinstance(s, SizeParam) for s in arg.shape)
    ]
    nr, ni, nj = mat.shape
    nx = nr

    return fnsm.canonicalize_einsum(einsum) == fnsm.canonicalize_einsum(
        fnsm.einsum(
            "xre,rij,xej->ei",
            fnsm.array("J", (nx, nr, "Ne"), mat.dtype),
            fnsm.array("M", (nr, ni, nj), mat.dtype),
            fnsm.array("u", (nx, "Ne", nj), mat.dtype),
        )
    )


@lp.for_each_kernel
def _merge_domains_for_potential_loop_nests(kernel: lp.LoopKernel) -> lp.LoopKernel:
    import islpy as isl
    from feinsum.loopy_utils import decouple_domain

    all_inames = kernel.all_inames()
    for iname in all_inames:
        hdi = kernel.get_home_domain_index(iname)
        assert kernel.domains[hdi].dim(isl.dim_type.set) > 0
        if kernel.domains[hdi].dim(isl.dim_type.set) == 1:
            continue
        kernel = decouple_domain(kernel, (iname,), ())

    new_domains: list[isl.BasicSet] = []
    loop_nests = frozenset(
        insn.within_inames | insn.reduction_inames()
        for insn in kernel.instructions
        if not isinstance(insn, lp.BarrierInstruction)
    )
    for loop_nest in sorted(loop_nests, key=sorted):
        new_domains.append(
            kernel.combine_domains(
                tuple(
                    kernel.get_home_domain_index(iname)
                    for iname in sorted(loop_nest)
                )
            )
        )
    return kernel.copy(domains=tuple(new_domains))


def transform_batched_einsum_loop_nests(
    t_unit: lp.TranslationUnit, cl_device: cl.Device
) -> lp.TranslationUnit:
    t_unit = _merge_domains_for_potential_loop_nests(t_unit)

    loop_nests = frozenset(
        insn.within_inames
        for insn in t_unit.default_entrypoint.instructions
        if not isinstance(insn, lp.BarrierInstruction)
    )
    for loop_nest in sorted(loop_nests, key=sorted):
        within = lp_match.And(tuple(lp_match.Iname(iname) for iname in loop_nest))
        batched_einsum, _ = fnsm.get_a_matched_einsum(
            t_unit,
            insn_match=within,
            long_dim_length=36,
        )
        try:
            transform = fnsm.retrieve(batched_einsum, cl_device)
        except NoFactInDatabaseError as err:
            if _is_facemass_einsum(batched_einsum):
                from .batched_facemass_einsum_transforms import transform

                t_unit = transform(t_unit, insn_match=within)
            elif _is_derivative_einsum(batched_einsum):
                from .batched_derivative_einsum_transforms import transform

                t_unit = transform(t_unit, insn_match=within)
            elif _is_divergence_einsum(batched_einsum):
                from .batched_divergence_einsum_transforms import transform

                t_unit = transform(t_unit, insn_match=within)
            else:
                raise NotImplementedError(f"{batched_einsum}") from err
        else:
            t_unit = transform(t_unit, insn_match=within)

    return t_unit
