import loopy as lp

from actx_dgfem_suite.arraycontext.metadata import EinsumAxisTag


def _get_fusion_order_key(tag: EinsumAxisTag) -> tuple[int, str, str]:
    return (
        int(tag.index not in tag.ensm.out_idx_set),
        tag.ensm.get_subscripts(),
        tag.index,
    )


@lp.for_each_kernel
def apply_kennedy_loop_fusion_for_einsum_tags(
    kernel: lp.LoopKernel,
) -> lp.LoopKernel:
    einsum_axis_tag_to_iname: dict[EinsumAxisTag, set[str]] = {}
    for name, iname in kernel.inames.items():
        (tag,) = iname.tags_of_type(EinsumAxisTag)
        einsum_axis_tag_to_iname.setdefault(tag, set()).add(name)

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
    return kernel
