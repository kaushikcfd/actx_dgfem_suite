import loopy as lp
import pytato as pt
from arraycontext import PytatoPyOpenCLArrayContext

from actx_dgfem_suite.arraycontext.mass_inverse_fuser import fuse_mass_inverses
from actx_dgfem_suite.arraycontext.materialization_policy import (
    materialize_for_dgfem_opt,
)
from actx_dgfem_suite.arraycontext.push_einsum_indices import (
    push_einsum_indices_to_operands,
)


def apply_distributive_law_to_mass_inverse(
    expr: pt.DictOfNamedArrays,
) -> pt.AbstractResultWithNamedArrays:
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

    return pt.make_dict_of_named_arrays(
        {
            name: apply_distributive_property_to_einsums(subexpr, how_to_distribute)
            for name, subexpr in expr._data.items()
        }
    )


class DGFEMOptimizerArrayContext(PytatoPyOpenCLArrayContext):
    """
    An :class:`~arraycontext.ArrayContext` tuned for DG-FEM array expressions.
    It fuses the nodes in the expression DAG to the granularity of functional
    batched einsums, where none of the functional operands exhibit reductions.

    See paper (TODO) for details.
    """

    def transform_dag(
        self, dag: pt.AbstractResultWithNamedArrays
    ) -> pt.AbstractResultWithNamedArrays:
        if pt.analysis.get_num_nodes(dag) < 10:
            # FIXME: This is only for debugging purposes, remove this once
            # everything is finalized.
            return super().transform_dag(dag)

        dag = apply_distributive_law_to_mass_inverse(dag)
        dag = push_einsum_indices_to_operands(dag)
        dag = fuse_mass_inverses(dag)
        dag = materialize_for_dgfem_opt(dag)
        dag = pt.push_index_to_materialized_nodes(dag)

        return dag

    def transform_loopy_program(
        self, t_unit: lp.TranslationUnit
    ) -> lp.TranslationUnit:
        # TODO: Implement the transformations.
        return super().transform_loopy_program(t_unit)
