from arraycontext import PytatoPyOpenCLArrayContext


class DGFEMOptimizerArrayContext(PytatoPyOpenCLArrayContext):
    """
    An :class:`~arraycontext.ArrayContext` tuned for DG-FEM array expressions.
    It fuses the nodes in the expression DAG to the granularity of functional
    batched einsums, where none of the functional operands exhibit reductions.

    See paper (TODO) for details.
    """

    def transform_dag(self, dag):
        # TODO: Implement the transformations. here.
        return super().transform_dag(dag)

    def transform_loopy_program(self, t_unit):
        # TODO: Implement the transformations.
        return super().transform_loopy_program(t_unit)
