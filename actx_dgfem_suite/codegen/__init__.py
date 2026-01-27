r"""
Helpers to generate python code compatible with any
:class:`~arraycontext.ArrayContext`\ 's compile method.

.. autoclass:: SuiteGeneratingArraycontext
"""

import ast
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import autoflake
import black
import numpy as np
import pytato as pt
from arraycontext import is_array_container_type
from arraycontext.container.traversal import (
    rec_keyed_map_array_container,
    rec_map_array_container,
    rec_multimap_array_container,
)
from arraycontext.impl.pytato.compile import (
    BaseLazilyCompilingFunctionCaller,
    CompiledFunction,
)
from meshmode.dof_array import array_context_for_pickling
from pytools import memoize_method

from actx_dgfem_suite.arraycontext import DGFEMOptimizerArrayContext
from actx_dgfem_suite.utils import (
    get_actx_dgfem_suite_path,
    is_dataclass_array_container,
)


def remove_tags_with_typenames(
    expr: pt.DictOfNamedArrays, names_to_remove: frozenset[str]
) -> pt.DictOfNamedArrays:
    def map_fn(subexpr: pt.transform.ArrayOrNames) -> pt.transform.ArrayOrNames:
        if isinstance(subexpr, pt.Array):
            new_tags = frozenset(
                [
                    tag
                    for tag in subexpr.tags
                    if tag.__class__.__name__ not in names_to_remove
                ]
            )
            return subexpr.copy(tags=new_tags)
        else:
            return subexpr

    return pt.transform.map_and_copy(expr, map_fn)


BAD_TAG_TYPENAMES = frozenset(["NameHint", "FEMEinsumTag"])


class LazilyArraycontextCompilingFunctionCaller(BaseLazilyCompilingFunctionCaller):
    """
    Traces :attr:`BaseLazilyCompilingFunctionCaller.f` to translate the array
    operations to python code that calls equivalent methods of
    :class:`arraycontext.ArrayContext` / :class:`arraycontext.FakeNumpyNamespace`.
    """

    @property
    def compiled_function_returning_array_container_class(
        self,
    ) -> type[CompiledFunction]:
        # This is purposefully left unimplemented to ensure that we do not run
        # into potential mishaps by using the super-class' implementation.
        # TODO: Maybe fix the abstract class' implementation so that it does
        # not rely on us overriding these routines.
        raise NotImplementedError

    @property
    def compiled_function_returning_array_class(self) -> type[CompiledFunction]:
        # This is purposefully left unimplemented to ensure that we do not run
        # into potential mishaps by using the super-class' implementation.
        # TODO: Maybe fix the abstract class' implementation so that it does
        # not rely on us overriding these routines.
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Performs the following operations:

        #. Writes the generated code to disk at the location
            :attr:`SuiteGeneratingArraycontext.main_file_path`.
        #. Compiles the generated code and executes it with the arguments
            *args*, *kwargs* and returns the output.

        .. note::

            The behavior of this routine emulates calling :attr:`f` itself.
        """
        from arraycontext.impl.pytato.compile import (
            _ary_container_key_stringifier,
            _get_arg_id_to_arg_and_arg_id_to_descr,
            _get_f_placeholder_args,
        )

        args, kwargs = (
            tuple(self.actx.thaw(self.actx.freeze(arg)) for arg in args),
            {
                kw: self.actx.thaw(self.actx.freeze(arg))
                for kw, arg in kwargs.items()
            },
        )

        # {{{ remove bad tags

        def _remove_bad_tags(ary):
            new_tags = {
                tag
                for tag in ary.tags
                if tag.__class__.__name__ not in BAD_TAG_TYPENAMES
            }
            return ary.copy(tags=frozenset(new_tags))

        args = tuple(
            (
                arg
                if np.isscalar(arg)
                else rec_map_array_container(_remove_bad_tags, arg)
            )
            for arg in args
        )
        kwargs = {
            kw: (
                arg
                if np.isscalar(arg)
                else rec_map_array_container(_remove_bad_tags, arg)
            )
            for kw, arg in kwargs.items()
        }

        # }}}

        arg_id_to_arg, arg_id_to_descr = _get_arg_id_to_arg_and_arg_id_to_descr(
            args, kwargs
        )

        try:
            compiled_f = self.program_cache[arg_id_to_descr]
        except KeyError:
            pass
        else:
            return compiled_f(arg_id_to_arg)

        dict_of_named_arrays = {}
        input_id_to_name_in_program = {
            arg_id: f"_actx_in_{_ary_container_key_stringifier(arg_id)}"
            for arg_id in arg_id_to_arg
        }

        output_template = self.f(
            *[
                _get_f_placeholder_args(
                    arg, iarg, input_id_to_name_in_program, self.actx
                )
                for iarg, arg in enumerate(args)
            ],
            **{
                kw: _get_f_placeholder_args(
                    arg, kw, input_id_to_name_in_program, self.actx
                )
                for kw, arg in kwargs.items()
            },
        )

        if not (
            is_array_container_type(output_template.__class__)
            or isinstance(output_template, pt.Array)
        ):
            # TODO: We could possibly just short-circuit this interface if the
            # returned type is a scalar. Not sure if it's worth it though.
            raise NotImplementedError(
                f"Function '{self.f.__name__}' to be compiled "
                "did not return an array container or pt.Array,"
                f" but an instance of '{output_template.__class__}' instead."
            )

        def _as_dict_of_named_arrays(keys, ary):
            name = "_pt_out_" + _ary_container_key_stringifier(keys)
            dict_of_named_arrays[name] = ary
            return ary

        rec_keyed_map_array_container(_as_dict_of_named_arrays, output_template)

        from .pytato_target import generate_arraycontext_code

        pt_dict_of_named_arrays = pt.transform.deduplicate_data_wrappers(
            pt.make_dict_of_named_arrays(dict_of_named_arrays)
        )
        pt_dict_of_named_arrays = pt.rewrite_einsums_with_no_broadcasts(
            pt_dict_of_named_arrays
        )
        pt_dict_of_named_arrays = remove_tags_with_typenames(
            pt_dict_of_named_arrays, BAD_TAG_TYPENAMES
        )
        inner_code_prg = generate_arraycontext_code(
            pt_dict_of_named_arrays,
            function_name="_rhs_inner",
            actx=self.actx,
            show_code=False,
        )

        host_code = f"""
        {ast.unparse(ast.fix_missing_locations(
            ast.Module(list(inner_code_prg.import_statements), type_ignores=[])))}
        from pytools import memoize_method
        from functools import cached_property
        from immutables import Map
        from arraycontext import ArrayContext, is_array_container_type
        from dataclasses import dataclass
        from arraycontext.container.traversal import (rec_map_array_container,
                                                      rec_keyed_map_array_container)
        from actx_dgfem_suite.utils import get_actx_dgfem_suite_path


        {ast.unparse(ast.fix_missing_locations(
            ast.Module([inner_code_prg.function_def], type_ignores=[])))}


        @dataclass(frozen=True)
        class RHSInvoker:
            actx: ArrayContext

            @cached_property
            def npzfile(self):
                from immutables import Map
                import os

                kw_to_ary = np.load(
                    os.path.join(get_actx_dgfem_suite_path(),
                                 "{os.path.relpath(self.actx.datawrappers_path,
                                                   start=get_actx_dgfem_suite_path())}")
                )
                return Map({{kw: self.actx.freeze(self.actx.from_numpy(ary))
                            for kw, ary in kw_to_ary.items()}})

            @memoize_method
            def _get_compiled_rhs_inner(self):
                return self.actx.compile(
                    lambda *args, **kwargs: _rhs_inner(self.actx, self.npzfile, *args, **kwargs))

            @memoize_method
            def _get_output_template(self):
                import os
                import pytato as pt
                from pickle import load
                from meshmode.dof_array import array_context_for_pickling

                fpath = os.path.join(get_actx_dgfem_suite_path(),
                                    "{os.path.relpath(self.actx.pickled_ref_output_path,
                                                      start=get_actx_dgfem_suite_path())}")
                with open(fpath, "rb") as fp:
                    with array_context_for_pickling(self.actx):
                        output_template = load(fp)

                def _convert_to_symbolic_array(ary):
                    return pt.zeros(ary.shape, ary.dtype)

                # convert to symbolic array to not free the memory corresponding to
                # output_template
                return rec_map_array_container(_convert_to_symbolic_array,
                                               output_template)

            @memoize_method
            def _get_key_to_pos_in_output_template(self):
                from arraycontext.impl.pytato.compile import (
                    _ary_container_key_stringifier)

                output_keys = set()
                output_template = self._get_output_template()

                def _as_dict_of_named_arrays(keys, ary):
                    output_keys.add(keys)
                    return ary

                rec_keyed_map_array_container(_as_dict_of_named_arrays,
                                              output_template)

                return Map({{output_key: i
                            for i, output_key in enumerate(sorted(
                                    output_keys, key=_ary_container_key_stringifier))}})

            @cached_property
            def _rhs_inner_argument_names(self):
                return {{
                    '{"', '".join(sorted(inner_code_prg.argument_names))}'
                }}

            def __call__(self, *args, **kwargs):
                from arraycontext.impl.pytato.compile import (
                    _get_arg_id_to_arg_and_arg_id_to_descr,
                    _ary_container_key_stringifier)
                arg_id_to_arg, _ = _get_arg_id_to_arg_and_arg_id_to_descr(args, kwargs)
                input_kwargs_to_rhs_inner = {{
                    "_actx_in_" + _ary_container_key_stringifier(arg_id): arg
                    for arg_id, arg in arg_id_to_arg.items()}}

                input_kwargs_to_rhs_inner = {{
                    kw: input_kwargs_to_rhs_inner[kw]
                    for kw in self._rhs_inner_argument_names
                }}

                compiled_rhs_inner = self._get_compiled_rhs_inner()
                result_as_np_obj_array = compiled_rhs_inner(**input_kwargs_to_rhs_inner)

                output_template = self._get_output_template()

                if is_array_container_type(output_template.__class__):
                    keys_to_pos = self._get_key_to_pos_in_output_template()

                    def to_output_template(keys, _):
                        return result_as_np_obj_array[keys_to_pos[keys]]

                    return rec_keyed_map_array_container(to_output_template,
                                                         self._get_output_template())
                else:
                    from pytato.array import Array
                    assert isinstance(output_template, Array)
                    assert result_as_np_obj_array.shape == (1,)
                    return result_as_np_obj_array[0]
        """  # noqa: E501
        host_code = re.sub(
            r"^        (?P<rest_of_line>.+)$",
            r"\g<rest_of_line>",
            host_code,
            flags=re.MULTILINE,
        )

        from pytools.codegen import remove_common_indentation

        host_code = remove_common_indentation(host_code)

        with open(f"{self.actx.main_file_path}", "w") as fp:
            fp.write(host_code)

        autoflake._main(
            [
                "--remove-unused-variables",
                "--imports",
                "loopy,arraycontext",
                "--in-place",
                self.actx.main_file_path,
            ],
            standard_out=None,
            standard_error=sys.stderr,
            standard_input=sys.stdin,
        )
        black.format_file_in_place(
            Path(self.actx.main_file_path),
            fast=False,
            mode=black.Mode(line_length=80),
            write_back=black.WriteBack.YES,
        )

        with open(f"{self.actx.datawrappers_path}", "wb") as fp:
            np.savez(fp, **inner_code_prg.numpy_arrays_to_store)

        with open(f"{self.actx.pickled_ref_input_args_path}", "wb") as fp:
            import pickle

            if all(
                (
                    is_dataclass_array_container(arg)
                    or (
                        isinstance(arg, np.ndarray)
                        and arg.dtype == "O"
                        and all(is_dataclass_array_container(el) for el in arg)
                    )
                    or np.isscalar(arg)
                )
                for arg in args
            ) and all(
                (
                    is_dataclass_array_container(arg)
                    or (
                        isinstance(arg, np.ndarray)
                        and arg.dtype == "O"
                        and all(is_dataclass_array_container(el) for el in arg)
                    )
                    or np.isscalar(arg)
                )
                for arg in kwargs.values()
            ):
                with array_context_for_pickling(self.actx.clone()):
                    pickle.dump((args, kwargs), fp)
            elif any(is_dataclass_array_container(arg) for arg in args) or any(
                is_dataclass_array_container(arg) for arg in kwargs.values()
            ):
                raise NotImplementedError(
                    "Pickling not implemented for input" " types."
                )
            else:
                np_args = tuple(self.actx.to_numpy(arg) for arg in args)
                np_kwargs = {
                    kw: self.actx.to_numpy(arg) for kw, arg in kwargs.items()
                }
                pickle.dump((np_args, np_kwargs), fp)

        ref_out = self.actx.thaw(self.actx.freeze(self.f(*args, **kwargs)))
        ref_out = rec_map_array_container(_remove_bad_tags, ref_out)

        with open(f"{self.actx.pickled_ref_output_path}", "wb") as fp:
            import pickle

            if is_dataclass_array_container(ref_out) or (
                isinstance(ref_out, np.ndarray)
                and ref_out.dtype == "O"
                and all(is_dataclass_array_container(el) for el in ref_out)
            ):
                with array_context_for_pickling(self.actx.clone()):
                    pickle.dump(ref_out, fp)
            else:
                pickle.dump(self.actx.to_numpy(ref_out), fp)

        self_actx_clone = self.actx.clone()

        # {{{ get 'rhs' callable

        variables_after_execution: dict[str, Any] = {
            "_MODULE_SOURCE_CODE": host_code,  # helps pudb
        }
        exec(host_code, variables_after_execution)
        compiled_func = variables_after_execution["RHSInvoker"](self_actx_clone)

        # }}}

        self.program_cache[arg_id_to_descr] = compiled_func

        # {{{ test that the codegen was successful

        output = self.program_cache[arg_id_to_descr](*args, **kwargs)

        rec_multimap_array_container(
            np.testing.assert_allclose,
            self_actx_clone.to_numpy(output),
            self.actx.to_numpy(ref_out),
        )

        # }}}

        return output


class SuiteGeneratingArraycontext(DGFEMOptimizerArrayContext):
    """
    Overrides the :meth:`compile` method of
    :class:`arraycontext.PytatoJAXArrayContext` to generate python code that is
    compatible to run with any :class:`ArrayContext` and then executes the
    generated code.
    """

    def __init__(
        self,
        queue,
        allocator,
        *,
        main_file_path: str,
        datawrappers_path: str,
        pickled_ref_input_args_path: str,
        pickled_ref_output_path: str,
    ) -> None:

        super().__init__(queue, allocator)

        if any(
            not os.path.isabs(filepath)
            for filepath in [
                main_file_path,
                datawrappers_path,
                pickled_ref_input_args_path,
                pickled_ref_output_path,
            ]
        ):
            raise ValueError("Absolute paths are expected.")

        self.main_file_path = main_file_path
        self.datawrappers_path = datawrappers_path
        self.pickled_ref_input_args_path = pickled_ref_input_args_path
        self.pickled_ref_output_path = pickled_ref_output_path

    def compile(self, f: Callable[..., Any]) -> Callable[..., Any]:
        if f.__name__ == "rhs":
            # We only compile RHS functions
            return LazilyArraycontextCompilingFunctionCaller(self, f)
        else:
            return super().compile(f)

    @memoize_method
    def clone(self):
        return DGFEMOptimizerArrayContext(self.queue, self.allocator)


# vim: fdm=marker
