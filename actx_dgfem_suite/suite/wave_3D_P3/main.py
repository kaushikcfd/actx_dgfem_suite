from dataclasses import dataclass
from functools import cached_property

import loopy as lp
import numpy as np
from arraycontext import ArrayContext, is_array_container_type, make_loopy_program
from arraycontext.container.traversal import (
    rec_keyed_map_array_container,
    rec_map_array_container,
)
from immutables import Map
from meshmode.transform_metadata import (
    DiscretizationAmbientDimAxisTag,
    DiscretizationDOFAxisTag,
    DiscretizationElementAxisTag,
    DiscretizationFaceAxisTag,
    DiscretizationTopologicalDimAxisTag,
    FirstAxisIsElementsTag,
)
from pytato.tags import PrefixNamed
from pytools import memoize_method
from pytools.obj_array import make_obj_array

from actx_dgfem_suite.utils import get_actx_dgfem_suite_path


def _rhs_inner(
    actx,
    npzfile,
    *,
    _actx_in_1_u_0,
    _actx_in_1_v_0_0,
    _actx_in_1_v_1_0,
    _actx_in_1_v_2_0,
):
    _pt_t_unit = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 10799 and 0 <= _1 <= 9 }",
        "out[_0, _1] = in_0[in_1[_0] % 4, _1]",
        [
            lp.GlobalArg("out", dtype=np.int32, shape=(10800, 10)),
            lp.GlobalArg("in_0", dtype=np.int32, shape=(4, 10)),
            lp.GlobalArg("in_1", dtype=np.int8, shape=(10800,)),
        ],
    )
    _pt_t_unit_0 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 10799 and 0 <= _1 <= 9 }",
        "out[_0, _1] = in_0[in_1[_0, 0] % 162000, in_2[_0, _1] % 20]",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(10800, 10)),
            lp.GlobalArg("in_0", dtype=np.float64, shape=(162000, 20)),
            lp.GlobalArg("in_1", dtype=np.int32, shape=(10800, 1)),
            lp.GlobalArg("in_2", dtype=np.int32, shape=(10800, 10)),
        ],
    )
    _pt_t_unit_1 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 10799 and 0 <= _1 <= 9 }",
        "out[_0, _1] = _in0[_0, _1]*_in1[_0, 0]",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(10800, 10)),
            lp.GlobalArg("_in0", dtype=np.float64, shape=(10800, 10)),
            lp.GlobalArg("_in1", dtype=np.float64, shape=(10800, 1)),
        ],
    )
    _pt_t_unit_10 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 647999 and 0 <= _1 <= 9 }",
        "out[_0, _1] = in_0[in_1[_0, 0] % 637200, in_2[_0, _1] % 10]",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(648000, 10)),
            lp.GlobalArg("in_0", dtype=np.float64, shape=(637200, 10)),
            lp.GlobalArg("in_1", dtype=np.int32, shape=(648000, 1)),
            lp.GlobalArg("in_2", dtype=np.int32, shape=(648000, 10)),
        ],
    )
    _pt_t_unit_2 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 647999 and 0 <= _1 <= 9 }",
        "out[_0, _1] = in_0[0, _1]",
        [
            lp.GlobalArg("out", dtype=np.int32, shape=(648000, 10)),
            lp.GlobalArg("in_0", dtype=np.int32, shape=(1, 10)),
            lp.GlobalArg("in_1", dtype=np.int32, shape=(648000,)),
        ],
    )
    _pt_t_unit_3 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 647999 and 0 <= _1 <= 9 }",
        "out[_0, _1] = in_0[in_1[_0, 0] % 10800, in_2[_0, _1] % 10]",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(648000, 10)),
            lp.GlobalArg("in_0", dtype=np.float64, shape=(10800, 10)),
            lp.GlobalArg("in_1", dtype=np.int32, shape=(648000, 1)),
            lp.GlobalArg("in_2", dtype=np.int32, shape=(648000, 10)),
        ],
    )
    _pt_t_unit_4 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 647999 and 0 <= _1 <= 9 }",
        "out[_0, _1] = _in1[_0, _1] if _in0[_0, 0] else 0",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(648000, 10)),
            lp.GlobalArg("_in0", dtype=np.int8, shape=(648000, 1)),
            lp.GlobalArg("_in1", dtype=np.float64, shape=(648000, 10)),
        ],
    )
    _pt_t_unit_5 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 637199 and 0 <= _1 <= 9 }",
        "out[_0, _1] = in_0[in_1[_0] % 4, _1]",
        [
            lp.GlobalArg("out", dtype=np.int32, shape=(637200, 10)),
            lp.GlobalArg("in_0", dtype=np.int32, shape=(4, 10)),
            lp.GlobalArg("in_1", dtype=np.int8, shape=(637200,)),
        ],
    )
    _pt_t_unit_6 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 637199 and 0 <= _1 <= 9 }",
        "out[_0, _1] = in_0[in_1[_0, 0] % 162000, in_2[_0, _1] % 20]",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(637200, 10)),
            lp.GlobalArg("in_0", dtype=np.float64, shape=(162000, 20)),
            lp.GlobalArg("in_1", dtype=np.int32, shape=(637200, 1)),
            lp.GlobalArg("in_2", dtype=np.int32, shape=(637200, 10)),
        ],
    )
    _pt_t_unit_7 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 637199 and 0 <= _1 <= 9 }",
        "out[_0, _1] = in_0[in_1[_0] % 3, _1]",
        [
            lp.GlobalArg("out", dtype=np.int32, shape=(637200, 10)),
            lp.GlobalArg("in_0", dtype=np.int32, shape=(3, 10)),
            lp.GlobalArg("in_1", dtype=np.int8, shape=(637200,)),
        ],
    )
    _pt_t_unit_8 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 637199 and 0 <= _1 <= 9 }",
        "out[_0, _1] = in_0[in_1[_0, 0] % 637200, in_2[_0, _1] % 10]",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(637200, 10)),
            lp.GlobalArg("in_0", dtype=np.float64, shape=(637200, 10)),
            lp.GlobalArg("in_1", dtype=np.int32, shape=(637200, 1)),
            lp.GlobalArg("in_2", dtype=np.int32, shape=(637200, 10)),
        ],
    )
    _pt_t_unit_9 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 637199 and 0 <= _1 <= 9 }",
        "out[_0, _1] = _in0[_0, _1]*_in1[_0, 0]",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(637200, 10)),
            lp.GlobalArg("_in0", dtype=np.float64, shape=(637200, 10)),
            lp.GlobalArg("_in1", dtype=np.float64, shape=(637200, 1)),
        ],
    )
    _pt_data = actx.thaw(npzfile["_pt_data"])
    _pt_data = actx.tag((PrefixNamed(prefix="area_el_vol"),), _pt_data)
    _pt_data = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_data)
    _pt_data = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_data)
    _pt_tmp_2 = 1.0 / _pt_data
    _pt_tmp_1 = _pt_tmp_2[:, 0]
    _pt_data_0 = actx.thaw(npzfile["_pt_data_0"])
    del _pt_tmp_2
    _pt_data_0 = actx.tag_axis(0, (DiscretizationDOFAxisTag(),), _pt_data_0)
    _pt_data_1 = actx.thaw(npzfile["_pt_data_1"])
    _pt_data_1 = actx.tag((PrefixNamed(prefix="inv_metric_deriv_vol"),), _pt_data_1)
    _pt_data_1 = actx.tag_axis(0, (DiscretizationAmbientDimAxisTag(),), _pt_data_1)
    _pt_data_1 = actx.tag_axis(
        1, (DiscretizationTopologicalDimAxisTag(),), _pt_data_1
    )
    _pt_data_1 = actx.tag_axis(2, (DiscretizationElementAxisTag(),), _pt_data_1)
    _pt_data_1 = actx.tag_axis(3, (DiscretizationDOFAxisTag(),), _pt_data_1)
    _pt_tmp_6 = _pt_data_1[:, :, :, 0]
    _pt_data_2 = actx.thaw(npzfile["_pt_data_2"])
    _pt_data_2 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_data_2)
    _actx_in_1_v_0_0 = actx.tag_axis(
        0, (DiscretizationElementAxisTag(),), _actx_in_1_v_0_0
    )
    _actx_in_1_v_0_0 = actx.tag_axis(
        1, (DiscretizationDOFAxisTag(),), _actx_in_1_v_0_0
    )
    _actx_in_1_v_1_0 = actx.tag_axis(
        0, (DiscretizationElementAxisTag(),), _actx_in_1_v_1_0
    )
    _actx_in_1_v_1_0 = actx.tag_axis(
        1, (DiscretizationDOFAxisTag(),), _actx_in_1_v_1_0
    )
    _actx_in_1_v_2_0 = actx.tag_axis(
        0, (DiscretizationElementAxisTag(),), _actx_in_1_v_2_0
    )
    _actx_in_1_v_2_0 = actx.tag_axis(
        1, (DiscretizationDOFAxisTag(),), _actx_in_1_v_2_0
    )
    _pt_tmp_7 = actx.np.stack(
        [_actx_in_1_v_0_0, _actx_in_1_v_1_0, _actx_in_1_v_2_0], axis=0
    )
    _pt_tmp_5 = actx.einsum("ijk, jlm, ikm -> kl", _pt_tmp_6, _pt_data_2, _pt_tmp_7)
    _pt_tmp_5 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_5)
    del _pt_tmp_7
    _pt_tmp_4 = -1 * _pt_tmp_5
    _pt_data_3 = actx.thaw(npzfile["_pt_data_3"])
    del _pt_tmp_5
    _pt_data_3 = actx.tag_axis(0, (DiscretizationDOFAxisTag(),), _pt_data_3)
    _pt_data_3 = actx.tag_axis(2, (DiscretizationDOFAxisTag(),), _pt_data_3)
    _pt_data_4 = actx.thaw(npzfile["_pt_data_4"])
    _pt_data_4 = actx.tag(
        (PrefixNamed(prefix="area_el_b_face_restr_all"),), _pt_data_4
    )
    _pt_data_4 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_data_4)
    _pt_data_4 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_data_4)
    _pt_tmp_10 = actx.np.reshape(_pt_data_4, (4, 162000, 1))
    _pt_tmp_10 = actx.tag_axis(1, (DiscretizationElementAxisTag(),), _pt_tmp_10)
    _pt_tmp_9 = _pt_tmp_10[:, :, 0]
    _pt_data_5 = actx.thaw(npzfile["_pt_data_5"])
    del _pt_tmp_10
    _pt_data_5 = actx.tag((PrefixNamed(prefix="from_el_present"),), _pt_data_5)
    _pt_tmp_15 = actx.np.reshape(_pt_data_5, (648000, 1))
    _pt_tmp_15 = actx.tag((PrefixNamed(prefix="from_el_present"),), _pt_tmp_15)
    _pt_data_6 = actx.thaw(npzfile["_pt_data_6"])
    _pt_data_6 = actx.tag((PrefixNamed(prefix="from_el_indices"),), _pt_data_6)
    _pt_tmp_26 = actx.np.reshape(_pt_data_6, (10800, 1))
    _pt_tmp_26 = actx.tag((PrefixNamed(prefix="from_el_indices"),), _pt_tmp_26)
    _pt_data_7 = actx.thaw(npzfile["_pt_data_7"])
    _pt_data_7 = actx.tag((PrefixNamed(prefix="dof_pick_lists"),), _pt_data_7)
    _pt_data_8 = actx.thaw(npzfile["_pt_data_8"])
    _pt_data_8 = actx.tag((PrefixNamed(prefix="dof_pick_list_indices"),), _pt_data_8)
    _pt_tmp_27 = (
        _pt_data_7[_pt_data_8]
        if actx.permits_advanced_indexing
        else actx.call_loopy(_pt_t_unit, in_0=_pt_data_7, in_1=_pt_data_8)["out"]
    )
    _pt_tmp_25 = (
        _actx_in_1_v_0_0[_pt_tmp_26, _pt_tmp_27]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0,
            in_0=_actx_in_1_v_0_0,
            in_1=_pt_tmp_26,
            in_2=_pt_tmp_27,
        )["out"]
    )
    _pt_tmp_25 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_25)
    _pt_tmp_25 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_25)
    _pt_tmp_24 = 0 + _pt_tmp_25
    _pt_tmp_23 = _pt_tmp_24 + _pt_tmp_24
    del _pt_tmp_25
    _pt_tmp_22 = 0.5 * _pt_tmp_23
    _pt_data_9 = actx.thaw(npzfile["_pt_data_9"])
    del _pt_tmp_23
    _pt_data_9 = actx.tag((PrefixNamed(prefix="normal_1_b_all"),), _pt_data_9)
    _pt_data_9 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_data_9)
    _pt_data_9 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_data_9)
    _pt_tmp_21 = (
        _pt_tmp_22 * _pt_data_9
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_1, _in0=_pt_tmp_22, _in1=_pt_data_9)["out"]
    )
    _pt_tmp_32 = (
        _actx_in_1_v_1_0[_pt_tmp_26, _pt_tmp_27]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0,
            in_0=_actx_in_1_v_1_0,
            in_1=_pt_tmp_26,
            in_2=_pt_tmp_27,
        )["out"]
    )
    del _pt_tmp_22
    _pt_tmp_32 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_32)
    _pt_tmp_32 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_32)
    _pt_tmp_31 = 0 + _pt_tmp_32
    _pt_tmp_30 = _pt_tmp_31 + _pt_tmp_31
    del _pt_tmp_32
    _pt_tmp_29 = 0.5 * _pt_tmp_30
    _pt_data_10 = actx.thaw(npzfile["_pt_data_10"])
    del _pt_tmp_30
    _pt_data_10 = actx.tag((PrefixNamed(prefix="normal_2_b_all"),), _pt_data_10)
    _pt_data_10 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_data_10)
    _pt_data_10 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_data_10)
    _pt_tmp_28 = (
        _pt_tmp_29 * _pt_data_10
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_1, _in0=_pt_tmp_29, _in1=_pt_data_10)["out"]
    )
    _pt_tmp_20 = _pt_tmp_21 + _pt_tmp_28
    del _pt_tmp_29
    _pt_tmp_37 = (
        _actx_in_1_v_2_0[_pt_tmp_26, _pt_tmp_27]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0,
            in_0=_actx_in_1_v_2_0,
            in_1=_pt_tmp_26,
            in_2=_pt_tmp_27,
        )["out"]
    )
    del _pt_tmp_21, _pt_tmp_28
    _pt_tmp_37 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_37)
    _pt_tmp_37 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_37)
    _pt_tmp_36 = 0 + _pt_tmp_37
    _pt_tmp_35 = _pt_tmp_36 + _pt_tmp_36
    del _pt_tmp_37
    _pt_tmp_34 = 0.5 * _pt_tmp_35
    _pt_data_11 = actx.thaw(npzfile["_pt_data_11"])
    del _pt_tmp_35
    _pt_data_11 = actx.tag((PrefixNamed(prefix="normal_4_b_all"),), _pt_data_11)
    _pt_data_11 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_data_11)
    _pt_data_11 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_data_11)
    _pt_tmp_33 = (
        _pt_tmp_34 * _pt_data_11
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_1, _in0=_pt_tmp_34, _in1=_pt_data_11)["out"]
    )
    _pt_tmp_19 = _pt_tmp_20 + _pt_tmp_33
    del _pt_tmp_34
    _actx_in_1_u_0 = actx.tag_axis(
        0, (DiscretizationElementAxisTag(),), _actx_in_1_u_0
    )
    del _pt_tmp_20, _pt_tmp_33
    _actx_in_1_u_0 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _actx_in_1_u_0)
    _pt_tmp_42 = (
        _actx_in_1_u_0[_pt_tmp_26, _pt_tmp_27]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_actx_in_1_u_0, in_1=_pt_tmp_26, in_2=_pt_tmp_27
        )["out"]
    )
    _pt_tmp_42 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_42)
    del _pt_tmp_26, _pt_tmp_27
    _pt_tmp_42 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_42)
    _pt_tmp_41 = 0 + _pt_tmp_42
    _pt_tmp_40 = -1 * _pt_tmp_41
    del _pt_tmp_42
    _pt_tmp_39 = _pt_tmp_40 - _pt_tmp_41
    _pt_tmp_38 = 0.5 * _pt_tmp_39
    _pt_tmp_18 = _pt_tmp_19 + _pt_tmp_38
    del _pt_tmp_39
    _pt_tmp_17 = 1 * _pt_tmp_18
    del _pt_tmp_19, _pt_tmp_38
    _pt_data_12 = actx.thaw(npzfile["_pt_data_12"])
    del _pt_tmp_18
    _pt_data_12 = actx.tag((PrefixNamed(prefix="from_el_indices"),), _pt_data_12)
    _pt_tmp_43 = actx.np.reshape(_pt_data_12, (648000, 1))
    _pt_tmp_43 = actx.tag((PrefixNamed(prefix="from_el_indices"),), _pt_tmp_43)
    _pt_data_13 = actx.thaw(npzfile["_pt_data_13"])
    _pt_data_13 = actx.tag((PrefixNamed(prefix="dof_pick_lists"),), _pt_data_13)
    _pt_tmp_45 = actx.np.zeros((648000,), dtype=np.int32)
    _pt_tmp_44 = (
        _pt_data_13[_pt_tmp_45]
        if actx.permits_advanced_indexing
        else actx.call_loopy(_pt_t_unit_2, in_0=_pt_data_13, in_1=_pt_tmp_45)["out"]
    )
    _pt_tmp_16 = (
        _pt_tmp_17[_pt_tmp_43, _pt_tmp_44]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_3, in_0=_pt_tmp_17, in_1=_pt_tmp_43, in_2=_pt_tmp_44
        )["out"]
    )
    _pt_tmp_14 = (
        actx.np.where(_pt_tmp_15, _pt_tmp_16, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_4, _in0=_pt_tmp_15, _in1=_pt_tmp_16)["out"]
    )
    del _pt_tmp_17
    _pt_tmp_14 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_14)
    del _pt_tmp_16
    _pt_tmp_14 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_14)
    _pt_tmp_13 = 0 + _pt_tmp_14
    _pt_data_14 = actx.thaw(npzfile["_pt_data_14"])
    del _pt_tmp_14
    _pt_data_14 = actx.tag((PrefixNamed(prefix="from_el_present"),), _pt_data_14)
    _pt_tmp_49 = actx.np.reshape(_pt_data_14, (648000, 1))
    _pt_tmp_49 = actx.tag((PrefixNamed(prefix="from_el_present"),), _pt_tmp_49)
    _pt_data_15 = actx.thaw(npzfile["_pt_data_15"])
    _pt_data_15 = actx.tag((PrefixNamed(prefix="from_el_indices"),), _pt_data_15)
    _pt_tmp_60 = actx.np.reshape(_pt_data_15, (637200, 1))
    _pt_tmp_60 = actx.tag((PrefixNamed(prefix="from_el_indices"),), _pt_tmp_60)
    _pt_data_16 = actx.thaw(npzfile["_pt_data_16"])
    _pt_data_16 = actx.tag((PrefixNamed(prefix="dof_pick_lists"),), _pt_data_16)
    _pt_data_17 = actx.thaw(npzfile["_pt_data_17"])
    _pt_data_17 = actx.tag(
        (PrefixNamed(prefix="dof_pick_list_indices"),), _pt_data_17
    )
    _pt_tmp_61 = (
        _pt_data_16[_pt_data_17]
        if actx.permits_advanced_indexing
        else actx.call_loopy(_pt_t_unit_5, in_0=_pt_data_16, in_1=_pt_data_17)["out"]
    )
    _pt_tmp_59 = (
        _actx_in_1_v_0_0[_pt_tmp_60, _pt_tmp_61]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_6,
            in_0=_actx_in_1_v_0_0,
            in_1=_pt_tmp_60,
            in_2=_pt_tmp_61,
        )["out"]
    )
    _pt_tmp_59 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_59)
    _pt_tmp_59 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_59)
    _pt_tmp_58 = 0 + _pt_tmp_59
    _pt_data_18 = actx.thaw(npzfile["_pt_data_18"])
    del _pt_tmp_59
    _pt_data_18 = actx.tag((PrefixNamed(prefix="from_el_indices"),), _pt_data_18)
    _pt_tmp_64 = actx.np.reshape(_pt_data_18, (637200, 1))
    _pt_tmp_64 = actx.tag((PrefixNamed(prefix="from_el_indices"),), _pt_tmp_64)
    _pt_data_19 = actx.thaw(npzfile["_pt_data_19"])
    _pt_data_19 = actx.tag((PrefixNamed(prefix="dof_pick_lists"),), _pt_data_19)
    _pt_data_20 = actx.thaw(npzfile["_pt_data_20"])
    _pt_data_20 = actx.tag(
        (PrefixNamed(prefix="dof_pick_list_indices"),), _pt_data_20
    )
    _pt_tmp_65 = (
        _pt_data_19[_pt_data_20]
        if actx.permits_advanced_indexing
        else actx.call_loopy(_pt_t_unit_7, in_0=_pt_data_19, in_1=_pt_data_20)["out"]
    )
    _pt_tmp_63 = (
        _pt_tmp_58[_pt_tmp_64, _pt_tmp_65]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_8, in_0=_pt_tmp_58, in_1=_pt_tmp_64, in_2=_pt_tmp_65
        )["out"]
    )
    _pt_tmp_63 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_63)
    _pt_tmp_63 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_63)
    _pt_tmp_62 = 0 + _pt_tmp_63
    _pt_tmp_57 = _pt_tmp_58 + _pt_tmp_62
    del _pt_tmp_63
    _pt_tmp_56 = 0.5 * _pt_tmp_57
    _pt_data_21 = actx.thaw(npzfile["_pt_data_21"])
    del _pt_tmp_57
    _pt_data_21 = actx.tag(
        (PrefixNamed(prefix="normal_1_b_face_restr_interior"),), _pt_data_21
    )
    _pt_data_21 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_data_21)
    _pt_data_21 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_data_21)
    _pt_tmp_55 = (
        _pt_tmp_56 * _pt_data_21
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_56, _in1=_pt_data_21)["out"]
    )
    _pt_tmp_70 = (
        _actx_in_1_v_1_0[_pt_tmp_60, _pt_tmp_61]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_6,
            in_0=_actx_in_1_v_1_0,
            in_1=_pt_tmp_60,
            in_2=_pt_tmp_61,
        )["out"]
    )
    del _pt_tmp_56
    _pt_tmp_70 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_70)
    _pt_tmp_70 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_70)
    _pt_tmp_69 = 0 + _pt_tmp_70
    _pt_tmp_72 = (
        _pt_tmp_69[_pt_tmp_64, _pt_tmp_65]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_8, in_0=_pt_tmp_69, in_1=_pt_tmp_64, in_2=_pt_tmp_65
        )["out"]
    )
    del _pt_tmp_70
    _pt_tmp_72 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_72)
    _pt_tmp_72 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_72)
    _pt_tmp_71 = 0 + _pt_tmp_72
    _pt_tmp_68 = _pt_tmp_69 + _pt_tmp_71
    del _pt_tmp_72
    _pt_tmp_67 = 0.5 * _pt_tmp_68
    _pt_data_22 = actx.thaw(npzfile["_pt_data_22"])
    del _pt_tmp_68
    _pt_data_22 = actx.tag(
        (PrefixNamed(prefix="normal_2_b_face_restr_interior"),), _pt_data_22
    )
    _pt_data_22 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_data_22)
    _pt_data_22 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_data_22)
    _pt_tmp_66 = (
        _pt_tmp_67 * _pt_data_22
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_67, _in1=_pt_data_22)["out"]
    )
    _pt_tmp_54 = _pt_tmp_55 + _pt_tmp_66
    del _pt_tmp_67
    _pt_tmp_77 = (
        _actx_in_1_v_2_0[_pt_tmp_60, _pt_tmp_61]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_6,
            in_0=_actx_in_1_v_2_0,
            in_1=_pt_tmp_60,
            in_2=_pt_tmp_61,
        )["out"]
    )
    del _pt_tmp_55, _pt_tmp_66
    _pt_tmp_77 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_77)
    _pt_tmp_77 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_77)
    _pt_tmp_76 = 0 + _pt_tmp_77
    _pt_tmp_79 = (
        _pt_tmp_76[_pt_tmp_64, _pt_tmp_65]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_8, in_0=_pt_tmp_76, in_1=_pt_tmp_64, in_2=_pt_tmp_65
        )["out"]
    )
    del _pt_tmp_77
    _pt_tmp_79 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_79)
    _pt_tmp_79 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_79)
    _pt_tmp_78 = 0 + _pt_tmp_79
    _pt_tmp_75 = _pt_tmp_76 + _pt_tmp_78
    del _pt_tmp_79
    _pt_tmp_74 = 0.5 * _pt_tmp_75
    _pt_data_23 = actx.thaw(npzfile["_pt_data_23"])
    del _pt_tmp_75
    _pt_data_23 = actx.tag(
        (PrefixNamed(prefix="normal_4_b_face_restr_interior"),), _pt_data_23
    )
    _pt_data_23 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_data_23)
    _pt_data_23 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_data_23)
    _pt_tmp_73 = (
        _pt_tmp_74 * _pt_data_23
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_74, _in1=_pt_data_23)["out"]
    )
    _pt_tmp_53 = _pt_tmp_54 + _pt_tmp_73
    del _pt_tmp_74
    _pt_tmp_85 = (
        _actx_in_1_u_0[_pt_tmp_60, _pt_tmp_61]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_6, in_0=_actx_in_1_u_0, in_1=_pt_tmp_60, in_2=_pt_tmp_61
        )["out"]
    )
    del _pt_tmp_54, _pt_tmp_73
    _pt_tmp_85 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_85)
    del _pt_tmp_60, _pt_tmp_61
    _pt_tmp_85 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_85)
    _pt_tmp_84 = 0 + _pt_tmp_85
    _pt_tmp_83 = (
        _pt_tmp_84[_pt_tmp_64, _pt_tmp_65]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_8, in_0=_pt_tmp_84, in_1=_pt_tmp_64, in_2=_pt_tmp_65
        )["out"]
    )
    del _pt_tmp_85
    _pt_tmp_83 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_83)
    del _pt_tmp_64, _pt_tmp_65
    _pt_tmp_83 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_83)
    _pt_tmp_82 = 0 + _pt_tmp_83
    _pt_tmp_81 = _pt_tmp_82 - _pt_tmp_84
    del _pt_tmp_83
    _pt_tmp_80 = 0.5 * _pt_tmp_81
    _pt_tmp_52 = _pt_tmp_53 + _pt_tmp_80
    del _pt_tmp_81
    _pt_tmp_51 = 1 * _pt_tmp_52
    del _pt_tmp_53, _pt_tmp_80
    _pt_data_24 = actx.thaw(npzfile["_pt_data_24"])
    del _pt_tmp_52
    _pt_data_24 = actx.tag((PrefixNamed(prefix="from_el_indices"),), _pt_data_24)
    _pt_tmp_86 = actx.np.reshape(_pt_data_24, (648000, 1))
    _pt_tmp_86 = actx.tag((PrefixNamed(prefix="from_el_indices"),), _pt_tmp_86)
    _pt_data_25 = actx.thaw(npzfile["_pt_data_25"])
    _pt_data_25 = actx.tag((PrefixNamed(prefix="dof_pick_lists"),), _pt_data_25)
    _pt_tmp_87 = (
        _pt_data_25[_pt_tmp_45]
        if actx.permits_advanced_indexing
        else actx.call_loopy(_pt_t_unit_2, in_0=_pt_data_25, in_1=_pt_tmp_45)["out"]
    )
    _pt_tmp_50 = (
        _pt_tmp_51[_pt_tmp_86, _pt_tmp_87]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_10, in_0=_pt_tmp_51, in_1=_pt_tmp_86, in_2=_pt_tmp_87
        )["out"]
    )
    del _pt_tmp_45
    _pt_tmp_48 = (
        actx.np.where(_pt_tmp_49, _pt_tmp_50, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_4, _in0=_pt_tmp_49, _in1=_pt_tmp_50)["out"]
    )
    del _pt_tmp_51
    _pt_tmp_48 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_48)
    del _pt_tmp_50
    _pt_tmp_48 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_48)
    _pt_tmp_47 = 0 + _pt_tmp_48
    _pt_tmp_46 = 0 + _pt_tmp_47
    del _pt_tmp_48
    _pt_tmp_12 = _pt_tmp_13 + _pt_tmp_46
    del _pt_tmp_47
    _pt_tmp_11 = actx.np.reshape(_pt_tmp_12, (4, 162000, 10))
    del _pt_tmp_13, _pt_tmp_46
    _pt_tmp_11 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_11)
    del _pt_tmp_12
    _pt_tmp_8 = actx.einsum("ijk, jl, jlk -> li", _pt_data_3, _pt_tmp_9, _pt_tmp_11)
    _pt_tmp_8 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_8)
    del _pt_tmp_11
    _pt_tmp_3 = _pt_tmp_4 + _pt_tmp_8
    _pt_tmp_0 = actx.einsum("i, jk, ik -> ij", _pt_tmp_1, _pt_data_0, _pt_tmp_3)
    del _pt_tmp_4, _pt_tmp_8
    _pt_tmp_0 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_0)
    del _pt_tmp_3
    _pt_tmp_92 = actx.einsum(
        "ijk, jlm, km -> ikl", _pt_tmp_6, _pt_data_2, _actx_in_1_u_0
    )
    _pt_tmp_92 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_92)
    del _pt_tmp_6
    _pt_tmp_91 = _pt_tmp_92[0]
    _pt_tmp_90 = -1 * _pt_tmp_91
    _pt_tmp_103 = _pt_tmp_41 + _pt_tmp_40
    del _pt_tmp_91
    _pt_tmp_102 = 0.5 * _pt_tmp_103
    del _pt_tmp_40, _pt_tmp_41
    _pt_tmp_101 = (
        _pt_tmp_102 * _pt_data_9
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_1, _in0=_pt_tmp_102, _in1=_pt_data_9)["out"]
    )
    del _pt_tmp_103
    _pt_tmp_109 = _pt_tmp_24 - _pt_tmp_24
    _pt_tmp_108 = (
        _pt_tmp_109 * _pt_data_9
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_1, _in0=_pt_tmp_109, _in1=_pt_data_9)["out"]
    )
    del _pt_tmp_24
    _pt_tmp_111 = _pt_tmp_31 - _pt_tmp_31
    del _pt_tmp_109
    _pt_tmp_110 = (
        _pt_tmp_111 * _pt_data_10
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_1, _in0=_pt_tmp_111, _in1=_pt_data_10)["out"]
    )
    del _pt_tmp_31
    _pt_tmp_107 = _pt_tmp_108 + _pt_tmp_110
    del _pt_tmp_111
    _pt_tmp_113 = _pt_tmp_36 - _pt_tmp_36
    del _pt_tmp_108, _pt_tmp_110
    _pt_tmp_112 = (
        _pt_tmp_113 * _pt_data_11
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_1, _in0=_pt_tmp_113, _in1=_pt_data_11)["out"]
    )
    del _pt_tmp_36
    _pt_tmp_106 = _pt_tmp_107 + _pt_tmp_112
    del _pt_tmp_113
    _pt_tmp_105 = 0.5 * _pt_tmp_106
    del _pt_tmp_107, _pt_tmp_112
    _pt_tmp_104 = (
        _pt_tmp_105 * _pt_data_9
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_1, _in0=_pt_tmp_105, _in1=_pt_data_9)["out"]
    )
    del _pt_tmp_106
    _pt_tmp_100 = _pt_tmp_101 + _pt_tmp_104
    _pt_tmp_99 = 1 * _pt_tmp_100
    del _pt_tmp_101, _pt_tmp_104
    _pt_tmp_98 = (
        _pt_tmp_99[_pt_tmp_43, _pt_tmp_44]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_3, in_0=_pt_tmp_99, in_1=_pt_tmp_43, in_2=_pt_tmp_44
        )["out"]
    )
    del _pt_tmp_100
    _pt_tmp_97 = (
        actx.np.where(_pt_tmp_15, _pt_tmp_98, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_4, _in0=_pt_tmp_15, _in1=_pt_tmp_98)["out"]
    )
    del _pt_tmp_99
    _pt_tmp_97 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_97)
    del _pt_tmp_98
    _pt_tmp_97 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_97)
    _pt_tmp_96 = 0 + _pt_tmp_97
    _pt_tmp_122 = _pt_tmp_84 + _pt_tmp_82
    del _pt_tmp_97
    _pt_tmp_121 = 0.5 * _pt_tmp_122
    del _pt_tmp_82, _pt_tmp_84
    _pt_tmp_120 = (
        _pt_tmp_121 * _pt_data_21
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_121, _in1=_pt_data_21)["out"]
    )
    del _pt_tmp_122
    _pt_tmp_128 = _pt_tmp_62 - _pt_tmp_58
    _pt_tmp_127 = (
        _pt_tmp_128 * _pt_data_21
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_128, _in1=_pt_data_21)["out"]
    )
    del _pt_tmp_58, _pt_tmp_62
    _pt_tmp_130 = _pt_tmp_71 - _pt_tmp_69
    del _pt_tmp_128
    _pt_tmp_129 = (
        _pt_tmp_130 * _pt_data_22
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_130, _in1=_pt_data_22)["out"]
    )
    del _pt_tmp_69, _pt_tmp_71
    _pt_tmp_126 = _pt_tmp_127 + _pt_tmp_129
    del _pt_tmp_130
    _pt_tmp_132 = _pt_tmp_78 - _pt_tmp_76
    del _pt_tmp_127, _pt_tmp_129
    _pt_tmp_131 = (
        _pt_tmp_132 * _pt_data_23
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_132, _in1=_pt_data_23)["out"]
    )
    del _pt_tmp_76, _pt_tmp_78
    _pt_tmp_125 = _pt_tmp_126 + _pt_tmp_131
    del _pt_tmp_132
    _pt_tmp_124 = 0.5 * _pt_tmp_125
    del _pt_tmp_126, _pt_tmp_131
    _pt_tmp_123 = (
        _pt_tmp_124 * _pt_data_21
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_124, _in1=_pt_data_21)["out"]
    )
    del _pt_tmp_125
    _pt_tmp_119 = _pt_tmp_120 + _pt_tmp_123
    _pt_tmp_118 = 1 * _pt_tmp_119
    del _pt_tmp_120, _pt_tmp_123
    _pt_tmp_117 = (
        _pt_tmp_118[_pt_tmp_86, _pt_tmp_87]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_10, in_0=_pt_tmp_118, in_1=_pt_tmp_86, in_2=_pt_tmp_87
        )["out"]
    )
    del _pt_tmp_119
    _pt_tmp_116 = (
        actx.np.where(_pt_tmp_49, _pt_tmp_117, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_4, _in0=_pt_tmp_49, _in1=_pt_tmp_117)["out"]
    )
    del _pt_tmp_118
    _pt_tmp_116 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_116)
    del _pt_tmp_117
    _pt_tmp_116 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_116)
    _pt_tmp_115 = 0 + _pt_tmp_116
    _pt_tmp_114 = 0 + _pt_tmp_115
    del _pt_tmp_116
    _pt_tmp_95 = _pt_tmp_96 + _pt_tmp_114
    del _pt_tmp_115
    _pt_tmp_94 = actx.np.reshape(_pt_tmp_95, (4, 162000, 10))
    del _pt_tmp_114, _pt_tmp_96
    _pt_tmp_94 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_94)
    del _pt_tmp_95
    _pt_tmp_93 = actx.einsum("ijk, jl, jlk -> li", _pt_data_3, _pt_tmp_9, _pt_tmp_94)
    _pt_tmp_93 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_93)
    del _pt_tmp_94
    _pt_tmp_89 = _pt_tmp_90 + _pt_tmp_93
    _pt_tmp_88 = actx.einsum("i, jk, ik -> ij", _pt_tmp_1, _pt_data_0, _pt_tmp_89)
    del _pt_tmp_90, _pt_tmp_93
    _pt_tmp_88 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_88)
    del _pt_tmp_89
    _pt_tmp_136 = _pt_tmp_92[1]
    _pt_tmp_135 = -1 * _pt_tmp_136
    _pt_tmp_145 = (
        _pt_tmp_102 * _pt_data_10
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_1, _in0=_pt_tmp_102, _in1=_pt_data_10)["out"]
    )
    del _pt_tmp_136
    _pt_tmp_146 = (
        _pt_tmp_105 * _pt_data_10
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_1, _in0=_pt_tmp_105, _in1=_pt_data_10)["out"]
    )
    _pt_tmp_144 = _pt_tmp_145 + _pt_tmp_146
    _pt_tmp_143 = 1 * _pt_tmp_144
    del _pt_tmp_145, _pt_tmp_146
    _pt_tmp_142 = (
        _pt_tmp_143[_pt_tmp_43, _pt_tmp_44]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_3, in_0=_pt_tmp_143, in_1=_pt_tmp_43, in_2=_pt_tmp_44
        )["out"]
    )
    del _pt_tmp_144
    _pt_tmp_141 = (
        actx.np.where(_pt_tmp_15, _pt_tmp_142, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_4, _in0=_pt_tmp_15, _in1=_pt_tmp_142)["out"]
    )
    del _pt_tmp_143
    _pt_tmp_141 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_141)
    del _pt_tmp_142
    _pt_tmp_141 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_141)
    _pt_tmp_140 = 0 + _pt_tmp_141
    _pt_tmp_153 = (
        _pt_tmp_121 * _pt_data_22
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_121, _in1=_pt_data_22)["out"]
    )
    del _pt_tmp_141
    _pt_tmp_154 = (
        _pt_tmp_124 * _pt_data_22
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_124, _in1=_pt_data_22)["out"]
    )
    _pt_tmp_152 = _pt_tmp_153 + _pt_tmp_154
    _pt_tmp_151 = 1 * _pt_tmp_152
    del _pt_tmp_153, _pt_tmp_154
    _pt_tmp_150 = (
        _pt_tmp_151[_pt_tmp_86, _pt_tmp_87]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_10, in_0=_pt_tmp_151, in_1=_pt_tmp_86, in_2=_pt_tmp_87
        )["out"]
    )
    del _pt_tmp_152
    _pt_tmp_149 = (
        actx.np.where(_pt_tmp_49, _pt_tmp_150, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_4, _in0=_pt_tmp_49, _in1=_pt_tmp_150)["out"]
    )
    del _pt_tmp_151
    _pt_tmp_149 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_149)
    del _pt_tmp_150
    _pt_tmp_149 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_149)
    _pt_tmp_148 = 0 + _pt_tmp_149
    _pt_tmp_147 = 0 + _pt_tmp_148
    del _pt_tmp_149
    _pt_tmp_139 = _pt_tmp_140 + _pt_tmp_147
    del _pt_tmp_148
    _pt_tmp_138 = actx.np.reshape(_pt_tmp_139, (4, 162000, 10))
    del _pt_tmp_140, _pt_tmp_147
    _pt_tmp_138 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_138)
    del _pt_tmp_139
    _pt_tmp_137 = actx.einsum(
        "ijk, jl, jlk -> li", _pt_data_3, _pt_tmp_9, _pt_tmp_138
    )
    _pt_tmp_137 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_137)
    del _pt_tmp_138
    _pt_tmp_134 = _pt_tmp_135 + _pt_tmp_137
    _pt_tmp_133 = actx.einsum("i, jk, ik -> ij", _pt_tmp_1, _pt_data_0, _pt_tmp_134)
    del _pt_tmp_135, _pt_tmp_137
    _pt_tmp_133 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_133)
    del _pt_tmp_134
    _pt_tmp_158 = _pt_tmp_92[2]
    _pt_tmp_157 = -1 * _pt_tmp_158
    del _pt_tmp_92
    _pt_tmp_167 = (
        _pt_tmp_102 * _pt_data_11
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_1, _in0=_pt_tmp_102, _in1=_pt_data_11)["out"]
    )
    del _pt_tmp_158
    _pt_tmp_168 = (
        _pt_tmp_105 * _pt_data_11
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_1, _in0=_pt_tmp_105, _in1=_pt_data_11)["out"]
    )
    del _pt_tmp_102
    _pt_tmp_166 = _pt_tmp_167 + _pt_tmp_168
    del _pt_tmp_105
    _pt_tmp_165 = 1 * _pt_tmp_166
    del _pt_tmp_167, _pt_tmp_168
    _pt_tmp_164 = (
        _pt_tmp_165[_pt_tmp_43, _pt_tmp_44]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_3, in_0=_pt_tmp_165, in_1=_pt_tmp_43, in_2=_pt_tmp_44
        )["out"]
    )
    del _pt_tmp_166
    _pt_tmp_163 = (
        actx.np.where(_pt_tmp_15, _pt_tmp_164, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_4, _in0=_pt_tmp_15, _in1=_pt_tmp_164)["out"]
    )
    del _pt_tmp_165, _pt_tmp_43, _pt_tmp_44
    _pt_tmp_163 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_163)
    del _pt_tmp_15, _pt_tmp_164
    _pt_tmp_163 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_163)
    _pt_tmp_162 = 0 + _pt_tmp_163
    _pt_tmp_175 = (
        _pt_tmp_121 * _pt_data_23
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_121, _in1=_pt_data_23)["out"]
    )
    del _pt_tmp_163
    _pt_tmp_176 = (
        _pt_tmp_124 * _pt_data_23
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_124, _in1=_pt_data_23)["out"]
    )
    del _pt_tmp_121
    _pt_tmp_174 = _pt_tmp_175 + _pt_tmp_176
    del _pt_tmp_124
    _pt_tmp_173 = 1 * _pt_tmp_174
    del _pt_tmp_175, _pt_tmp_176
    _pt_tmp_172 = (
        _pt_tmp_173[_pt_tmp_86, _pt_tmp_87]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_10, in_0=_pt_tmp_173, in_1=_pt_tmp_86, in_2=_pt_tmp_87
        )["out"]
    )
    del _pt_tmp_174
    _pt_tmp_171 = (
        actx.np.where(_pt_tmp_49, _pt_tmp_172, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_4, _in0=_pt_tmp_49, _in1=_pt_tmp_172)["out"]
    )
    del _pt_tmp_173, _pt_tmp_86, _pt_tmp_87
    _pt_tmp_171 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_171)
    del _pt_tmp_172, _pt_tmp_49
    _pt_tmp_171 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_171)
    _pt_tmp_170 = 0 + _pt_tmp_171
    _pt_tmp_169 = 0 + _pt_tmp_170
    del _pt_tmp_171
    _pt_tmp_161 = _pt_tmp_162 + _pt_tmp_169
    del _pt_tmp_170
    _pt_tmp_160 = actx.np.reshape(_pt_tmp_161, (4, 162000, 10))
    del _pt_tmp_162, _pt_tmp_169
    _pt_tmp_160 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_160)
    del _pt_tmp_161
    _pt_tmp_159 = actx.einsum(
        "ijk, jl, jlk -> li", _pt_data_3, _pt_tmp_9, _pt_tmp_160
    )
    _pt_tmp_159 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_159)
    del _pt_tmp_160, _pt_tmp_9
    _pt_tmp_156 = _pt_tmp_157 + _pt_tmp_159
    _pt_tmp_155 = actx.einsum("i, jk, ik -> ij", _pt_tmp_1, _pt_data_0, _pt_tmp_156)
    del _pt_tmp_157, _pt_tmp_159
    _pt_tmp_155 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_155)
    del _pt_tmp_1, _pt_tmp_156
    _pt_tmp = make_obj_array([_pt_tmp_0, _pt_tmp_88, _pt_tmp_133, _pt_tmp_155])
    return _pt_tmp
    del _pt_tmp_0, _pt_tmp_133, _pt_tmp_155, _pt_tmp_88


@dataclass(frozen=True)
class RHSInvoker:
    actx: ArrayContext

    @cached_property
    def npzfile(self):
        import os

        from immutables import Map

        kw_to_ary = np.load(
            os.path.join(
                get_actx_dgfem_suite_path(), "suite/wave_3D_P3/literals.npz"
            )
        )
        return Map(
            {
                kw: self.actx.freeze(self.actx.from_numpy(ary))
                for kw, ary in kw_to_ary.items()
            }
        )

    @memoize_method
    def _get_compiled_rhs_inner(self):
        return self.actx.compile(
            lambda *args, **kwargs: _rhs_inner(
                self.actx, self.npzfile, *args, **kwargs
            )
        )

    @memoize_method
    def _get_output_template(self):
        import os
        from pickle import load

        import pytato as pt
        from meshmode.dof_array import array_context_for_pickling

        fpath = os.path.join(
            get_actx_dgfem_suite_path(), "suite/wave_3D_P3/ref_outputs.pkl"
        )
        with open(fpath, "rb") as fp:
            with array_context_for_pickling(self.actx):
                output_template = load(fp)

        def _convert_to_symbolic_array(ary):
            return pt.zeros(ary.shape, ary.dtype)

        # convert to symbolic array to not free the memory corresponding to
        # output_template
        return rec_map_array_container(_convert_to_symbolic_array, output_template)

    @memoize_method
    def _get_key_to_pos_in_output_template(self):
        from arraycontext.impl.pytato.compile import (
            _ary_container_key_stringifier,
        )

        output_keys = set()
        output_template = self._get_output_template()

        def _as_dict_of_named_arrays(keys, ary):
            output_keys.add(keys)
            return ary

        rec_keyed_map_array_container(_as_dict_of_named_arrays, output_template)

        return Map(
            {
                output_key: i
                for i, output_key in enumerate(
                    sorted(output_keys, key=_ary_container_key_stringifier)
                )
            }
        )

    @cached_property
    def _rhs_inner_argument_names(self):
        return {
            "_actx_in_1_u_0",
            "_actx_in_1_v_0_0",
            "_actx_in_1_v_1_0",
            "_actx_in_1_v_2_0",
        }

    def __call__(self, *args, **kwargs):
        from arraycontext.impl.pytato.compile import (
            _ary_container_key_stringifier,
            _get_arg_id_to_arg_and_arg_id_to_descr,
        )

        arg_id_to_arg, _ = _get_arg_id_to_arg_and_arg_id_to_descr(args, kwargs)
        input_kwargs_to_rhs_inner = {
            "_actx_in_" + _ary_container_key_stringifier(arg_id): arg
            for arg_id, arg in arg_id_to_arg.items()
        }

        input_kwargs_to_rhs_inner = {
            kw: input_kwargs_to_rhs_inner[kw]
            for kw in self._rhs_inner_argument_names
        }

        compiled_rhs_inner = self._get_compiled_rhs_inner()
        result_as_np_obj_array = compiled_rhs_inner(**input_kwargs_to_rhs_inner)

        output_template = self._get_output_template()

        if is_array_container_type(output_template.__class__):
            keys_to_pos = self._get_key_to_pos_in_output_template()

            def to_output_template(keys, _):
                return result_as_np_obj_array[keys_to_pos[keys]]

            return rec_keyed_map_array_container(
                to_output_template, self._get_output_template()
            )
        else:
            from pytato.array import Array

            assert isinstance(output_template, Array)
            assert result_as_np_obj_array.shape == (1,)
            return result_as_np_obj_array[0]
