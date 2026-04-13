from pytools.obj_array import make_obj_array
import numpy as np
from arraycontext import make_loopy_program
import loopy as lp
from meshmode.transform_metadata import DiscretizationDOFAxisTag
from meshmode.transform_metadata import DiscretizationElementAxisTag
from meshmode.transform_metadata import DiscretizationFaceAxisTag
from meshmode.transform_metadata import FirstAxisIsElementsTag
from pytools import memoize_method
from functools import cached_property
from constantdict import constantdict
from arraycontext import ArrayContext, is_array_container_type
from dataclasses import dataclass
from arraycontext.container.traversal import (
    rec_map_array_container,
    rec_keyed_map_array_container,
)
from actx_dgfem_suite.utils import get_actx_dgfem_suite_path


def _rhs_inner(
    actx,
    npzfile,
    *,
    _actx_in_1_0_0,
    _actx_in_1_1_0,
    _actx_in_1_2_0,
    _actx_in_1_3_0,
    _actx_in_1_4_0,
    _actx_in_1_5_0,
):
    _pt_t_unit = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 637199 and 0 <= _1 <= 9 }",
        "out[_0, _1] = in_0[in_1[_0] % 4, _1]",
        [
            lp.GlobalArg("out", dtype=np.int64, shape=(637200, 10)),
            lp.GlobalArg("in_0", dtype=np.int64, shape=(4, 10)),
            lp.GlobalArg("in_1", dtype=np.int8, shape=(637200,)),
        ],
    )
    _pt_t_unit_0 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 637199 and 0 <= _1 <= 9 }",
        "out[_0, _1] = in_0[in_1[_0, 0] % 162000, in_2[_0, _1] % 20]",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(637200, 10)),
            lp.GlobalArg("in_0", dtype=np.float64, shape=(162000, 20)),
            lp.GlobalArg("in_1", dtype=np.int32, shape=(637200, 1)),
            lp.GlobalArg("in_2", dtype=np.int64, shape=(637200, 10)),
        ],
    )
    _pt_t_unit_1 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 637199 and 0 <= _1 <= 9 }",
        "out[_0, _1] = in_0[in_1[_0] % 3, _1]",
        [
            lp.GlobalArg("out", dtype=np.int64, shape=(637200, 10)),
            lp.GlobalArg("in_0", dtype=np.int64, shape=(3, 10)),
            lp.GlobalArg("in_1", dtype=np.int8, shape=(637200,)),
        ],
    )
    _pt_t_unit_10 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 647999 and 0 <= _1 <= 9 }",
        "out[_0, _1] = in_0[in_1[_0, 0] % 10800, in_2[_0, _1] % 10]",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(648000, 10)),
            lp.GlobalArg("in_0", dtype=np.float64, shape=(10800, 10)),
            lp.GlobalArg("in_1", dtype=np.int32, shape=(648000, 1)),
            lp.GlobalArg("in_2", dtype=np.int64, shape=(648000, 10)),
        ],
    )
    _pt_t_unit_2 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 637199 and 0 <= _1 <= 9 }",
        "out[_0, _1] = in_0[in_1[_0, 0] % 637200, in_2[_0, _1] % 10]",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(637200, 10)),
            lp.GlobalArg("in_0", dtype=np.float64, shape=(637200, 10)),
            lp.GlobalArg("in_1", dtype=np.int32, shape=(637200, 1)),
            lp.GlobalArg("in_2", dtype=np.int64, shape=(637200, 10)),
        ],
    )
    _pt_t_unit_3 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 637199 and 0 <= _1 <= 9 }",
        "out[_0, _1] = _in0[_0, 0]*_in1[_0, _1]",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(637200, 10)),
            lp.GlobalArg("_in0", dtype=np.float64, shape=(637200, 1)),
            lp.GlobalArg("_in1", dtype=np.float64, shape=(637200, 10)),
        ],
    )
    _pt_t_unit_4 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 647999 and 0 <= _1 <= 9 }",
        "out[_0, _1] = in_0[in_1[_0] % 1, _1]",
        [
            lp.GlobalArg("out", dtype=np.int64, shape=(648000, 10)),
            lp.GlobalArg("in_0", dtype=np.int64, shape=(1, 10)),
            lp.GlobalArg("in_1", dtype=np.int8, shape=(648000,)),
        ],
    )
    _pt_t_unit_5 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 647999 and 0 <= _1 <= 9 }",
        "out[_0, _1] = in_0[in_1[_0, 0] % 637200, in_2[_0, _1] % 10]",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(648000, 10)),
            lp.GlobalArg("in_0", dtype=np.float64, shape=(637200, 10)),
            lp.GlobalArg("in_1", dtype=np.int32, shape=(648000, 1)),
            lp.GlobalArg("in_2", dtype=np.int64, shape=(648000, 10)),
        ],
    )
    _pt_t_unit_6 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 647999 and 0 <= _1 <= 9 }",
        "out[_0, _1] = _in1[_0, _1] if _in0[_0, 0] else 0",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(648000, 10)),
            lp.GlobalArg("_in0", dtype=np.int8, shape=(648000, 1)),
            lp.GlobalArg("_in1", dtype=np.float64, shape=(648000, 10)),
        ],
    )
    _pt_t_unit_7 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 10799 and 0 <= _1 <= 9 }",
        "out[_0, _1] = in_0[in_1[_0] % 4, _1]",
        [
            lp.GlobalArg("out", dtype=np.int64, shape=(10800, 10)),
            lp.GlobalArg("in_0", dtype=np.int64, shape=(4, 10)),
            lp.GlobalArg("in_1", dtype=np.int8, shape=(10800,)),
        ],
    )
    _pt_t_unit_8 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 10799 and 0 <= _1 <= 9 }",
        "out[_0, _1] = in_0[in_1[_0, 0] % 162000, in_2[_0, _1] % 20]",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(10800, 10)),
            lp.GlobalArg("in_0", dtype=np.float64, shape=(162000, 20)),
            lp.GlobalArg("in_1", dtype=np.int32, shape=(10800, 1)),
            lp.GlobalArg("in_2", dtype=np.int64, shape=(10800, 10)),
        ],
    )
    _pt_t_unit_9 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 10799 and 0 <= _1 <= 9 }",
        "out[_0, _1] = _in0[_0, 0]*_in1[_0, _1]",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(10800, 10)),
            lp.GlobalArg("_in0", dtype=np.float64, shape=(10800, 1)),
            lp.GlobalArg("_in1", dtype=np.float64, shape=(10800, 10)),
        ],
    )
    _pt_data = actx.thaw(npzfile["_pt_data"])
    _pt_tmp_9 = _pt_data[1]
    _pt_tmp_8 = _pt_tmp_9[:, :, 0]
    _pt_data_0 = actx.thaw(npzfile["_pt_data_0"])
    del _pt_tmp_9
    _pt_tmp_7 = actx.einsum("ij,ikl,jl->jk", _pt_tmp_8, _pt_data_0, _actx_in_1_5_0)
    _pt_tmp_7 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_7)
    _pt_tmp_6 = 1.0 * _pt_tmp_7
    _pt_tmp_5 = 0.0 + _pt_tmp_6
    del _pt_tmp_7
    _pt_tmp_13 = _pt_data[2]
    del _pt_tmp_6
    _pt_tmp_12 = _pt_tmp_13[:, :, 0]
    _pt_tmp_11 = actx.einsum("ij,ikl,jl->jk", _pt_tmp_12, _pt_data_0, _actx_in_1_4_0)
    del _pt_tmp_13
    _pt_tmp_11 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_11)
    _pt_tmp_10 = -1.0 * _pt_tmp_11
    _pt_tmp_4 = _pt_tmp_5 + _pt_tmp_10
    del _pt_tmp_11
    _pt_tmp_3 = 0.0 - _pt_tmp_4
    del _pt_tmp_10, _pt_tmp_5
    _pt_tmp_2 = -1 * _pt_tmp_3
    del _pt_tmp_4
    _pt_data_1 = actx.thaw(npzfile["_pt_data_1"])
    del _pt_tmp_3
    _pt_tmp_16 = 1.0 / _pt_data_1
    _pt_tmp_15 = _pt_tmp_16[:, 0]
    _pt_data_2 = actx.thaw(npzfile["_pt_data_2"])
    del _pt_tmp_16
    _pt_data_3 = actx.thaw(npzfile["_pt_data_3"])
    _pt_data_4 = actx.thaw(npzfile["_pt_data_4"])
    _pt_tmp_19 = actx.np.reshape(_pt_data_4, (4, 162000, 1))
    _pt_tmp_19 = actx.tag_axis(1, (DiscretizationElementAxisTag(),), _pt_tmp_19)
    _pt_tmp_18 = _pt_tmp_19[:, :, 0]
    _pt_data_5 = actx.thaw(npzfile["_pt_data_5"])
    del _pt_tmp_19
    _pt_tmp_25 = actx.np.reshape(_pt_data_5, (648000, 1))
    _pt_data_6 = actx.thaw(npzfile["_pt_data_6"])
    _pt_tmp_31 = 1.0 * _pt_data_6
    _pt_data_7 = actx.thaw(npzfile["_pt_data_7"])
    _pt_tmp_39 = actx.np.reshape(_pt_data_7, (637200, 1))
    _pt_data_8 = actx.thaw(npzfile["_pt_data_8"])
    _pt_data_9 = actx.thaw(npzfile["_pt_data_9"])
    _pt_tmp_40 = (
        _pt_data_8[_pt_data_9]
        if actx.permits_advanced_indexing
        else actx.call_loopy(_pt_t_unit, in_0=_pt_data_8, in_1=_pt_data_9)["out"]
    )
    _pt_tmp_38 = (
        _actx_in_1_5_0[_pt_tmp_39, _pt_tmp_40]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_actx_in_1_5_0, in_1=_pt_tmp_39, in_2=_pt_tmp_40
        )["out"]
    )
    _pt_tmp_38 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_38)
    _pt_tmp_38 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_38)
    _pt_tmp_37 = 0.0 + _pt_tmp_38
    _pt_data_10 = actx.thaw(npzfile["_pt_data_10"])
    del _pt_tmp_38
    _pt_tmp_41 = actx.np.reshape(_pt_data_10, (637200, 1))
    _pt_data_11 = actx.thaw(npzfile["_pt_data_11"])
    _pt_data_12 = actx.thaw(npzfile["_pt_data_12"])
    _pt_tmp_42 = (
        _pt_data_11[_pt_data_12]
        if actx.permits_advanced_indexing
        else actx.call_loopy(_pt_t_unit_1, in_0=_pt_data_11, in_1=_pt_data_12)["out"]
    )
    _pt_tmp_36 = (
        _pt_tmp_37[_pt_tmp_41, _pt_tmp_42]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_37, in_1=_pt_tmp_41, in_2=_pt_tmp_42
        )["out"]
    )
    _pt_tmp_36 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_36)
    _pt_tmp_36 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_36)
    _pt_tmp_35 = 0.0 + _pt_tmp_36
    _pt_tmp_34 = _pt_tmp_35 - _pt_tmp_37
    del _pt_tmp_36
    _pt_tmp_33 = 1.0 * _pt_tmp_34
    del _pt_tmp_35, _pt_tmp_37
    _pt_data_13 = actx.thaw(npzfile["_pt_data_13"])
    _pt_tmp_47 = 1.0 * _pt_data_13
    _pt_tmp_52 = (
        _actx_in_1_1_0[_pt_tmp_39, _pt_tmp_40]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_actx_in_1_1_0, in_1=_pt_tmp_39, in_2=_pt_tmp_40
        )["out"]
    )
    _pt_tmp_52 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_52)
    _pt_tmp_52 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_52)
    _pt_tmp_51 = 0.0 + _pt_tmp_52
    _pt_tmp_50 = (
        _pt_tmp_51[_pt_tmp_41, _pt_tmp_42]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_51, in_1=_pt_tmp_41, in_2=_pt_tmp_42
        )["out"]
    )
    del _pt_tmp_52
    _pt_tmp_50 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_50)
    _pt_tmp_50 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_50)
    _pt_tmp_49 = 0.0 + _pt_tmp_50
    _pt_tmp_48 = _pt_tmp_49 - _pt_tmp_51
    del _pt_tmp_50
    _pt_tmp_46 = (
        _pt_tmp_47 * _pt_tmp_48
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_47, _in1=_pt_tmp_48)["out"]
    )
    del _pt_tmp_49, _pt_tmp_51
    _pt_tmp_45 = 0.0 + _pt_tmp_46
    _pt_tmp_54 = -1.0 * _pt_data_6
    del _pt_tmp_46
    _pt_tmp_59 = (
        _actx_in_1_0_0[_pt_tmp_39, _pt_tmp_40]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_actx_in_1_0_0, in_1=_pt_tmp_39, in_2=_pt_tmp_40
        )["out"]
    )
    _pt_tmp_59 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_59)
    _pt_tmp_59 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_59)
    _pt_tmp_58 = 0.0 + _pt_tmp_59
    _pt_tmp_57 = (
        _pt_tmp_58[_pt_tmp_41, _pt_tmp_42]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_58, in_1=_pt_tmp_41, in_2=_pt_tmp_42
        )["out"]
    )
    del _pt_tmp_59
    _pt_tmp_57 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_57)
    _pt_tmp_57 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_57)
    _pt_tmp_56 = 0.0 + _pt_tmp_57
    _pt_tmp_55 = _pt_tmp_56 - _pt_tmp_58
    del _pt_tmp_57
    _pt_tmp_53 = (
        _pt_tmp_54 * _pt_tmp_55
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_54, _in1=_pt_tmp_55)["out"]
    )
    del _pt_tmp_56, _pt_tmp_58
    _pt_tmp_44 = _pt_tmp_45 + _pt_tmp_53
    _pt_tmp_43 = 0.5 * _pt_tmp_44
    del _pt_tmp_45, _pt_tmp_53
    _pt_tmp_32 = _pt_tmp_33 - _pt_tmp_43
    del _pt_tmp_44
    _pt_tmp_30 = (
        _pt_tmp_31 * _pt_tmp_32
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_31, _in1=_pt_tmp_32)["out"]
    )
    del _pt_tmp_33, _pt_tmp_43
    _pt_tmp_29 = 0.0 + _pt_tmp_30
    _pt_data_14 = actx.thaw(npzfile["_pt_data_14"])
    del _pt_tmp_30
    _pt_tmp_61 = -1.0 * _pt_data_14
    _pt_tmp_68 = (
        _actx_in_1_4_0[_pt_tmp_39, _pt_tmp_40]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_actx_in_1_4_0, in_1=_pt_tmp_39, in_2=_pt_tmp_40
        )["out"]
    )
    _pt_tmp_68 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_68)
    _pt_tmp_68 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_68)
    _pt_tmp_67 = 0.0 + _pt_tmp_68
    _pt_tmp_66 = (
        _pt_tmp_67[_pt_tmp_41, _pt_tmp_42]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_67, in_1=_pt_tmp_41, in_2=_pt_tmp_42
        )["out"]
    )
    del _pt_tmp_68
    _pt_tmp_66 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_66)
    _pt_tmp_66 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_66)
    _pt_tmp_65 = 0.0 + _pt_tmp_66
    _pt_tmp_64 = _pt_tmp_65 - _pt_tmp_67
    del _pt_tmp_66
    _pt_tmp_63 = 1.0 * _pt_tmp_64
    del _pt_tmp_65, _pt_tmp_67
    _pt_tmp_73 = -1.0 * _pt_data_13
    _pt_tmp_78 = (
        _actx_in_1_2_0[_pt_tmp_39, _pt_tmp_40]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_actx_in_1_2_0, in_1=_pt_tmp_39, in_2=_pt_tmp_40
        )["out"]
    )
    _pt_tmp_78 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_78)
    _pt_tmp_78 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_78)
    _pt_tmp_77 = 0.0 + _pt_tmp_78
    _pt_tmp_76 = (
        _pt_tmp_77[_pt_tmp_41, _pt_tmp_42]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_77, in_1=_pt_tmp_41, in_2=_pt_tmp_42
        )["out"]
    )
    del _pt_tmp_78
    _pt_tmp_76 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_76)
    _pt_tmp_76 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_76)
    _pt_tmp_75 = 0.0 + _pt_tmp_76
    _pt_tmp_74 = _pt_tmp_75 - _pt_tmp_77
    del _pt_tmp_76
    _pt_tmp_72 = (
        _pt_tmp_73 * _pt_tmp_74
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_73, _in1=_pt_tmp_74)["out"]
    )
    del _pt_tmp_75, _pt_tmp_77
    _pt_tmp_71 = 0.0 + _pt_tmp_72
    _pt_tmp_80 = 1.0 * _pt_data_14
    del _pt_tmp_72
    _pt_tmp_79 = (
        _pt_tmp_80 * _pt_tmp_55
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_80, _in1=_pt_tmp_55)["out"]
    )
    _pt_tmp_70 = _pt_tmp_71 + _pt_tmp_79
    _pt_tmp_69 = 0.5 * _pt_tmp_70
    del _pt_tmp_71, _pt_tmp_79
    _pt_tmp_62 = _pt_tmp_63 - _pt_tmp_69
    del _pt_tmp_70
    _pt_tmp_60 = (
        _pt_tmp_61 * _pt_tmp_62
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_61, _in1=_pt_tmp_62)["out"]
    )
    del _pt_tmp_63, _pt_tmp_69
    _pt_tmp_28 = _pt_tmp_29 + _pt_tmp_60
    _pt_tmp_27 = -0.5 * _pt_tmp_28
    del _pt_tmp_29, _pt_tmp_60
    _pt_data_15 = actx.thaw(npzfile["_pt_data_15"])
    del _pt_tmp_28
    _pt_tmp_81 = actx.np.reshape(_pt_data_15, (648000, 1))
    _pt_data_16 = actx.thaw(npzfile["_pt_data_16"])
    _pt_data_17 = actx.thaw(npzfile["_pt_data_17"])
    _pt_tmp_82 = (
        _pt_data_16[_pt_data_17]
        if actx.permits_advanced_indexing
        else actx.call_loopy(_pt_t_unit_4, in_0=_pt_data_16, in_1=_pt_data_17)["out"]
    )
    _pt_tmp_26 = (
        _pt_tmp_27[_pt_tmp_81, _pt_tmp_82]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_5, in_0=_pt_tmp_27, in_1=_pt_tmp_81, in_2=_pt_tmp_82
        )["out"]
    )
    _pt_tmp_24 = (
        actx.np.where(_pt_tmp_25, _pt_tmp_26, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_6, _in0=_pt_tmp_25, _in1=_pt_tmp_26)["out"]
    )
    del _pt_tmp_27
    _pt_tmp_24 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_24)
    del _pt_tmp_26
    _pt_tmp_24 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_24)
    _pt_tmp_23 = 0.0 + _pt_tmp_24
    _pt_tmp_22 = 0.0 + _pt_tmp_23
    del _pt_tmp_24
    _pt_data_18 = actx.thaw(npzfile["_pt_data_18"])
    del _pt_tmp_23
    _pt_tmp_89 = actx.np.reshape(_pt_data_18, (648000, 1))
    _pt_data_19 = actx.thaw(npzfile["_pt_data_19"])
    _pt_tmp_95 = 1.0 * _pt_data_19
    _pt_data_20 = actx.thaw(npzfile["_pt_data_20"])
    _pt_tmp_101 = actx.np.reshape(_pt_data_20, (10800, 1))
    _pt_data_21 = actx.thaw(npzfile["_pt_data_21"])
    _pt_tmp_102 = (
        _pt_data_8[_pt_data_21]
        if actx.permits_advanced_indexing
        else actx.call_loopy(_pt_t_unit_7, in_0=_pt_data_8, in_1=_pt_data_21)["out"]
    )
    _pt_tmp_100 = (
        _actx_in_1_5_0[_pt_tmp_101, _pt_tmp_102]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_8, in_0=_actx_in_1_5_0, in_1=_pt_tmp_101, in_2=_pt_tmp_102
        )["out"]
    )
    _pt_tmp_100 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_100)
    _pt_tmp_100 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_100)
    _pt_tmp_99 = 0.0 + _pt_tmp_100
    _pt_tmp_98 = _pt_tmp_99 - _pt_tmp_99
    del _pt_tmp_100
    _pt_tmp_97 = 1.0 * _pt_tmp_98
    del _pt_tmp_99
    _pt_data_22 = actx.thaw(npzfile["_pt_data_22"])
    _pt_tmp_107 = 1.0 * _pt_data_22
    _pt_tmp_111 = (
        _actx_in_1_1_0[_pt_tmp_101, _pt_tmp_102]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_8, in_0=_actx_in_1_1_0, in_1=_pt_tmp_101, in_2=_pt_tmp_102
        )["out"]
    )
    _pt_tmp_111 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_111)
    _pt_tmp_111 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_111)
    _pt_tmp_110 = 0.0 + _pt_tmp_111
    _pt_tmp_109 = -1 * _pt_tmp_110
    del _pt_tmp_111
    _pt_tmp_108 = _pt_tmp_109 - _pt_tmp_110
    _pt_tmp_106 = (
        _pt_tmp_107 * _pt_tmp_108
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_107, _in1=_pt_tmp_108)["out"]
    )
    del _pt_tmp_109, _pt_tmp_110
    _pt_tmp_105 = 0.0 + _pt_tmp_106
    _pt_tmp_113 = -1.0 * _pt_data_19
    del _pt_tmp_106
    _pt_tmp_117 = (
        _actx_in_1_0_0[_pt_tmp_101, _pt_tmp_102]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_8, in_0=_actx_in_1_0_0, in_1=_pt_tmp_101, in_2=_pt_tmp_102
        )["out"]
    )
    _pt_tmp_117 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_117)
    _pt_tmp_117 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_117)
    _pt_tmp_116 = 0.0 + _pt_tmp_117
    _pt_tmp_115 = -1 * _pt_tmp_116
    del _pt_tmp_117
    _pt_tmp_114 = _pt_tmp_115 - _pt_tmp_116
    _pt_tmp_112 = (
        _pt_tmp_113 * _pt_tmp_114
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_113, _in1=_pt_tmp_114)["out"]
    )
    del _pt_tmp_115, _pt_tmp_116
    _pt_tmp_104 = _pt_tmp_105 + _pt_tmp_112
    _pt_tmp_103 = 0.5 * _pt_tmp_104
    del _pt_tmp_105, _pt_tmp_112
    _pt_tmp_96 = _pt_tmp_97 - _pt_tmp_103
    del _pt_tmp_104
    _pt_tmp_94 = (
        _pt_tmp_95 * _pt_tmp_96
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_95, _in1=_pt_tmp_96)["out"]
    )
    del _pt_tmp_103, _pt_tmp_97
    _pt_tmp_93 = 0.0 + _pt_tmp_94
    _pt_data_23 = actx.thaw(npzfile["_pt_data_23"])
    del _pt_tmp_94
    _pt_tmp_119 = -1.0 * _pt_data_23
    _pt_tmp_124 = (
        _actx_in_1_4_0[_pt_tmp_101, _pt_tmp_102]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_8, in_0=_actx_in_1_4_0, in_1=_pt_tmp_101, in_2=_pt_tmp_102
        )["out"]
    )
    _pt_tmp_124 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_124)
    _pt_tmp_124 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_124)
    _pt_tmp_123 = 0.0 + _pt_tmp_124
    _pt_tmp_122 = _pt_tmp_123 - _pt_tmp_123
    del _pt_tmp_124
    _pt_tmp_121 = 1.0 * _pt_tmp_122
    del _pt_tmp_123
    _pt_tmp_129 = -1.0 * _pt_data_22
    _pt_tmp_133 = (
        _actx_in_1_2_0[_pt_tmp_101, _pt_tmp_102]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_8, in_0=_actx_in_1_2_0, in_1=_pt_tmp_101, in_2=_pt_tmp_102
        )["out"]
    )
    _pt_tmp_133 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_133)
    _pt_tmp_133 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_133)
    _pt_tmp_132 = 0.0 + _pt_tmp_133
    _pt_tmp_131 = -1 * _pt_tmp_132
    del _pt_tmp_133
    _pt_tmp_130 = _pt_tmp_131 - _pt_tmp_132
    _pt_tmp_128 = (
        _pt_tmp_129 * _pt_tmp_130
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_129, _in1=_pt_tmp_130)["out"]
    )
    del _pt_tmp_131, _pt_tmp_132
    _pt_tmp_127 = 0.0 + _pt_tmp_128
    _pt_tmp_135 = 1.0 * _pt_data_23
    del _pt_tmp_128
    _pt_tmp_134 = (
        _pt_tmp_135 * _pt_tmp_114
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_135, _in1=_pt_tmp_114)["out"]
    )
    _pt_tmp_126 = _pt_tmp_127 + _pt_tmp_134
    _pt_tmp_125 = 0.5 * _pt_tmp_126
    del _pt_tmp_127, _pt_tmp_134
    _pt_tmp_120 = _pt_tmp_121 - _pt_tmp_125
    del _pt_tmp_126
    _pt_tmp_118 = (
        _pt_tmp_119 * _pt_tmp_120
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_119, _in1=_pt_tmp_120)["out"]
    )
    del _pt_tmp_121, _pt_tmp_125
    _pt_tmp_92 = _pt_tmp_93 + _pt_tmp_118
    _pt_tmp_91 = -0.5 * _pt_tmp_92
    del _pt_tmp_118, _pt_tmp_93
    _pt_data_24 = actx.thaw(npzfile["_pt_data_24"])
    del _pt_tmp_92
    _pt_tmp_136 = actx.np.reshape(_pt_data_24, (648000, 1))
    _pt_tmp_90 = (
        _pt_tmp_91[_pt_tmp_136, _pt_tmp_82]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_10, in_0=_pt_tmp_91, in_1=_pt_tmp_136, in_2=_pt_tmp_82
        )["out"]
    )
    _pt_tmp_88 = (
        actx.np.where(_pt_tmp_89, _pt_tmp_90, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_6, _in0=_pt_tmp_89, _in1=_pt_tmp_90)["out"]
    )
    del _pt_tmp_91
    _pt_tmp_88 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_88)
    del _pt_tmp_90
    _pt_tmp_88 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_88)
    _pt_tmp_87 = 0.0 + _pt_tmp_88
    _pt_tmp_86 = 0.0 + _pt_tmp_87
    del _pt_tmp_88
    _pt_tmp_137 = actx.np.zeros((648000, 10), dtype=np.float64)
    del _pt_tmp_87
    _pt_tmp_85 = _pt_tmp_86 + _pt_tmp_137
    _pt_tmp_84 = _pt_tmp_85 + _pt_tmp_137
    del _pt_tmp_86
    _pt_tmp_83 = _pt_tmp_84 + _pt_tmp_137
    del _pt_tmp_85
    _pt_tmp_21 = _pt_tmp_22 + _pt_tmp_83
    del _pt_tmp_84
    _pt_tmp_20 = actx.np.reshape(_pt_tmp_21, (4, 162000, 10))
    del _pt_tmp_22, _pt_tmp_83
    _pt_tmp_20 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_20)
    del _pt_tmp_21
    _pt_tmp_17 = actx.einsum("ijk,jl,jlk->li", _pt_data_3, _pt_tmp_18, _pt_tmp_20)
    _pt_tmp_17 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_17)
    del _pt_tmp_20
    _pt_tmp_14 = actx.einsum("i,jk,ik->ij", _pt_tmp_15, _pt_data_2, _pt_tmp_17)
    _pt_tmp_14 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_14)
    del _pt_tmp_17
    _pt_tmp_1 = _pt_tmp_2 - _pt_tmp_14
    _pt_tmp_0 = _pt_tmp_1 / np.float64(1.0)
    del _pt_tmp_14, _pt_tmp_2
    _pt_tmp_147 = _pt_data[0]
    del _pt_tmp_1
    _pt_tmp_146 = _pt_tmp_147[:, :, 0]
    _pt_tmp_145 = actx.einsum(
        "ij,ikl,jl->jk", _pt_tmp_146, _pt_data_0, _actx_in_1_5_0
    )
    del _pt_tmp_147
    _pt_tmp_145 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_145)
    _pt_tmp_144 = -1.0 * _pt_tmp_145
    _pt_tmp_143 = 0.0 + _pt_tmp_144
    del _pt_tmp_145
    _pt_tmp_149 = actx.einsum(
        "ij,ikl,jl->jk", _pt_tmp_12, _pt_data_0, _actx_in_1_3_0
    )
    del _pt_tmp_144
    _pt_tmp_149 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_149)
    _pt_tmp_148 = 1.0 * _pt_tmp_149
    _pt_tmp_142 = _pt_tmp_143 + _pt_tmp_148
    del _pt_tmp_149
    _pt_tmp_141 = 0.0 - _pt_tmp_142
    del _pt_tmp_143, _pt_tmp_148
    _pt_tmp_140 = -1 * _pt_tmp_141
    del _pt_tmp_142
    _pt_tmp_161 = (
        _pt_tmp_73 * _pt_tmp_32
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_73, _in1=_pt_tmp_32)["out"]
    )
    del _pt_tmp_141
    _pt_tmp_160 = 0.0 + _pt_tmp_161
    del _pt_tmp_32
    _pt_tmp_169 = (
        _actx_in_1_3_0[_pt_tmp_39, _pt_tmp_40]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_actx_in_1_3_0, in_1=_pt_tmp_39, in_2=_pt_tmp_40
        )["out"]
    )
    del _pt_tmp_161
    _pt_tmp_169 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_169)
    del _pt_tmp_39, _pt_tmp_40
    _pt_tmp_169 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_169)
    _pt_tmp_168 = 0.0 + _pt_tmp_169
    _pt_tmp_167 = (
        _pt_tmp_168[_pt_tmp_41, _pt_tmp_42]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_168, in_1=_pt_tmp_41, in_2=_pt_tmp_42
        )["out"]
    )
    del _pt_tmp_169
    _pt_tmp_167 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_167)
    del _pt_tmp_41, _pt_tmp_42
    _pt_tmp_167 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_167)
    _pt_tmp_166 = 0.0 + _pt_tmp_167
    _pt_tmp_165 = _pt_tmp_166 - _pt_tmp_168
    del _pt_tmp_167
    _pt_tmp_164 = 1.0 * _pt_tmp_165
    del _pt_tmp_166, _pt_tmp_168
    _pt_tmp_173 = (
        _pt_tmp_31 * _pt_tmp_74
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_31, _in1=_pt_tmp_74)["out"]
    )
    _pt_tmp_172 = 0.0 + _pt_tmp_173
    _pt_tmp_174 = (
        _pt_tmp_61 * _pt_tmp_48
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_61, _in1=_pt_tmp_48)["out"]
    )
    del _pt_tmp_173
    _pt_tmp_171 = _pt_tmp_172 + _pt_tmp_174
    _pt_tmp_170 = 0.5 * _pt_tmp_171
    del _pt_tmp_172, _pt_tmp_174
    _pt_tmp_163 = _pt_tmp_164 - _pt_tmp_170
    del _pt_tmp_171
    _pt_tmp_162 = (
        _pt_tmp_80 * _pt_tmp_163
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_80, _in1=_pt_tmp_163)["out"]
    )
    del _pt_tmp_164, _pt_tmp_170
    _pt_tmp_159 = _pt_tmp_160 + _pt_tmp_162
    _pt_tmp_158 = -0.5 * _pt_tmp_159
    del _pt_tmp_160, _pt_tmp_162
    _pt_tmp_157 = (
        _pt_tmp_158[_pt_tmp_81, _pt_tmp_82]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_5, in_0=_pt_tmp_158, in_1=_pt_tmp_81, in_2=_pt_tmp_82
        )["out"]
    )
    del _pt_tmp_159
    _pt_tmp_156 = (
        actx.np.where(_pt_tmp_25, _pt_tmp_157, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_6, _in0=_pt_tmp_25, _in1=_pt_tmp_157)["out"]
    )
    del _pt_tmp_158
    _pt_tmp_156 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_156)
    del _pt_tmp_157
    _pt_tmp_156 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_156)
    _pt_tmp_155 = 0.0 + _pt_tmp_156
    _pt_tmp_154 = 0.0 + _pt_tmp_155
    del _pt_tmp_156
    _pt_tmp_185 = (
        _pt_tmp_129 * _pt_tmp_96
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_129, _in1=_pt_tmp_96)["out"]
    )
    del _pt_tmp_155
    _pt_tmp_184 = 0.0 + _pt_tmp_185
    del _pt_tmp_96
    _pt_tmp_191 = (
        _actx_in_1_3_0[_pt_tmp_101, _pt_tmp_102]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_8, in_0=_actx_in_1_3_0, in_1=_pt_tmp_101, in_2=_pt_tmp_102
        )["out"]
    )
    del _pt_tmp_185
    _pt_tmp_191 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_191)
    del _pt_tmp_101, _pt_tmp_102
    _pt_tmp_191 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_191)
    _pt_tmp_190 = 0.0 + _pt_tmp_191
    _pt_tmp_189 = _pt_tmp_190 - _pt_tmp_190
    del _pt_tmp_191
    _pt_tmp_188 = 1.0 * _pt_tmp_189
    del _pt_tmp_190
    _pt_tmp_195 = (
        _pt_tmp_95 * _pt_tmp_130
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_95, _in1=_pt_tmp_130)["out"]
    )
    _pt_tmp_194 = 0.0 + _pt_tmp_195
    _pt_tmp_196 = (
        _pt_tmp_119 * _pt_tmp_108
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_119, _in1=_pt_tmp_108)["out"]
    )
    del _pt_tmp_195
    _pt_tmp_193 = _pt_tmp_194 + _pt_tmp_196
    _pt_tmp_192 = 0.5 * _pt_tmp_193
    del _pt_tmp_194, _pt_tmp_196
    _pt_tmp_187 = _pt_tmp_188 - _pt_tmp_192
    del _pt_tmp_193
    _pt_tmp_186 = (
        _pt_tmp_135 * _pt_tmp_187
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_135, _in1=_pt_tmp_187)["out"]
    )
    del _pt_tmp_188, _pt_tmp_192
    _pt_tmp_183 = _pt_tmp_184 + _pt_tmp_186
    _pt_tmp_182 = -0.5 * _pt_tmp_183
    del _pt_tmp_184, _pt_tmp_186
    _pt_tmp_181 = (
        _pt_tmp_182[_pt_tmp_136, _pt_tmp_82]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_10, in_0=_pt_tmp_182, in_1=_pt_tmp_136, in_2=_pt_tmp_82
        )["out"]
    )
    del _pt_tmp_183
    _pt_tmp_180 = (
        actx.np.where(_pt_tmp_89, _pt_tmp_181, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_6, _in0=_pt_tmp_89, _in1=_pt_tmp_181)["out"]
    )
    del _pt_tmp_182
    _pt_tmp_180 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_180)
    del _pt_tmp_181
    _pt_tmp_180 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_180)
    _pt_tmp_179 = 0.0 + _pt_tmp_180
    _pt_tmp_178 = 0.0 + _pt_tmp_179
    del _pt_tmp_180
    _pt_tmp_177 = _pt_tmp_178 + _pt_tmp_137
    del _pt_tmp_179
    _pt_tmp_176 = _pt_tmp_177 + _pt_tmp_137
    del _pt_tmp_178
    _pt_tmp_175 = _pt_tmp_176 + _pt_tmp_137
    del _pt_tmp_177
    _pt_tmp_153 = _pt_tmp_154 + _pt_tmp_175
    del _pt_tmp_176
    _pt_tmp_152 = actx.np.reshape(_pt_tmp_153, (4, 162000, 10))
    del _pt_tmp_154, _pt_tmp_175
    _pt_tmp_152 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_152)
    del _pt_tmp_153
    _pt_tmp_151 = actx.einsum("ijk,jl,jlk->li", _pt_data_3, _pt_tmp_18, _pt_tmp_152)
    _pt_tmp_151 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_151)
    del _pt_tmp_152
    _pt_tmp_150 = actx.einsum("i,jk,ik->ij", _pt_tmp_15, _pt_data_2, _pt_tmp_151)
    _pt_tmp_150 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_150)
    del _pt_tmp_151
    _pt_tmp_139 = _pt_tmp_140 - _pt_tmp_150
    _pt_tmp_138 = _pt_tmp_139 / np.float64(1.0)
    del _pt_tmp_140, _pt_tmp_150
    _pt_tmp_204 = actx.einsum(
        "ij,ikl,jl->jk", _pt_tmp_146, _pt_data_0, _actx_in_1_4_0
    )
    del _pt_tmp_139
    _pt_tmp_204 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_204)
    _pt_tmp_203 = 1.0 * _pt_tmp_204
    _pt_tmp_202 = 0.0 + _pt_tmp_203
    del _pt_tmp_204
    _pt_tmp_206 = actx.einsum("ij,ikl,jl->jk", _pt_tmp_8, _pt_data_0, _actx_in_1_3_0)
    del _pt_tmp_203
    _pt_tmp_206 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_206)
    _pt_tmp_205 = -1.0 * _pt_tmp_206
    _pt_tmp_201 = _pt_tmp_202 + _pt_tmp_205
    del _pt_tmp_206
    _pt_tmp_200 = 0.0 - _pt_tmp_201
    del _pt_tmp_202, _pt_tmp_205
    _pt_tmp_199 = -1 * _pt_tmp_200
    del _pt_tmp_201
    _pt_tmp_218 = (
        _pt_tmp_47 * _pt_tmp_62
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_47, _in1=_pt_tmp_62)["out"]
    )
    del _pt_tmp_200
    _pt_tmp_217 = 0.0 + _pt_tmp_218
    del _pt_tmp_62
    _pt_tmp_219 = (
        _pt_tmp_54 * _pt_tmp_163
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_54, _in1=_pt_tmp_163)["out"]
    )
    del _pt_tmp_218
    _pt_tmp_216 = _pt_tmp_217 + _pt_tmp_219
    del _pt_tmp_163
    _pt_tmp_215 = -0.5 * _pt_tmp_216
    del _pt_tmp_217, _pt_tmp_219
    _pt_tmp_214 = (
        _pt_tmp_215[_pt_tmp_81, _pt_tmp_82]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_5, in_0=_pt_tmp_215, in_1=_pt_tmp_81, in_2=_pt_tmp_82
        )["out"]
    )
    del _pt_tmp_216
    _pt_tmp_213 = (
        actx.np.where(_pt_tmp_25, _pt_tmp_214, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_6, _in0=_pt_tmp_25, _in1=_pt_tmp_214)["out"]
    )
    del _pt_tmp_215
    _pt_tmp_213 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_213)
    del _pt_tmp_214
    _pt_tmp_213 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_213)
    _pt_tmp_212 = 0.0 + _pt_tmp_213
    _pt_tmp_211 = 0.0 + _pt_tmp_212
    del _pt_tmp_213
    _pt_tmp_230 = (
        _pt_tmp_107 * _pt_tmp_120
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_107, _in1=_pt_tmp_120)["out"]
    )
    del _pt_tmp_212
    _pt_tmp_229 = 0.0 + _pt_tmp_230
    del _pt_tmp_120
    _pt_tmp_231 = (
        _pt_tmp_113 * _pt_tmp_187
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_113, _in1=_pt_tmp_187)["out"]
    )
    del _pt_tmp_230
    _pt_tmp_228 = _pt_tmp_229 + _pt_tmp_231
    del _pt_tmp_187
    _pt_tmp_227 = -0.5 * _pt_tmp_228
    del _pt_tmp_229, _pt_tmp_231
    _pt_tmp_226 = (
        _pt_tmp_227[_pt_tmp_136, _pt_tmp_82]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_10, in_0=_pt_tmp_227, in_1=_pt_tmp_136, in_2=_pt_tmp_82
        )["out"]
    )
    del _pt_tmp_228
    _pt_tmp_225 = (
        actx.np.where(_pt_tmp_89, _pt_tmp_226, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_6, _in0=_pt_tmp_89, _in1=_pt_tmp_226)["out"]
    )
    del _pt_tmp_227
    _pt_tmp_225 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_225)
    del _pt_tmp_226
    _pt_tmp_225 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_225)
    _pt_tmp_224 = 0.0 + _pt_tmp_225
    _pt_tmp_223 = 0.0 + _pt_tmp_224
    del _pt_tmp_225
    _pt_tmp_222 = _pt_tmp_223 + _pt_tmp_137
    del _pt_tmp_224
    _pt_tmp_221 = _pt_tmp_222 + _pt_tmp_137
    del _pt_tmp_223
    _pt_tmp_220 = _pt_tmp_221 + _pt_tmp_137
    del _pt_tmp_222
    _pt_tmp_210 = _pt_tmp_211 + _pt_tmp_220
    del _pt_tmp_221
    _pt_tmp_209 = actx.np.reshape(_pt_tmp_210, (4, 162000, 10))
    del _pt_tmp_211, _pt_tmp_220
    _pt_tmp_209 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_209)
    del _pt_tmp_210
    _pt_tmp_208 = actx.einsum("ijk,jl,jlk->li", _pt_data_3, _pt_tmp_18, _pt_tmp_209)
    _pt_tmp_208 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_208)
    del _pt_tmp_209
    _pt_tmp_207 = actx.einsum("i,jk,ik->ij", _pt_tmp_15, _pt_data_2, _pt_tmp_208)
    _pt_tmp_207 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_207)
    del _pt_tmp_208
    _pt_tmp_198 = _pt_tmp_199 - _pt_tmp_207
    _pt_tmp_197 = _pt_tmp_198 / np.float64(1.0)
    del _pt_tmp_199, _pt_tmp_207
    _pt_tmp_238 = actx.einsum("ij,ikl,jl->jk", _pt_tmp_8, _pt_data_0, _actx_in_1_2_0)
    del _pt_tmp_198
    _pt_tmp_238 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_238)
    _pt_tmp_237 = 1.0 * _pt_tmp_238
    _pt_tmp_236 = 0.0 + _pt_tmp_237
    del _pt_tmp_238
    _pt_tmp_240 = actx.einsum(
        "ij,ikl,jl->jk", _pt_tmp_12, _pt_data_0, _actx_in_1_1_0
    )
    del _pt_tmp_237
    _pt_tmp_240 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_240)
    _pt_tmp_239 = -1.0 * _pt_tmp_240
    _pt_tmp_235 = _pt_tmp_236 + _pt_tmp_239
    del _pt_tmp_240
    _pt_tmp_234 = -1 * _pt_tmp_235
    del _pt_tmp_236, _pt_tmp_239
    _pt_tmp_254 = 1.0 * _pt_tmp_74
    del _pt_tmp_235
    _pt_tmp_258 = (
        _pt_tmp_47 * _pt_tmp_64
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_47, _in1=_pt_tmp_64)["out"]
    )
    del _pt_tmp_74
    _pt_tmp_257 = 0.0 + _pt_tmp_258
    _pt_tmp_259 = (
        _pt_tmp_54 * _pt_tmp_165
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_54, _in1=_pt_tmp_165)["out"]
    )
    del _pt_tmp_258
    _pt_tmp_256 = _pt_tmp_257 + _pt_tmp_259
    _pt_tmp_255 = 0.5 * _pt_tmp_256
    del _pt_tmp_257, _pt_tmp_259
    _pt_tmp_253 = _pt_tmp_254 + _pt_tmp_255
    del _pt_tmp_256
    _pt_tmp_252 = (
        _pt_tmp_31 * _pt_tmp_253
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_31, _in1=_pt_tmp_253)["out"]
    )
    del _pt_tmp_254, _pt_tmp_255
    _pt_tmp_251 = 0.0 + _pt_tmp_252
    _pt_tmp_262 = 1.0 * _pt_tmp_48
    del _pt_tmp_252
    _pt_tmp_266 = (
        _pt_tmp_73 * _pt_tmp_34
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_73, _in1=_pt_tmp_34)["out"]
    )
    del _pt_tmp_48
    _pt_tmp_265 = 0.0 + _pt_tmp_266
    _pt_tmp_267 = (
        _pt_tmp_80 * _pt_tmp_165
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_80, _in1=_pt_tmp_165)["out"]
    )
    del _pt_tmp_266
    _pt_tmp_264 = _pt_tmp_265 + _pt_tmp_267
    del _pt_tmp_165
    _pt_tmp_263 = 0.5 * _pt_tmp_264
    del _pt_tmp_265, _pt_tmp_267
    _pt_tmp_261 = _pt_tmp_262 + _pt_tmp_263
    del _pt_tmp_264
    _pt_tmp_260 = (
        _pt_tmp_61 * _pt_tmp_261
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_61, _in1=_pt_tmp_261)["out"]
    )
    del _pt_tmp_262, _pt_tmp_263
    _pt_tmp_250 = _pt_tmp_251 + _pt_tmp_260
    _pt_tmp_249 = 0.5 * _pt_tmp_250
    del _pt_tmp_251, _pt_tmp_260
    _pt_tmp_248 = (
        _pt_tmp_249[_pt_tmp_81, _pt_tmp_82]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_5, in_0=_pt_tmp_249, in_1=_pt_tmp_81, in_2=_pt_tmp_82
        )["out"]
    )
    del _pt_tmp_250
    _pt_tmp_247 = (
        actx.np.where(_pt_tmp_25, _pt_tmp_248, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_6, _in0=_pt_tmp_25, _in1=_pt_tmp_248)["out"]
    )
    del _pt_tmp_249
    _pt_tmp_247 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_247)
    del _pt_tmp_248
    _pt_tmp_247 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_247)
    _pt_tmp_246 = 0.0 + _pt_tmp_247
    _pt_tmp_245 = 0.0 + _pt_tmp_246
    del _pt_tmp_247
    _pt_tmp_280 = 1.0 * _pt_tmp_130
    del _pt_tmp_246
    _pt_tmp_284 = (
        _pt_tmp_107 * _pt_tmp_122
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_107, _in1=_pt_tmp_122)["out"]
    )
    del _pt_tmp_130
    _pt_tmp_283 = 0.0 + _pt_tmp_284
    _pt_tmp_285 = (
        _pt_tmp_113 * _pt_tmp_189
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_113, _in1=_pt_tmp_189)["out"]
    )
    del _pt_tmp_284
    _pt_tmp_282 = _pt_tmp_283 + _pt_tmp_285
    _pt_tmp_281 = 0.5 * _pt_tmp_282
    del _pt_tmp_283, _pt_tmp_285
    _pt_tmp_279 = _pt_tmp_280 + _pt_tmp_281
    del _pt_tmp_282
    _pt_tmp_278 = (
        _pt_tmp_95 * _pt_tmp_279
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_95, _in1=_pt_tmp_279)["out"]
    )
    del _pt_tmp_280, _pt_tmp_281
    _pt_tmp_277 = 0.0 + _pt_tmp_278
    _pt_tmp_288 = 1.0 * _pt_tmp_108
    del _pt_tmp_278
    _pt_tmp_292 = (
        _pt_tmp_129 * _pt_tmp_98
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_129, _in1=_pt_tmp_98)["out"]
    )
    del _pt_tmp_108
    _pt_tmp_291 = 0.0 + _pt_tmp_292
    _pt_tmp_293 = (
        _pt_tmp_135 * _pt_tmp_189
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_135, _in1=_pt_tmp_189)["out"]
    )
    del _pt_tmp_292
    _pt_tmp_290 = _pt_tmp_291 + _pt_tmp_293
    del _pt_tmp_189
    _pt_tmp_289 = 0.5 * _pt_tmp_290
    del _pt_tmp_291, _pt_tmp_293
    _pt_tmp_287 = _pt_tmp_288 + _pt_tmp_289
    del _pt_tmp_290
    _pt_tmp_286 = (
        _pt_tmp_119 * _pt_tmp_287
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_119, _in1=_pt_tmp_287)["out"]
    )
    del _pt_tmp_288, _pt_tmp_289
    _pt_tmp_276 = _pt_tmp_277 + _pt_tmp_286
    _pt_tmp_275 = 0.5 * _pt_tmp_276
    del _pt_tmp_277, _pt_tmp_286
    _pt_tmp_274 = (
        _pt_tmp_275[_pt_tmp_136, _pt_tmp_82]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_10, in_0=_pt_tmp_275, in_1=_pt_tmp_136, in_2=_pt_tmp_82
        )["out"]
    )
    del _pt_tmp_276
    _pt_tmp_273 = (
        actx.np.where(_pt_tmp_89, _pt_tmp_274, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_6, _in0=_pt_tmp_89, _in1=_pt_tmp_274)["out"]
    )
    del _pt_tmp_275
    _pt_tmp_273 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_273)
    del _pt_tmp_274
    _pt_tmp_273 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_273)
    _pt_tmp_272 = 0.0 + _pt_tmp_273
    _pt_tmp_271 = 0.0 + _pt_tmp_272
    del _pt_tmp_273
    _pt_tmp_270 = _pt_tmp_271 + _pt_tmp_137
    del _pt_tmp_272
    _pt_tmp_269 = _pt_tmp_270 + _pt_tmp_137
    del _pt_tmp_271
    _pt_tmp_268 = _pt_tmp_269 + _pt_tmp_137
    del _pt_tmp_270
    _pt_tmp_244 = _pt_tmp_245 + _pt_tmp_268
    del _pt_tmp_269
    _pt_tmp_243 = actx.np.reshape(_pt_tmp_244, (4, 162000, 10))
    del _pt_tmp_245, _pt_tmp_268
    _pt_tmp_243 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_243)
    del _pt_tmp_244
    _pt_tmp_242 = actx.einsum("ijk,jl,jlk->li", _pt_data_3, _pt_tmp_18, _pt_tmp_243)
    _pt_tmp_242 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_242)
    del _pt_tmp_243
    _pt_tmp_241 = actx.einsum("i,jk,ik->ij", _pt_tmp_15, _pt_data_2, _pt_tmp_242)
    _pt_tmp_241 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_241)
    del _pt_tmp_242
    _pt_tmp_233 = _pt_tmp_234 - _pt_tmp_241
    _pt_tmp_232 = _pt_tmp_233 / np.float64(1.0)
    del _pt_tmp_234, _pt_tmp_241
    _pt_tmp_300 = actx.einsum(
        "ij,ikl,jl->jk", _pt_tmp_146, _pt_data_0, _actx_in_1_2_0
    )
    del _pt_tmp_233
    _pt_tmp_300 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_300)
    _pt_tmp_299 = -1.0 * _pt_tmp_300
    _pt_tmp_298 = 0.0 + _pt_tmp_299
    del _pt_tmp_300
    _pt_tmp_302 = actx.einsum(
        "ij,ikl,jl->jk", _pt_tmp_12, _pt_data_0, _actx_in_1_0_0
    )
    del _pt_tmp_299
    _pt_tmp_302 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_302)
    del _pt_tmp_12
    _pt_tmp_301 = 1.0 * _pt_tmp_302
    _pt_tmp_297 = _pt_tmp_298 + _pt_tmp_301
    del _pt_tmp_302
    _pt_tmp_296 = -1 * _pt_tmp_297
    del _pt_tmp_298, _pt_tmp_301
    _pt_tmp_314 = (
        _pt_tmp_73 * _pt_tmp_253
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_73, _in1=_pt_tmp_253)["out"]
    )
    del _pt_tmp_297
    _pt_tmp_313 = 0.0 + _pt_tmp_314
    del _pt_tmp_253, _pt_tmp_73
    _pt_tmp_317 = 1.0 * _pt_tmp_55
    del _pt_tmp_314
    _pt_tmp_321 = (
        _pt_tmp_31 * _pt_tmp_34
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_31, _in1=_pt_tmp_34)["out"]
    )
    del _pt_tmp_55
    _pt_tmp_320 = 0.0 + _pt_tmp_321
    del _pt_tmp_31, _pt_tmp_34
    _pt_tmp_322 = (
        _pt_tmp_61 * _pt_tmp_64
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_61, _in1=_pt_tmp_64)["out"]
    )
    del _pt_tmp_321
    _pt_tmp_319 = _pt_tmp_320 + _pt_tmp_322
    del _pt_tmp_61, _pt_tmp_64
    _pt_tmp_318 = 0.5 * _pt_tmp_319
    del _pt_tmp_320, _pt_tmp_322
    _pt_tmp_316 = _pt_tmp_317 + _pt_tmp_318
    del _pt_tmp_319
    _pt_tmp_315 = (
        _pt_tmp_80 * _pt_tmp_316
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_80, _in1=_pt_tmp_316)["out"]
    )
    del _pt_tmp_317, _pt_tmp_318
    _pt_tmp_312 = _pt_tmp_313 + _pt_tmp_315
    del _pt_tmp_80
    _pt_tmp_311 = 0.5 * _pt_tmp_312
    del _pt_tmp_313, _pt_tmp_315
    _pt_tmp_310 = (
        _pt_tmp_311[_pt_tmp_81, _pt_tmp_82]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_5, in_0=_pt_tmp_311, in_1=_pt_tmp_81, in_2=_pt_tmp_82
        )["out"]
    )
    del _pt_tmp_312
    _pt_tmp_309 = (
        actx.np.where(_pt_tmp_25, _pt_tmp_310, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_6, _in0=_pt_tmp_25, _in1=_pt_tmp_310)["out"]
    )
    del _pt_tmp_311
    _pt_tmp_309 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_309)
    del _pt_tmp_310
    _pt_tmp_309 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_309)
    _pt_tmp_308 = 0.0 + _pt_tmp_309
    _pt_tmp_307 = 0.0 + _pt_tmp_308
    del _pt_tmp_309
    _pt_tmp_333 = (
        _pt_tmp_129 * _pt_tmp_279
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_129, _in1=_pt_tmp_279)["out"]
    )
    del _pt_tmp_308
    _pt_tmp_332 = 0.0 + _pt_tmp_333
    del _pt_tmp_129, _pt_tmp_279
    _pt_tmp_336 = 1.0 * _pt_tmp_114
    del _pt_tmp_333
    _pt_tmp_340 = (
        _pt_tmp_95 * _pt_tmp_98
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_95, _in1=_pt_tmp_98)["out"]
    )
    del _pt_tmp_114
    _pt_tmp_339 = 0.0 + _pt_tmp_340
    del _pt_tmp_95, _pt_tmp_98
    _pt_tmp_341 = (
        _pt_tmp_119 * _pt_tmp_122
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_119, _in1=_pt_tmp_122)["out"]
    )
    del _pt_tmp_340
    _pt_tmp_338 = _pt_tmp_339 + _pt_tmp_341
    del _pt_tmp_119, _pt_tmp_122
    _pt_tmp_337 = 0.5 * _pt_tmp_338
    del _pt_tmp_339, _pt_tmp_341
    _pt_tmp_335 = _pt_tmp_336 + _pt_tmp_337
    del _pt_tmp_338
    _pt_tmp_334 = (
        _pt_tmp_135 * _pt_tmp_335
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_135, _in1=_pt_tmp_335)["out"]
    )
    del _pt_tmp_336, _pt_tmp_337
    _pt_tmp_331 = _pt_tmp_332 + _pt_tmp_334
    del _pt_tmp_135
    _pt_tmp_330 = 0.5 * _pt_tmp_331
    del _pt_tmp_332, _pt_tmp_334
    _pt_tmp_329 = (
        _pt_tmp_330[_pt_tmp_136, _pt_tmp_82]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_10, in_0=_pt_tmp_330, in_1=_pt_tmp_136, in_2=_pt_tmp_82
        )["out"]
    )
    del _pt_tmp_331
    _pt_tmp_328 = (
        actx.np.where(_pt_tmp_89, _pt_tmp_329, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_6, _in0=_pt_tmp_89, _in1=_pt_tmp_329)["out"]
    )
    del _pt_tmp_330
    _pt_tmp_328 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_328)
    del _pt_tmp_329
    _pt_tmp_328 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_328)
    _pt_tmp_327 = 0.0 + _pt_tmp_328
    _pt_tmp_326 = 0.0 + _pt_tmp_327
    del _pt_tmp_328
    _pt_tmp_325 = _pt_tmp_326 + _pt_tmp_137
    del _pt_tmp_327
    _pt_tmp_324 = _pt_tmp_325 + _pt_tmp_137
    del _pt_tmp_326
    _pt_tmp_323 = _pt_tmp_324 + _pt_tmp_137
    del _pt_tmp_325
    _pt_tmp_306 = _pt_tmp_307 + _pt_tmp_323
    del _pt_tmp_324
    _pt_tmp_305 = actx.np.reshape(_pt_tmp_306, (4, 162000, 10))
    del _pt_tmp_307, _pt_tmp_323
    _pt_tmp_305 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_305)
    del _pt_tmp_306
    _pt_tmp_304 = actx.einsum("ijk,jl,jlk->li", _pt_data_3, _pt_tmp_18, _pt_tmp_305)
    _pt_tmp_304 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_304)
    del _pt_tmp_305
    _pt_tmp_303 = actx.einsum("i,jk,ik->ij", _pt_tmp_15, _pt_data_2, _pt_tmp_304)
    _pt_tmp_303 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_303)
    del _pt_tmp_304
    _pt_tmp_295 = _pt_tmp_296 - _pt_tmp_303
    _pt_tmp_294 = _pt_tmp_295 / np.float64(1.0)
    del _pt_tmp_296, _pt_tmp_303
    _pt_tmp_348 = actx.einsum(
        "ij,ikl,jl->jk", _pt_tmp_146, _pt_data_0, _actx_in_1_1_0
    )
    del _pt_tmp_295
    _pt_tmp_348 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_348)
    del _pt_tmp_146
    _pt_tmp_347 = 1.0 * _pt_tmp_348
    _pt_tmp_346 = 0.0 + _pt_tmp_347
    del _pt_tmp_348
    _pt_tmp_350 = actx.einsum("ij,ikl,jl->jk", _pt_tmp_8, _pt_data_0, _actx_in_1_0_0)
    del _pt_tmp_347
    _pt_tmp_350 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_350)
    del _pt_tmp_8
    _pt_tmp_349 = -1.0 * _pt_tmp_350
    _pt_tmp_345 = _pt_tmp_346 + _pt_tmp_349
    del _pt_tmp_350
    _pt_tmp_344 = -1 * _pt_tmp_345
    del _pt_tmp_346, _pt_tmp_349
    _pt_tmp_362 = (
        _pt_tmp_47 * _pt_tmp_261
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_47, _in1=_pt_tmp_261)["out"]
    )
    del _pt_tmp_345
    _pt_tmp_361 = 0.0 + _pt_tmp_362
    del _pt_tmp_261, _pt_tmp_47
    _pt_tmp_363 = (
        _pt_tmp_54 * _pt_tmp_316
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_54, _in1=_pt_tmp_316)["out"]
    )
    del _pt_tmp_362
    _pt_tmp_360 = _pt_tmp_361 + _pt_tmp_363
    del _pt_tmp_316, _pt_tmp_54
    _pt_tmp_359 = 0.5 * _pt_tmp_360
    del _pt_tmp_361, _pt_tmp_363
    _pt_tmp_358 = (
        _pt_tmp_359[_pt_tmp_81, _pt_tmp_82]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_5, in_0=_pt_tmp_359, in_1=_pt_tmp_81, in_2=_pt_tmp_82
        )["out"]
    )
    del _pt_tmp_360
    _pt_tmp_357 = (
        actx.np.where(_pt_tmp_25, _pt_tmp_358, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_6, _in0=_pt_tmp_25, _in1=_pt_tmp_358)["out"]
    )
    del _pt_tmp_359, _pt_tmp_81
    _pt_tmp_357 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_357)
    del _pt_tmp_25, _pt_tmp_358
    _pt_tmp_357 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_357)
    _pt_tmp_356 = 0.0 + _pt_tmp_357
    _pt_tmp_355 = 0.0 + _pt_tmp_356
    del _pt_tmp_357
    _pt_tmp_374 = (
        _pt_tmp_107 * _pt_tmp_287
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_107, _in1=_pt_tmp_287)["out"]
    )
    del _pt_tmp_356
    _pt_tmp_373 = 0.0 + _pt_tmp_374
    del _pt_tmp_107, _pt_tmp_287
    _pt_tmp_375 = (
        _pt_tmp_113 * _pt_tmp_335
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_113, _in1=_pt_tmp_335)["out"]
    )
    del _pt_tmp_374
    _pt_tmp_372 = _pt_tmp_373 + _pt_tmp_375
    del _pt_tmp_113, _pt_tmp_335
    _pt_tmp_371 = 0.5 * _pt_tmp_372
    del _pt_tmp_373, _pt_tmp_375
    _pt_tmp_370 = (
        _pt_tmp_371[_pt_tmp_136, _pt_tmp_82]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_10, in_0=_pt_tmp_371, in_1=_pt_tmp_136, in_2=_pt_tmp_82
        )["out"]
    )
    del _pt_tmp_372
    _pt_tmp_369 = (
        actx.np.where(_pt_tmp_89, _pt_tmp_370, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_6, _in0=_pt_tmp_89, _in1=_pt_tmp_370)["out"]
    )
    del _pt_tmp_136, _pt_tmp_371, _pt_tmp_82
    _pt_tmp_369 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_369)
    del _pt_tmp_370, _pt_tmp_89
    _pt_tmp_369 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_369)
    _pt_tmp_368 = 0.0 + _pt_tmp_369
    _pt_tmp_367 = 0.0 + _pt_tmp_368
    del _pt_tmp_369
    _pt_tmp_366 = _pt_tmp_367 + _pt_tmp_137
    del _pt_tmp_368
    _pt_tmp_365 = _pt_tmp_366 + _pt_tmp_137
    del _pt_tmp_367
    _pt_tmp_364 = _pt_tmp_365 + _pt_tmp_137
    del _pt_tmp_366
    _pt_tmp_354 = _pt_tmp_355 + _pt_tmp_364
    del _pt_tmp_137, _pt_tmp_365
    _pt_tmp_353 = actx.np.reshape(_pt_tmp_354, (4, 162000, 10))
    del _pt_tmp_355, _pt_tmp_364
    _pt_tmp_353 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_353)
    del _pt_tmp_354
    _pt_tmp_352 = actx.einsum("ijk,jl,jlk->li", _pt_data_3, _pt_tmp_18, _pt_tmp_353)
    _pt_tmp_352 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_352)
    del _pt_tmp_18, _pt_tmp_353
    _pt_tmp_351 = actx.einsum("i,jk,ik->ij", _pt_tmp_15, _pt_data_2, _pt_tmp_352)
    _pt_tmp_351 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_351)
    del _pt_tmp_15, _pt_tmp_352
    _pt_tmp_343 = _pt_tmp_344 - _pt_tmp_351
    _pt_tmp_342 = _pt_tmp_343 / np.float64(1.0)
    del _pt_tmp_344, _pt_tmp_351
    _pt_tmp = make_obj_array(
        [_pt_tmp_0, _pt_tmp_138, _pt_tmp_197, _pt_tmp_232, _pt_tmp_294, _pt_tmp_342]
    )
    del _pt_tmp_343
    return _pt_tmp
    del _pt_tmp_0, _pt_tmp_138, _pt_tmp_197, _pt_tmp_232, _pt_tmp_294, _pt_tmp_342


@dataclass(frozen=True)
class RHSInvoker:
    actx: ArrayContext

    @cached_property
    def npzfile(self):
        import os

        kw_to_ary = np.load(
            os.path.join(
                get_actx_dgfem_suite_path(), "suite/maxwell_3D_P3/literals.npz"
            )
        )
        return constantdict(
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
        import pytato as pt
        from pickle import load
        from meshmode.dof_array import array_context_for_pickling

        fpath = os.path.join(
            get_actx_dgfem_suite_path(), "suite/maxwell_3D_P3/ref_outputs.pkl"
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
        from arraycontext.impl.pytato.utils import _ary_container_key_stringifier

        output_keys = set()
        output_template = self._get_output_template()

        def _as_dict_of_named_arrays(keys, ary):
            output_keys.add(keys)
            return ary

        rec_keyed_map_array_container(_as_dict_of_named_arrays, output_template)

        return constantdict(
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
            "_actx_in_1_0_0",
            "_actx_in_1_1_0",
            "_actx_in_1_2_0",
            "_actx_in_1_3_0",
            "_actx_in_1_4_0",
            "_actx_in_1_5_0",
        }

    def __call__(self, *args, **kwargs):
        from arraycontext.impl.pytato.compile import (
            _get_arg_id_to_arg_and_arg_id_to_descr,
        )
        from arraycontext.impl.pytato.utils import _ary_container_key_stringifier

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
