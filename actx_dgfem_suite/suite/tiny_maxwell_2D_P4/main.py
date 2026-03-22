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
        "{ [_0, _1] : 0 <= _0 <= 191 and 0 <= _1 <= 4 }",
        "out[_0, _1] = in_0[in_1[_0] % 3, _1]",
        [
            lp.GlobalArg("out", dtype=np.int64, shape=(192, 5)),
            lp.GlobalArg("in_0", dtype=np.int64, shape=(3, 5)),
            lp.GlobalArg("in_1", dtype=np.int8, shape=(192,)),
        ],
    )
    _pt_t_unit_0 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 191 and 0 <= _1 <= 4 }",
        "out[_0, _1] = in_0[in_1[_0, 0] % 72, in_2[_0, _1] % 15]",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(192, 5)),
            lp.GlobalArg("in_0", dtype=np.float64, shape=(72, 15)),
            lp.GlobalArg("in_1", dtype=np.int32, shape=(192, 1)),
            lp.GlobalArg("in_2", dtype=np.int64, shape=(192, 5)),
        ],
    )
    _pt_t_unit_1 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 191 and 0 <= _1 <= 4 }",
        "out[_0, _1] = in_0[in_1[_0] % 1, _1]",
        [
            lp.GlobalArg("out", dtype=np.int64, shape=(192, 5)),
            lp.GlobalArg("in_0", dtype=np.int64, shape=(1, 5)),
            lp.GlobalArg("in_1", dtype=np.int8, shape=(192,)),
        ],
    )
    _pt_t_unit_10 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 215 and 0 <= _1 <= 4 }",
        "out[_0, _1] = in_0[in_1[_0, 0] % 24, in_2[_0, _1] % 5]",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(216, 5)),
            lp.GlobalArg("in_0", dtype=np.float64, shape=(24, 5)),
            lp.GlobalArg("in_1", dtype=np.int32, shape=(216, 1)),
            lp.GlobalArg("in_2", dtype=np.int64, shape=(216, 5)),
        ],
    )
    _pt_t_unit_2 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 191 and 0 <= _1 <= 4 }",
        "out[_0, _1] = in_0[in_1[_0, 0] % 192, in_2[_0, _1] % 5]",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(192, 5)),
            lp.GlobalArg("in_0", dtype=np.float64, shape=(192, 5)),
            lp.GlobalArg("in_1", dtype=np.int32, shape=(192, 1)),
            lp.GlobalArg("in_2", dtype=np.int64, shape=(192, 5)),
        ],
    )
    _pt_t_unit_3 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 191 and 0 <= _1 <= 4 }",
        "out[_0, _1] = _in0[_0, 0]*_in1[_0, _1]",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(192, 5)),
            lp.GlobalArg("_in0", dtype=np.float64, shape=(192, 1)),
            lp.GlobalArg("_in1", dtype=np.float64, shape=(192, 5)),
        ],
    )
    _pt_t_unit_4 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 215 and 0 <= _1 <= 4 }",
        "out[_0, _1] = in_0[in_1[_0] % 1, _1]",
        [
            lp.GlobalArg("out", dtype=np.int64, shape=(216, 5)),
            lp.GlobalArg("in_0", dtype=np.int64, shape=(1, 5)),
            lp.GlobalArg("in_1", dtype=np.int8, shape=(216,)),
        ],
    )
    _pt_t_unit_5 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 215 and 0 <= _1 <= 4 }",
        "out[_0, _1] = in_0[in_1[_0, 0] % 192, in_2[_0, _1] % 5]",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(216, 5)),
            lp.GlobalArg("in_0", dtype=np.float64, shape=(192, 5)),
            lp.GlobalArg("in_1", dtype=np.int32, shape=(216, 1)),
            lp.GlobalArg("in_2", dtype=np.int64, shape=(216, 5)),
        ],
    )
    _pt_t_unit_6 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 215 and 0 <= _1 <= 4 }",
        "out[_0, _1] = _in1[_0, _1] if _in0[_0, 0] else 0",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(216, 5)),
            lp.GlobalArg("_in0", dtype=np.int8, shape=(216, 1)),
            lp.GlobalArg("_in1", dtype=np.float64, shape=(216, 5)),
        ],
    )
    _pt_t_unit_7 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 23 and 0 <= _1 <= 4 }",
        "out[_0, _1] = in_0[in_1[_0] % 3, _1]",
        [
            lp.GlobalArg("out", dtype=np.int64, shape=(24, 5)),
            lp.GlobalArg("in_0", dtype=np.int64, shape=(3, 5)),
            lp.GlobalArg("in_1", dtype=np.int8, shape=(24,)),
        ],
    )
    _pt_t_unit_8 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 23 and 0 <= _1 <= 4 }",
        "out[_0, _1] = in_0[in_1[_0, 0] % 72, in_2[_0, _1] % 15]",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(24, 5)),
            lp.GlobalArg("in_0", dtype=np.float64, shape=(72, 15)),
            lp.GlobalArg("in_1", dtype=np.int32, shape=(24, 1)),
            lp.GlobalArg("in_2", dtype=np.int64, shape=(24, 5)),
        ],
    )
    _pt_t_unit_9 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 23 and 0 <= _1 <= 4 }",
        "out[_0, _1] = _in0[_0, 0]*_in1[_0, _1]",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(24, 5)),
            lp.GlobalArg("_in0", dtype=np.float64, shape=(24, 1)),
            lp.GlobalArg("_in1", dtype=np.float64, shape=(24, 5)),
        ],
    )
    _pt_data = actx.thaw(npzfile["_pt_data"])
    _pt_tmp_8 = _pt_data[1]
    _pt_tmp_7 = _pt_tmp_8[:, :, 0]
    _pt_data_0 = actx.thaw(npzfile["_pt_data_0"])
    del _pt_tmp_8
    _pt_tmp_6 = actx.einsum("ij,ikl,jl->jk", _pt_tmp_7, _pt_data_0, _actx_in_1_5_0)
    _pt_tmp_6 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_6)
    _pt_tmp_5 = 1.0 * _pt_tmp_6
    _pt_tmp_4 = 0.0 + _pt_tmp_5
    del _pt_tmp_6
    _pt_tmp_3 = 0.0 - _pt_tmp_4
    del _pt_tmp_5
    _pt_tmp_2 = -1 * _pt_tmp_3
    del _pt_tmp_4
    _pt_data_1 = actx.thaw(npzfile["_pt_data_1"])
    del _pt_tmp_3
    _pt_tmp_11 = 1.0 / _pt_data_1
    _pt_tmp_10 = _pt_tmp_11[:, 0]
    _pt_data_2 = actx.thaw(npzfile["_pt_data_2"])
    del _pt_tmp_11
    _pt_data_3 = actx.thaw(npzfile["_pt_data_3"])
    _pt_data_4 = actx.thaw(npzfile["_pt_data_4"])
    _pt_tmp_14 = actx.np.reshape(_pt_data_4, (3, 72, 1))
    _pt_tmp_14 = actx.tag_axis(1, (DiscretizationElementAxisTag(),), _pt_tmp_14)
    _pt_tmp_13 = _pt_tmp_14[:, :, 0]
    _pt_data_5 = actx.thaw(npzfile["_pt_data_5"])
    del _pt_tmp_14
    _pt_tmp_20 = actx.np.reshape(_pt_data_5, (216, 1))
    _pt_data_6 = actx.thaw(npzfile["_pt_data_6"])
    _pt_tmp_25 = 1.0 * _pt_data_6
    _pt_data_7 = actx.thaw(npzfile["_pt_data_7"])
    _pt_tmp_33 = actx.np.reshape(_pt_data_7, (192, 1))
    _pt_data_8 = actx.thaw(npzfile["_pt_data_8"])
    _pt_data_9 = actx.thaw(npzfile["_pt_data_9"])
    _pt_tmp_34 = (
        _pt_data_8[_pt_data_9]
        if actx.permits_advanced_indexing
        else actx.call_loopy(_pt_t_unit, in_0=_pt_data_8, in_1=_pt_data_9)["out"]
    )
    _pt_tmp_32 = (
        _actx_in_1_5_0[_pt_tmp_33, _pt_tmp_34]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_actx_in_1_5_0, in_1=_pt_tmp_33, in_2=_pt_tmp_34
        )["out"]
    )
    _pt_tmp_32 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_32)
    _pt_tmp_32 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_32)
    _pt_tmp_31 = 0.0 + _pt_tmp_32
    _pt_data_10 = actx.thaw(npzfile["_pt_data_10"])
    del _pt_tmp_32
    _pt_tmp_35 = actx.np.reshape(_pt_data_10, (192, 1))
    _pt_data_11 = actx.thaw(npzfile["_pt_data_11"])
    _pt_data_12 = actx.thaw(npzfile["_pt_data_12"])
    _pt_tmp_36 = (
        _pt_data_11[_pt_data_12]
        if actx.permits_advanced_indexing
        else actx.call_loopy(_pt_t_unit_1, in_0=_pt_data_11, in_1=_pt_data_12)["out"]
    )
    _pt_tmp_30 = (
        _pt_tmp_31[_pt_tmp_35, _pt_tmp_36]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_31, in_1=_pt_tmp_35, in_2=_pt_tmp_36
        )["out"]
    )
    _pt_tmp_30 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_30)
    _pt_tmp_30 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_30)
    _pt_tmp_29 = 0.0 + _pt_tmp_30
    _pt_tmp_28 = _pt_tmp_29 - _pt_tmp_31
    del _pt_tmp_30
    _pt_tmp_27 = 1.0 * _pt_tmp_28
    del _pt_tmp_29, _pt_tmp_31
    _pt_data_13 = actx.thaw(npzfile["_pt_data_13"])
    _pt_tmp_41 = 1.0 * _pt_data_13
    _pt_tmp_46 = (
        _actx_in_1_1_0[_pt_tmp_33, _pt_tmp_34]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_actx_in_1_1_0, in_1=_pt_tmp_33, in_2=_pt_tmp_34
        )["out"]
    )
    _pt_tmp_46 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_46)
    _pt_tmp_46 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_46)
    _pt_tmp_45 = 0.0 + _pt_tmp_46
    _pt_tmp_44 = (
        _pt_tmp_45[_pt_tmp_35, _pt_tmp_36]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_45, in_1=_pt_tmp_35, in_2=_pt_tmp_36
        )["out"]
    )
    del _pt_tmp_46
    _pt_tmp_44 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_44)
    _pt_tmp_44 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_44)
    _pt_tmp_43 = 0.0 + _pt_tmp_44
    _pt_tmp_42 = _pt_tmp_43 - _pt_tmp_45
    del _pt_tmp_44
    _pt_tmp_40 = (
        _pt_tmp_41 * _pt_tmp_42
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_41, _in1=_pt_tmp_42)["out"]
    )
    del _pt_tmp_43, _pt_tmp_45
    _pt_tmp_39 = 0.0 + _pt_tmp_40
    _pt_tmp_48 = -1.0 * _pt_data_6
    del _pt_tmp_40
    _pt_tmp_53 = (
        _actx_in_1_0_0[_pt_tmp_33, _pt_tmp_34]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_actx_in_1_0_0, in_1=_pt_tmp_33, in_2=_pt_tmp_34
        )["out"]
    )
    _pt_tmp_53 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_53)
    _pt_tmp_53 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_53)
    _pt_tmp_52 = 0.0 + _pt_tmp_53
    _pt_tmp_51 = (
        _pt_tmp_52[_pt_tmp_35, _pt_tmp_36]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_52, in_1=_pt_tmp_35, in_2=_pt_tmp_36
        )["out"]
    )
    del _pt_tmp_53
    _pt_tmp_51 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_51)
    _pt_tmp_51 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_51)
    _pt_tmp_50 = 0.0 + _pt_tmp_51
    _pt_tmp_49 = _pt_tmp_50 - _pt_tmp_52
    del _pt_tmp_51
    _pt_tmp_47 = (
        _pt_tmp_48 * _pt_tmp_49
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_48, _in1=_pt_tmp_49)["out"]
    )
    del _pt_tmp_50, _pt_tmp_52
    _pt_tmp_38 = _pt_tmp_39 + _pt_tmp_47
    _pt_tmp_37 = 0.5 * _pt_tmp_38
    del _pt_tmp_39, _pt_tmp_47
    _pt_tmp_26 = _pt_tmp_27 - _pt_tmp_37
    del _pt_tmp_38
    _pt_tmp_24 = (
        _pt_tmp_25 * _pt_tmp_26
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_25, _in1=_pt_tmp_26)["out"]
    )
    del _pt_tmp_27, _pt_tmp_37
    _pt_tmp_23 = 0.0 + _pt_tmp_24
    _pt_tmp_22 = -0.5 * _pt_tmp_23
    del _pt_tmp_24
    _pt_data_14 = actx.thaw(npzfile["_pt_data_14"])
    del _pt_tmp_23
    _pt_tmp_54 = actx.np.reshape(_pt_data_14, (216, 1))
    _pt_data_15 = actx.thaw(npzfile["_pt_data_15"])
    _pt_data_16 = actx.thaw(npzfile["_pt_data_16"])
    _pt_tmp_55 = (
        _pt_data_15[_pt_data_16]
        if actx.permits_advanced_indexing
        else actx.call_loopy(_pt_t_unit_4, in_0=_pt_data_15, in_1=_pt_data_16)["out"]
    )
    _pt_tmp_21 = (
        _pt_tmp_22[_pt_tmp_54, _pt_tmp_55]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_5, in_0=_pt_tmp_22, in_1=_pt_tmp_54, in_2=_pt_tmp_55
        )["out"]
    )
    _pt_tmp_19 = (
        actx.np.where(_pt_tmp_20, _pt_tmp_21, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_6, _in0=_pt_tmp_20, _in1=_pt_tmp_21)["out"]
    )
    del _pt_tmp_22
    _pt_tmp_19 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_19)
    del _pt_tmp_21
    _pt_tmp_19 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_19)
    _pt_tmp_18 = 0.0 + _pt_tmp_19
    _pt_tmp_17 = 0.0 + _pt_tmp_18
    del _pt_tmp_19
    _pt_data_17 = actx.thaw(npzfile["_pt_data_17"])
    del _pt_tmp_18
    _pt_tmp_62 = actx.np.reshape(_pt_data_17, (216, 1))
    _pt_data_18 = actx.thaw(npzfile["_pt_data_18"])
    _pt_tmp_67 = 1.0 * _pt_data_18
    _pt_data_19 = actx.thaw(npzfile["_pt_data_19"])
    _pt_tmp_73 = actx.np.reshape(_pt_data_19, (24, 1))
    _pt_data_20 = actx.thaw(npzfile["_pt_data_20"])
    _pt_tmp_74 = (
        _pt_data_8[_pt_data_20]
        if actx.permits_advanced_indexing
        else actx.call_loopy(_pt_t_unit_7, in_0=_pt_data_8, in_1=_pt_data_20)["out"]
    )
    _pt_tmp_72 = (
        _actx_in_1_5_0[_pt_tmp_73, _pt_tmp_74]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_8, in_0=_actx_in_1_5_0, in_1=_pt_tmp_73, in_2=_pt_tmp_74
        )["out"]
    )
    _pt_tmp_72 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_72)
    _pt_tmp_72 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_72)
    _pt_tmp_71 = 0.0 + _pt_tmp_72
    _pt_tmp_70 = _pt_tmp_71 - _pt_tmp_71
    del _pt_tmp_72
    _pt_tmp_69 = 1.0 * _pt_tmp_70
    del _pt_tmp_71
    _pt_data_21 = actx.thaw(npzfile["_pt_data_21"])
    _pt_tmp_79 = 1.0 * _pt_data_21
    _pt_tmp_83 = (
        _actx_in_1_1_0[_pt_tmp_73, _pt_tmp_74]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_8, in_0=_actx_in_1_1_0, in_1=_pt_tmp_73, in_2=_pt_tmp_74
        )["out"]
    )
    _pt_tmp_83 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_83)
    _pt_tmp_83 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_83)
    _pt_tmp_82 = 0.0 + _pt_tmp_83
    _pt_tmp_81 = -1 * _pt_tmp_82
    del _pt_tmp_83
    _pt_tmp_80 = _pt_tmp_81 - _pt_tmp_82
    _pt_tmp_78 = (
        _pt_tmp_79 * _pt_tmp_80
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_79, _in1=_pt_tmp_80)["out"]
    )
    del _pt_tmp_81, _pt_tmp_82
    _pt_tmp_77 = 0.0 + _pt_tmp_78
    _pt_tmp_85 = -1.0 * _pt_data_18
    del _pt_tmp_78
    _pt_tmp_89 = (
        _actx_in_1_0_0[_pt_tmp_73, _pt_tmp_74]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_8, in_0=_actx_in_1_0_0, in_1=_pt_tmp_73, in_2=_pt_tmp_74
        )["out"]
    )
    _pt_tmp_89 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_89)
    _pt_tmp_89 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_89)
    _pt_tmp_88 = 0.0 + _pt_tmp_89
    _pt_tmp_87 = -1 * _pt_tmp_88
    del _pt_tmp_89
    _pt_tmp_86 = _pt_tmp_87 - _pt_tmp_88
    _pt_tmp_84 = (
        _pt_tmp_85 * _pt_tmp_86
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_85, _in1=_pt_tmp_86)["out"]
    )
    del _pt_tmp_87, _pt_tmp_88
    _pt_tmp_76 = _pt_tmp_77 + _pt_tmp_84
    _pt_tmp_75 = 0.5 * _pt_tmp_76
    del _pt_tmp_77, _pt_tmp_84
    _pt_tmp_68 = _pt_tmp_69 - _pt_tmp_75
    del _pt_tmp_76
    _pt_tmp_66 = (
        _pt_tmp_67 * _pt_tmp_68
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_67, _in1=_pt_tmp_68)["out"]
    )
    del _pt_tmp_69, _pt_tmp_75
    _pt_tmp_65 = 0.0 + _pt_tmp_66
    _pt_tmp_64 = -0.5 * _pt_tmp_65
    del _pt_tmp_66
    _pt_data_22 = actx.thaw(npzfile["_pt_data_22"])
    del _pt_tmp_65
    _pt_tmp_90 = actx.np.reshape(_pt_data_22, (216, 1))
    _pt_tmp_63 = (
        _pt_tmp_64[_pt_tmp_90, _pt_tmp_55]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_10, in_0=_pt_tmp_64, in_1=_pt_tmp_90, in_2=_pt_tmp_55
        )["out"]
    )
    _pt_tmp_61 = (
        actx.np.where(_pt_tmp_62, _pt_tmp_63, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_6, _in0=_pt_tmp_62, _in1=_pt_tmp_63)["out"]
    )
    del _pt_tmp_64
    _pt_tmp_61 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_61)
    del _pt_tmp_63
    _pt_tmp_61 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_61)
    _pt_tmp_60 = 0.0 + _pt_tmp_61
    _pt_tmp_59 = 0.0 + _pt_tmp_60
    del _pt_tmp_61
    _pt_tmp_91 = actx.np.zeros((216, 5), dtype=np.float64)
    del _pt_tmp_60
    _pt_tmp_58 = _pt_tmp_59 + _pt_tmp_91
    _pt_tmp_57 = _pt_tmp_58 + _pt_tmp_91
    del _pt_tmp_59
    _pt_tmp_56 = _pt_tmp_57 + _pt_tmp_91
    del _pt_tmp_58
    _pt_tmp_16 = _pt_tmp_17 + _pt_tmp_56
    del _pt_tmp_57
    _pt_tmp_15 = actx.np.reshape(_pt_tmp_16, (3, 72, 5))
    del _pt_tmp_17, _pt_tmp_56
    _pt_tmp_15 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_15)
    del _pt_tmp_16
    _pt_tmp_12 = actx.einsum("ijk,jl,jlk->li", _pt_data_3, _pt_tmp_13, _pt_tmp_15)
    _pt_tmp_12 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_12)
    del _pt_tmp_15
    _pt_tmp_9 = actx.einsum("i,jk,ik->ij", _pt_tmp_10, _pt_data_2, _pt_tmp_12)
    _pt_tmp_9 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_9)
    del _pt_tmp_12
    _pt_tmp_1 = _pt_tmp_2 - _pt_tmp_9
    _pt_tmp_0 = _pt_tmp_1 / np.float64(1.0)
    del _pt_tmp_2, _pt_tmp_9
    _pt_tmp_100 = _pt_data[0]
    del _pt_tmp_1
    _pt_tmp_99 = _pt_tmp_100[:, :, 0]
    _pt_tmp_98 = actx.einsum("ij,ikl,jl->jk", _pt_tmp_99, _pt_data_0, _actx_in_1_5_0)
    del _pt_tmp_100
    _pt_tmp_98 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_98)
    _pt_tmp_97 = -1.0 * _pt_tmp_98
    _pt_tmp_96 = 0.0 + _pt_tmp_97
    del _pt_tmp_98
    _pt_tmp_95 = 0.0 - _pt_tmp_96
    del _pt_tmp_97
    _pt_tmp_94 = -1 * _pt_tmp_95
    del _pt_tmp_96
    _pt_tmp_112 = -1.0 * _pt_data_13
    del _pt_tmp_95
    _pt_tmp_111 = (
        _pt_tmp_112 * _pt_tmp_26
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_112, _in1=_pt_tmp_26)["out"]
    )
    _pt_tmp_110 = 0.0 + _pt_tmp_111
    del _pt_tmp_26
    _pt_tmp_109 = -0.5 * _pt_tmp_110
    del _pt_tmp_111
    _pt_tmp_108 = (
        _pt_tmp_109[_pt_tmp_54, _pt_tmp_55]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_5, in_0=_pt_tmp_109, in_1=_pt_tmp_54, in_2=_pt_tmp_55
        )["out"]
    )
    del _pt_tmp_110
    _pt_tmp_107 = (
        actx.np.where(_pt_tmp_20, _pt_tmp_108, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_6, _in0=_pt_tmp_20, _in1=_pt_tmp_108)["out"]
    )
    del _pt_tmp_109
    _pt_tmp_107 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_107)
    del _pt_tmp_108
    _pt_tmp_107 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_107)
    _pt_tmp_106 = 0.0 + _pt_tmp_107
    _pt_tmp_105 = 0.0 + _pt_tmp_106
    del _pt_tmp_107
    _pt_tmp_123 = -1.0 * _pt_data_21
    del _pt_tmp_106
    _pt_tmp_122 = (
        _pt_tmp_123 * _pt_tmp_68
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_123, _in1=_pt_tmp_68)["out"]
    )
    _pt_tmp_121 = 0.0 + _pt_tmp_122
    del _pt_tmp_68
    _pt_tmp_120 = -0.5 * _pt_tmp_121
    del _pt_tmp_122
    _pt_tmp_119 = (
        _pt_tmp_120[_pt_tmp_90, _pt_tmp_55]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_10, in_0=_pt_tmp_120, in_1=_pt_tmp_90, in_2=_pt_tmp_55
        )["out"]
    )
    del _pt_tmp_121
    _pt_tmp_118 = (
        actx.np.where(_pt_tmp_62, _pt_tmp_119, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_6, _in0=_pt_tmp_62, _in1=_pt_tmp_119)["out"]
    )
    del _pt_tmp_120
    _pt_tmp_118 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_118)
    del _pt_tmp_119
    _pt_tmp_118 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_118)
    _pt_tmp_117 = 0.0 + _pt_tmp_118
    _pt_tmp_116 = 0.0 + _pt_tmp_117
    del _pt_tmp_118
    _pt_tmp_115 = _pt_tmp_116 + _pt_tmp_91
    del _pt_tmp_117
    _pt_tmp_114 = _pt_tmp_115 + _pt_tmp_91
    del _pt_tmp_116
    _pt_tmp_113 = _pt_tmp_114 + _pt_tmp_91
    del _pt_tmp_115
    _pt_tmp_104 = _pt_tmp_105 + _pt_tmp_113
    del _pt_tmp_114
    _pt_tmp_103 = actx.np.reshape(_pt_tmp_104, (3, 72, 5))
    del _pt_tmp_105, _pt_tmp_113
    _pt_tmp_103 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_103)
    del _pt_tmp_104
    _pt_tmp_102 = actx.einsum("ijk,jl,jlk->li", _pt_data_3, _pt_tmp_13, _pt_tmp_103)
    _pt_tmp_102 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_102)
    del _pt_tmp_103
    _pt_tmp_101 = actx.einsum("i,jk,ik->ij", _pt_tmp_10, _pt_data_2, _pt_tmp_102)
    _pt_tmp_101 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_101)
    del _pt_tmp_102
    _pt_tmp_93 = _pt_tmp_94 - _pt_tmp_101
    _pt_tmp_92 = _pt_tmp_93 / np.float64(1.0)
    del _pt_tmp_101, _pt_tmp_94
    _pt_tmp_131 = actx.einsum(
        "ij,ikl,jl->jk", _pt_tmp_99, _pt_data_0, _actx_in_1_4_0
    )
    del _pt_tmp_93
    _pt_tmp_131 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_131)
    _pt_tmp_130 = 1.0 * _pt_tmp_131
    _pt_tmp_129 = 0.0 + _pt_tmp_130
    del _pt_tmp_131
    _pt_tmp_133 = actx.einsum("ij,ikl,jl->jk", _pt_tmp_7, _pt_data_0, _actx_in_1_3_0)
    del _pt_tmp_130
    _pt_tmp_133 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_133)
    _pt_tmp_132 = -1.0 * _pt_tmp_133
    _pt_tmp_128 = _pt_tmp_129 + _pt_tmp_132
    del _pt_tmp_133
    _pt_tmp_127 = 0.0 - _pt_tmp_128
    del _pt_tmp_129, _pt_tmp_132
    _pt_tmp_126 = -1 * _pt_tmp_127
    del _pt_tmp_128
    _pt_tmp_152 = (
        _actx_in_1_4_0[_pt_tmp_33, _pt_tmp_34]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_actx_in_1_4_0, in_1=_pt_tmp_33, in_2=_pt_tmp_34
        )["out"]
    )
    del _pt_tmp_127
    _pt_tmp_152 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_152)
    _pt_tmp_152 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_152)
    _pt_tmp_151 = 0.0 + _pt_tmp_152
    _pt_tmp_150 = (
        _pt_tmp_151[_pt_tmp_35, _pt_tmp_36]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_151, in_1=_pt_tmp_35, in_2=_pt_tmp_36
        )["out"]
    )
    del _pt_tmp_152
    _pt_tmp_150 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_150)
    _pt_tmp_150 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_150)
    _pt_tmp_149 = 0.0 + _pt_tmp_150
    _pt_tmp_148 = _pt_tmp_149 - _pt_tmp_151
    del _pt_tmp_150
    _pt_tmp_147 = 1.0 * _pt_tmp_148
    del _pt_tmp_149, _pt_tmp_151
    _pt_tmp_160 = (
        _actx_in_1_2_0[_pt_tmp_33, _pt_tmp_34]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_actx_in_1_2_0, in_1=_pt_tmp_33, in_2=_pt_tmp_34
        )["out"]
    )
    _pt_tmp_160 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_160)
    _pt_tmp_160 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_160)
    _pt_tmp_159 = 0.0 + _pt_tmp_160
    _pt_tmp_158 = (
        _pt_tmp_159[_pt_tmp_35, _pt_tmp_36]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_159, in_1=_pt_tmp_35, in_2=_pt_tmp_36
        )["out"]
    )
    del _pt_tmp_160
    _pt_tmp_158 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_158)
    _pt_tmp_158 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_158)
    _pt_tmp_157 = 0.0 + _pt_tmp_158
    _pt_tmp_156 = _pt_tmp_157 - _pt_tmp_159
    del _pt_tmp_158
    _pt_tmp_155 = (
        _pt_tmp_112 * _pt_tmp_156
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_112, _in1=_pt_tmp_156)["out"]
    )
    del _pt_tmp_157, _pt_tmp_159
    _pt_tmp_154 = 0.0 + _pt_tmp_155
    _pt_tmp_153 = 0.5 * _pt_tmp_154
    del _pt_tmp_155
    _pt_tmp_146 = _pt_tmp_147 - _pt_tmp_153
    del _pt_tmp_154
    _pt_tmp_145 = (
        _pt_tmp_41 * _pt_tmp_146
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_41, _in1=_pt_tmp_146)["out"]
    )
    del _pt_tmp_147, _pt_tmp_153
    _pt_tmp_144 = 0.0 + _pt_tmp_145
    del _pt_tmp_146
    _pt_tmp_168 = (
        _actx_in_1_3_0[_pt_tmp_33, _pt_tmp_34]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_actx_in_1_3_0, in_1=_pt_tmp_33, in_2=_pt_tmp_34
        )["out"]
    )
    del _pt_tmp_145
    _pt_tmp_168 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_168)
    del _pt_tmp_33, _pt_tmp_34
    _pt_tmp_168 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_168)
    _pt_tmp_167 = 0.0 + _pt_tmp_168
    _pt_tmp_166 = (
        _pt_tmp_167[_pt_tmp_35, _pt_tmp_36]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_167, in_1=_pt_tmp_35, in_2=_pt_tmp_36
        )["out"]
    )
    del _pt_tmp_168
    _pt_tmp_166 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_166)
    del _pt_tmp_35, _pt_tmp_36
    _pt_tmp_166 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_166)
    _pt_tmp_165 = 0.0 + _pt_tmp_166
    _pt_tmp_164 = _pt_tmp_165 - _pt_tmp_167
    del _pt_tmp_166
    _pt_tmp_163 = 1.0 * _pt_tmp_164
    del _pt_tmp_165, _pt_tmp_167
    _pt_tmp_171 = (
        _pt_tmp_25 * _pt_tmp_156
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_25, _in1=_pt_tmp_156)["out"]
    )
    _pt_tmp_170 = 0.0 + _pt_tmp_171
    _pt_tmp_169 = 0.5 * _pt_tmp_170
    del _pt_tmp_171
    _pt_tmp_162 = _pt_tmp_163 - _pt_tmp_169
    del _pt_tmp_170
    _pt_tmp_161 = (
        _pt_tmp_48 * _pt_tmp_162
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_48, _in1=_pt_tmp_162)["out"]
    )
    del _pt_tmp_163, _pt_tmp_169
    _pt_tmp_143 = _pt_tmp_144 + _pt_tmp_161
    del _pt_tmp_162
    _pt_tmp_142 = -0.5 * _pt_tmp_143
    del _pt_tmp_144, _pt_tmp_161
    _pt_tmp_141 = (
        _pt_tmp_142[_pt_tmp_54, _pt_tmp_55]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_5, in_0=_pt_tmp_142, in_1=_pt_tmp_54, in_2=_pt_tmp_55
        )["out"]
    )
    del _pt_tmp_143
    _pt_tmp_140 = (
        actx.np.where(_pt_tmp_20, _pt_tmp_141, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_6, _in0=_pt_tmp_20, _in1=_pt_tmp_141)["out"]
    )
    del _pt_tmp_142
    _pt_tmp_140 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_140)
    del _pt_tmp_141
    _pt_tmp_140 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_140)
    _pt_tmp_139 = 0.0 + _pt_tmp_140
    _pt_tmp_138 = 0.0 + _pt_tmp_139
    del _pt_tmp_140
    _pt_tmp_187 = (
        _actx_in_1_4_0[_pt_tmp_73, _pt_tmp_74]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_8, in_0=_actx_in_1_4_0, in_1=_pt_tmp_73, in_2=_pt_tmp_74
        )["out"]
    )
    del _pt_tmp_139
    _pt_tmp_187 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_187)
    _pt_tmp_187 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_187)
    _pt_tmp_186 = 0.0 + _pt_tmp_187
    _pt_tmp_185 = _pt_tmp_186 - _pt_tmp_186
    del _pt_tmp_187
    _pt_tmp_184 = 1.0 * _pt_tmp_185
    del _pt_tmp_186
    _pt_tmp_194 = (
        _actx_in_1_2_0[_pt_tmp_73, _pt_tmp_74]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_8, in_0=_actx_in_1_2_0, in_1=_pt_tmp_73, in_2=_pt_tmp_74
        )["out"]
    )
    _pt_tmp_194 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_194)
    _pt_tmp_194 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_194)
    _pt_tmp_193 = 0.0 + _pt_tmp_194
    _pt_tmp_192 = -1 * _pt_tmp_193
    del _pt_tmp_194
    _pt_tmp_191 = _pt_tmp_192 - _pt_tmp_193
    _pt_tmp_190 = (
        _pt_tmp_123 * _pt_tmp_191
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_123, _in1=_pt_tmp_191)["out"]
    )
    del _pt_tmp_192, _pt_tmp_193
    _pt_tmp_189 = 0.0 + _pt_tmp_190
    _pt_tmp_188 = 0.5 * _pt_tmp_189
    del _pt_tmp_190
    _pt_tmp_183 = _pt_tmp_184 - _pt_tmp_188
    del _pt_tmp_189
    _pt_tmp_182 = (
        _pt_tmp_79 * _pt_tmp_183
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_79, _in1=_pt_tmp_183)["out"]
    )
    del _pt_tmp_184, _pt_tmp_188
    _pt_tmp_181 = 0.0 + _pt_tmp_182
    del _pt_tmp_183
    _pt_tmp_200 = (
        _actx_in_1_3_0[_pt_tmp_73, _pt_tmp_74]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_8, in_0=_actx_in_1_3_0, in_1=_pt_tmp_73, in_2=_pt_tmp_74
        )["out"]
    )
    del _pt_tmp_182
    _pt_tmp_200 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_200)
    del _pt_tmp_73, _pt_tmp_74
    _pt_tmp_200 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_200)
    _pt_tmp_199 = 0.0 + _pt_tmp_200
    _pt_tmp_198 = _pt_tmp_199 - _pt_tmp_199
    del _pt_tmp_200
    _pt_tmp_197 = 1.0 * _pt_tmp_198
    del _pt_tmp_199
    _pt_tmp_203 = (
        _pt_tmp_67 * _pt_tmp_191
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_67, _in1=_pt_tmp_191)["out"]
    )
    _pt_tmp_202 = 0.0 + _pt_tmp_203
    _pt_tmp_201 = 0.5 * _pt_tmp_202
    del _pt_tmp_203
    _pt_tmp_196 = _pt_tmp_197 - _pt_tmp_201
    del _pt_tmp_202
    _pt_tmp_195 = (
        _pt_tmp_85 * _pt_tmp_196
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_85, _in1=_pt_tmp_196)["out"]
    )
    del _pt_tmp_197, _pt_tmp_201
    _pt_tmp_180 = _pt_tmp_181 + _pt_tmp_195
    del _pt_tmp_196
    _pt_tmp_179 = -0.5 * _pt_tmp_180
    del _pt_tmp_181, _pt_tmp_195
    _pt_tmp_178 = (
        _pt_tmp_179[_pt_tmp_90, _pt_tmp_55]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_10, in_0=_pt_tmp_179, in_1=_pt_tmp_90, in_2=_pt_tmp_55
        )["out"]
    )
    del _pt_tmp_180
    _pt_tmp_177 = (
        actx.np.where(_pt_tmp_62, _pt_tmp_178, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_6, _in0=_pt_tmp_62, _in1=_pt_tmp_178)["out"]
    )
    del _pt_tmp_179
    _pt_tmp_177 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_177)
    del _pt_tmp_178
    _pt_tmp_177 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_177)
    _pt_tmp_176 = 0.0 + _pt_tmp_177
    _pt_tmp_175 = 0.0 + _pt_tmp_176
    del _pt_tmp_177
    _pt_tmp_174 = _pt_tmp_175 + _pt_tmp_91
    del _pt_tmp_176
    _pt_tmp_173 = _pt_tmp_174 + _pt_tmp_91
    del _pt_tmp_175
    _pt_tmp_172 = _pt_tmp_173 + _pt_tmp_91
    del _pt_tmp_174
    _pt_tmp_137 = _pt_tmp_138 + _pt_tmp_172
    del _pt_tmp_173
    _pt_tmp_136 = actx.np.reshape(_pt_tmp_137, (3, 72, 5))
    del _pt_tmp_138, _pt_tmp_172
    _pt_tmp_136 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_136)
    del _pt_tmp_137
    _pt_tmp_135 = actx.einsum("ijk,jl,jlk->li", _pt_data_3, _pt_tmp_13, _pt_tmp_136)
    _pt_tmp_135 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_135)
    del _pt_tmp_136
    _pt_tmp_134 = actx.einsum("i,jk,ik->ij", _pt_tmp_10, _pt_data_2, _pt_tmp_135)
    _pt_tmp_134 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_134)
    del _pt_tmp_135
    _pt_tmp_125 = _pt_tmp_126 - _pt_tmp_134
    _pt_tmp_124 = _pt_tmp_125 / np.float64(1.0)
    del _pt_tmp_126, _pt_tmp_134
    _pt_tmp_209 = actx.einsum("ij,ikl,jl->jk", _pt_tmp_7, _pt_data_0, _actx_in_1_2_0)
    del _pt_tmp_125
    _pt_tmp_209 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_209)
    _pt_tmp_208 = 1.0 * _pt_tmp_209
    _pt_tmp_207 = 0.0 + _pt_tmp_208
    del _pt_tmp_209
    _pt_tmp_206 = -1 * _pt_tmp_207
    del _pt_tmp_208
    _pt_tmp_222 = 1.0 * _pt_tmp_156
    del _pt_tmp_207
    _pt_tmp_226 = (
        _pt_tmp_41 * _pt_tmp_148
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_41, _in1=_pt_tmp_148)["out"]
    )
    del _pt_tmp_156
    _pt_tmp_225 = 0.0 + _pt_tmp_226
    del _pt_tmp_148
    _pt_tmp_227 = (
        _pt_tmp_48 * _pt_tmp_164
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_48, _in1=_pt_tmp_164)["out"]
    )
    del _pt_tmp_226
    _pt_tmp_224 = _pt_tmp_225 + _pt_tmp_227
    del _pt_tmp_164
    _pt_tmp_223 = 0.5 * _pt_tmp_224
    del _pt_tmp_225, _pt_tmp_227
    _pt_tmp_221 = _pt_tmp_222 + _pt_tmp_223
    del _pt_tmp_224
    _pt_tmp_220 = (
        _pt_tmp_25 * _pt_tmp_221
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_25, _in1=_pt_tmp_221)["out"]
    )
    del _pt_tmp_222, _pt_tmp_223
    _pt_tmp_219 = 0.0 + _pt_tmp_220
    _pt_tmp_218 = 0.5 * _pt_tmp_219
    del _pt_tmp_220
    _pt_tmp_217 = (
        _pt_tmp_218[_pt_tmp_54, _pt_tmp_55]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_5, in_0=_pt_tmp_218, in_1=_pt_tmp_54, in_2=_pt_tmp_55
        )["out"]
    )
    del _pt_tmp_219
    _pt_tmp_216 = (
        actx.np.where(_pt_tmp_20, _pt_tmp_217, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_6, _in0=_pt_tmp_20, _in1=_pt_tmp_217)["out"]
    )
    del _pt_tmp_218
    _pt_tmp_216 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_216)
    del _pt_tmp_217
    _pt_tmp_216 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_216)
    _pt_tmp_215 = 0.0 + _pt_tmp_216
    _pt_tmp_214 = 0.0 + _pt_tmp_215
    del _pt_tmp_216
    _pt_tmp_239 = 1.0 * _pt_tmp_191
    del _pt_tmp_215
    _pt_tmp_243 = (
        _pt_tmp_79 * _pt_tmp_185
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_79, _in1=_pt_tmp_185)["out"]
    )
    del _pt_tmp_191
    _pt_tmp_242 = 0.0 + _pt_tmp_243
    del _pt_tmp_185
    _pt_tmp_244 = (
        _pt_tmp_85 * _pt_tmp_198
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_85, _in1=_pt_tmp_198)["out"]
    )
    del _pt_tmp_243
    _pt_tmp_241 = _pt_tmp_242 + _pt_tmp_244
    del _pt_tmp_198
    _pt_tmp_240 = 0.5 * _pt_tmp_241
    del _pt_tmp_242, _pt_tmp_244
    _pt_tmp_238 = _pt_tmp_239 + _pt_tmp_240
    del _pt_tmp_241
    _pt_tmp_237 = (
        _pt_tmp_67 * _pt_tmp_238
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_67, _in1=_pt_tmp_238)["out"]
    )
    del _pt_tmp_239, _pt_tmp_240
    _pt_tmp_236 = 0.0 + _pt_tmp_237
    _pt_tmp_235 = 0.5 * _pt_tmp_236
    del _pt_tmp_237
    _pt_tmp_234 = (
        _pt_tmp_235[_pt_tmp_90, _pt_tmp_55]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_10, in_0=_pt_tmp_235, in_1=_pt_tmp_90, in_2=_pt_tmp_55
        )["out"]
    )
    del _pt_tmp_236
    _pt_tmp_233 = (
        actx.np.where(_pt_tmp_62, _pt_tmp_234, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_6, _in0=_pt_tmp_62, _in1=_pt_tmp_234)["out"]
    )
    del _pt_tmp_235
    _pt_tmp_233 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_233)
    del _pt_tmp_234
    _pt_tmp_233 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_233)
    _pt_tmp_232 = 0.0 + _pt_tmp_233
    _pt_tmp_231 = 0.0 + _pt_tmp_232
    del _pt_tmp_233
    _pt_tmp_230 = _pt_tmp_231 + _pt_tmp_91
    del _pt_tmp_232
    _pt_tmp_229 = _pt_tmp_230 + _pt_tmp_91
    del _pt_tmp_231
    _pt_tmp_228 = _pt_tmp_229 + _pt_tmp_91
    del _pt_tmp_230
    _pt_tmp_213 = _pt_tmp_214 + _pt_tmp_228
    del _pt_tmp_229
    _pt_tmp_212 = actx.np.reshape(_pt_tmp_213, (3, 72, 5))
    del _pt_tmp_214, _pt_tmp_228
    _pt_tmp_212 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_212)
    del _pt_tmp_213
    _pt_tmp_211 = actx.einsum("ijk,jl,jlk->li", _pt_data_3, _pt_tmp_13, _pt_tmp_212)
    _pt_tmp_211 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_211)
    del _pt_tmp_212
    _pt_tmp_210 = actx.einsum("i,jk,ik->ij", _pt_tmp_10, _pt_data_2, _pt_tmp_211)
    _pt_tmp_210 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_210)
    del _pt_tmp_211
    _pt_tmp_205 = _pt_tmp_206 - _pt_tmp_210
    _pt_tmp_204 = _pt_tmp_205 / np.float64(1.0)
    del _pt_tmp_206, _pt_tmp_210
    _pt_tmp_250 = actx.einsum(
        "ij,ikl,jl->jk", _pt_tmp_99, _pt_data_0, _actx_in_1_2_0
    )
    del _pt_tmp_205
    _pt_tmp_250 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_250)
    _pt_tmp_249 = -1.0 * _pt_tmp_250
    _pt_tmp_248 = 0.0 + _pt_tmp_249
    del _pt_tmp_250
    _pt_tmp_247 = -1 * _pt_tmp_248
    del _pt_tmp_249
    _pt_tmp_261 = (
        _pt_tmp_112 * _pt_tmp_221
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_112, _in1=_pt_tmp_221)["out"]
    )
    del _pt_tmp_248
    _pt_tmp_260 = 0.0 + _pt_tmp_261
    del _pt_tmp_221
    _pt_tmp_259 = 0.5 * _pt_tmp_260
    del _pt_tmp_261
    _pt_tmp_258 = (
        _pt_tmp_259[_pt_tmp_54, _pt_tmp_55]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_5, in_0=_pt_tmp_259, in_1=_pt_tmp_54, in_2=_pt_tmp_55
        )["out"]
    )
    del _pt_tmp_260
    _pt_tmp_257 = (
        actx.np.where(_pt_tmp_20, _pt_tmp_258, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_6, _in0=_pt_tmp_20, _in1=_pt_tmp_258)["out"]
    )
    del _pt_tmp_259
    _pt_tmp_257 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_257)
    del _pt_tmp_258
    _pt_tmp_257 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_257)
    _pt_tmp_256 = 0.0 + _pt_tmp_257
    _pt_tmp_255 = 0.0 + _pt_tmp_256
    del _pt_tmp_257
    _pt_tmp_271 = (
        _pt_tmp_123 * _pt_tmp_238
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_123, _in1=_pt_tmp_238)["out"]
    )
    del _pt_tmp_256
    _pt_tmp_270 = 0.0 + _pt_tmp_271
    del _pt_tmp_238
    _pt_tmp_269 = 0.5 * _pt_tmp_270
    del _pt_tmp_271
    _pt_tmp_268 = (
        _pt_tmp_269[_pt_tmp_90, _pt_tmp_55]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_10, in_0=_pt_tmp_269, in_1=_pt_tmp_90, in_2=_pt_tmp_55
        )["out"]
    )
    del _pt_tmp_270
    _pt_tmp_267 = (
        actx.np.where(_pt_tmp_62, _pt_tmp_268, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_6, _in0=_pt_tmp_62, _in1=_pt_tmp_268)["out"]
    )
    del _pt_tmp_269
    _pt_tmp_267 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_267)
    del _pt_tmp_268
    _pt_tmp_267 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_267)
    _pt_tmp_266 = 0.0 + _pt_tmp_267
    _pt_tmp_265 = 0.0 + _pt_tmp_266
    del _pt_tmp_267
    _pt_tmp_264 = _pt_tmp_265 + _pt_tmp_91
    del _pt_tmp_266
    _pt_tmp_263 = _pt_tmp_264 + _pt_tmp_91
    del _pt_tmp_265
    _pt_tmp_262 = _pt_tmp_263 + _pt_tmp_91
    del _pt_tmp_264
    _pt_tmp_254 = _pt_tmp_255 + _pt_tmp_262
    del _pt_tmp_263
    _pt_tmp_253 = actx.np.reshape(_pt_tmp_254, (3, 72, 5))
    del _pt_tmp_255, _pt_tmp_262
    _pt_tmp_253 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_253)
    del _pt_tmp_254
    _pt_tmp_252 = actx.einsum("ijk,jl,jlk->li", _pt_data_3, _pt_tmp_13, _pt_tmp_253)
    _pt_tmp_252 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_252)
    del _pt_tmp_253
    _pt_tmp_251 = actx.einsum("i,jk,ik->ij", _pt_tmp_10, _pt_data_2, _pt_tmp_252)
    _pt_tmp_251 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_251)
    del _pt_tmp_252
    _pt_tmp_246 = _pt_tmp_247 - _pt_tmp_251
    _pt_tmp_245 = _pt_tmp_246 / np.float64(1.0)
    del _pt_tmp_247, _pt_tmp_251
    _pt_tmp_278 = actx.einsum(
        "ij,ikl,jl->jk", _pt_tmp_99, _pt_data_0, _actx_in_1_1_0
    )
    del _pt_tmp_246
    _pt_tmp_278 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_278)
    del _pt_tmp_99
    _pt_tmp_277 = 1.0 * _pt_tmp_278
    _pt_tmp_276 = 0.0 + _pt_tmp_277
    del _pt_tmp_278
    _pt_tmp_280 = actx.einsum("ij,ikl,jl->jk", _pt_tmp_7, _pt_data_0, _actx_in_1_0_0)
    del _pt_tmp_277
    _pt_tmp_280 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_280)
    del _pt_tmp_7
    _pt_tmp_279 = -1.0 * _pt_tmp_280
    _pt_tmp_275 = _pt_tmp_276 + _pt_tmp_279
    del _pt_tmp_280
    _pt_tmp_274 = -1 * _pt_tmp_275
    del _pt_tmp_276, _pt_tmp_279
    _pt_tmp_294 = 1.0 * _pt_tmp_42
    del _pt_tmp_275
    _pt_tmp_297 = (
        _pt_tmp_112 * _pt_tmp_28
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_112, _in1=_pt_tmp_28)["out"]
    )
    del _pt_tmp_42
    _pt_tmp_296 = 0.0 + _pt_tmp_297
    del _pt_tmp_112
    _pt_tmp_295 = 0.5 * _pt_tmp_296
    del _pt_tmp_297
    _pt_tmp_293 = _pt_tmp_294 + _pt_tmp_295
    del _pt_tmp_296
    _pt_tmp_292 = (
        _pt_tmp_41 * _pt_tmp_293
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_41, _in1=_pt_tmp_293)["out"]
    )
    del _pt_tmp_294, _pt_tmp_295
    _pt_tmp_291 = 0.0 + _pt_tmp_292
    del _pt_tmp_293, _pt_tmp_41
    _pt_tmp_300 = 1.0 * _pt_tmp_49
    del _pt_tmp_292
    _pt_tmp_303 = (
        _pt_tmp_25 * _pt_tmp_28
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_25, _in1=_pt_tmp_28)["out"]
    )
    del _pt_tmp_49
    _pt_tmp_302 = 0.0 + _pt_tmp_303
    del _pt_tmp_25, _pt_tmp_28
    _pt_tmp_301 = 0.5 * _pt_tmp_302
    del _pt_tmp_303
    _pt_tmp_299 = _pt_tmp_300 + _pt_tmp_301
    del _pt_tmp_302
    _pt_tmp_298 = (
        _pt_tmp_48 * _pt_tmp_299
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_48, _in1=_pt_tmp_299)["out"]
    )
    del _pt_tmp_300, _pt_tmp_301
    _pt_tmp_290 = _pt_tmp_291 + _pt_tmp_298
    del _pt_tmp_299, _pt_tmp_48
    _pt_tmp_289 = 0.5 * _pt_tmp_290
    del _pt_tmp_291, _pt_tmp_298
    _pt_tmp_288 = (
        _pt_tmp_289[_pt_tmp_54, _pt_tmp_55]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_5, in_0=_pt_tmp_289, in_1=_pt_tmp_54, in_2=_pt_tmp_55
        )["out"]
    )
    del _pt_tmp_290
    _pt_tmp_287 = (
        actx.np.where(_pt_tmp_20, _pt_tmp_288, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_6, _in0=_pt_tmp_20, _in1=_pt_tmp_288)["out"]
    )
    del _pt_tmp_289, _pt_tmp_54
    _pt_tmp_287 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_287)
    del _pt_tmp_20, _pt_tmp_288
    _pt_tmp_287 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_287)
    _pt_tmp_286 = 0.0 + _pt_tmp_287
    _pt_tmp_285 = 0.0 + _pt_tmp_286
    del _pt_tmp_287
    _pt_tmp_316 = 1.0 * _pt_tmp_80
    del _pt_tmp_286
    _pt_tmp_319 = (
        _pt_tmp_123 * _pt_tmp_70
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_123, _in1=_pt_tmp_70)["out"]
    )
    del _pt_tmp_80
    _pt_tmp_318 = 0.0 + _pt_tmp_319
    del _pt_tmp_123
    _pt_tmp_317 = 0.5 * _pt_tmp_318
    del _pt_tmp_319
    _pt_tmp_315 = _pt_tmp_316 + _pt_tmp_317
    del _pt_tmp_318
    _pt_tmp_314 = (
        _pt_tmp_79 * _pt_tmp_315
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_79, _in1=_pt_tmp_315)["out"]
    )
    del _pt_tmp_316, _pt_tmp_317
    _pt_tmp_313 = 0.0 + _pt_tmp_314
    del _pt_tmp_315, _pt_tmp_79
    _pt_tmp_322 = 1.0 * _pt_tmp_86
    del _pt_tmp_314
    _pt_tmp_325 = (
        _pt_tmp_67 * _pt_tmp_70
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_67, _in1=_pt_tmp_70)["out"]
    )
    del _pt_tmp_86
    _pt_tmp_324 = 0.0 + _pt_tmp_325
    del _pt_tmp_67, _pt_tmp_70
    _pt_tmp_323 = 0.5 * _pt_tmp_324
    del _pt_tmp_325
    _pt_tmp_321 = _pt_tmp_322 + _pt_tmp_323
    del _pt_tmp_324
    _pt_tmp_320 = (
        _pt_tmp_85 * _pt_tmp_321
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_85, _in1=_pt_tmp_321)["out"]
    )
    del _pt_tmp_322, _pt_tmp_323
    _pt_tmp_312 = _pt_tmp_313 + _pt_tmp_320
    del _pt_tmp_321, _pt_tmp_85
    _pt_tmp_311 = 0.5 * _pt_tmp_312
    del _pt_tmp_313, _pt_tmp_320
    _pt_tmp_310 = (
        _pt_tmp_311[_pt_tmp_90, _pt_tmp_55]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_10, in_0=_pt_tmp_311, in_1=_pt_tmp_90, in_2=_pt_tmp_55
        )["out"]
    )
    del _pt_tmp_312
    _pt_tmp_309 = (
        actx.np.where(_pt_tmp_62, _pt_tmp_310, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_6, _in0=_pt_tmp_62, _in1=_pt_tmp_310)["out"]
    )
    del _pt_tmp_311, _pt_tmp_55, _pt_tmp_90
    _pt_tmp_309 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_309)
    del _pt_tmp_310, _pt_tmp_62
    _pt_tmp_309 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_309)
    _pt_tmp_308 = 0.0 + _pt_tmp_309
    _pt_tmp_307 = 0.0 + _pt_tmp_308
    del _pt_tmp_309
    _pt_tmp_306 = _pt_tmp_307 + _pt_tmp_91
    del _pt_tmp_308
    _pt_tmp_305 = _pt_tmp_306 + _pt_tmp_91
    del _pt_tmp_307
    _pt_tmp_304 = _pt_tmp_305 + _pt_tmp_91
    del _pt_tmp_306
    _pt_tmp_284 = _pt_tmp_285 + _pt_tmp_304
    del _pt_tmp_305, _pt_tmp_91
    _pt_tmp_283 = actx.np.reshape(_pt_tmp_284, (3, 72, 5))
    del _pt_tmp_285, _pt_tmp_304
    _pt_tmp_283 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_283)
    del _pt_tmp_284
    _pt_tmp_282 = actx.einsum("ijk,jl,jlk->li", _pt_data_3, _pt_tmp_13, _pt_tmp_283)
    _pt_tmp_282 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_282)
    del _pt_tmp_13, _pt_tmp_283
    _pt_tmp_281 = actx.einsum("i,jk,ik->ij", _pt_tmp_10, _pt_data_2, _pt_tmp_282)
    _pt_tmp_281 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_281)
    del _pt_tmp_10, _pt_tmp_282
    _pt_tmp_273 = _pt_tmp_274 - _pt_tmp_281
    _pt_tmp_272 = _pt_tmp_273 / np.float64(1.0)
    del _pt_tmp_274, _pt_tmp_281
    _pt_tmp = make_obj_array(
        [_pt_tmp_0, _pt_tmp_92, _pt_tmp_124, _pt_tmp_204, _pt_tmp_245, _pt_tmp_272]
    )
    del _pt_tmp_273
    return _pt_tmp
    del _pt_tmp_0, _pt_tmp_124, _pt_tmp_204, _pt_tmp_245, _pt_tmp_272, _pt_tmp_92


@dataclass(frozen=True)
class RHSInvoker:
    actx: ArrayContext

    @cached_property
    def npzfile(self):
        import os

        kw_to_ary = np.load(
            os.path.join(
                get_actx_dgfem_suite_path(), "suite/tiny_maxwell_2D_P4/literals.npz"
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
            get_actx_dgfem_suite_path(), "suite/tiny_maxwell_2D_P4/ref_outputs.pkl"
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
