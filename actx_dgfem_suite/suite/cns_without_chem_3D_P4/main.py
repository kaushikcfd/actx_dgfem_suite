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
    _actx_in_1_0_energy_0,
    _actx_in_1_0_mass_0,
    _actx_in_1_0_momentum_0_0,
    _actx_in_1_0_momentum_1_0,
    _actx_in_1_0_momentum_2_0,
    _actx_in_1_1_0,
):
    _pt_t_unit = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 191999 and 0 <= _1 <= 14 }",
        "out[_0, _1] = in_0[in_1[_0] % 4, _1]",
        [
            lp.GlobalArg("out", dtype=np.int32, shape=(192000, 15)),
            lp.GlobalArg("in_0", dtype=np.int32, shape=(4, 15)),
            lp.GlobalArg("in_1", dtype=np.int8, shape=(192000,)),
        ],
    )
    _pt_t_unit_0 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 191999 and 0 <= _1 <= 14 }",
        "out[_0, _1] = in_0[in_1[_0, 0] % 48000, in_2[_0, _1] % 35]",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(192000, 15)),
            lp.GlobalArg("in_0", dtype=np.float64, shape=(48000, 35)),
            lp.GlobalArg("in_1", dtype=np.int32, shape=(192000, 1)),
            lp.GlobalArg("in_2", dtype=np.int32, shape=(192000, 15)),
        ],
    )
    _pt_t_unit_1 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 191999 and 0 <= _1 <= 14 }",
        "out[_0, _1] = in_0[in_1[_0] % 3, _1]",
        [
            lp.GlobalArg("out", dtype=np.int32, shape=(192000, 15)),
            lp.GlobalArg("in_0", dtype=np.int32, shape=(3, 15)),
            lp.GlobalArg("in_1", dtype=np.int8, shape=(192000,)),
        ],
    )
    _pt_t_unit_2 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 191999 and 0 <= _1 <= 14 }",
        "out[_0, _1] = in_0[in_1[_0, 0] % 192000, in_2[_0, _1] % 15]",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(192000, 15)),
            lp.GlobalArg("in_0", dtype=np.float64, shape=(192000, 15)),
            lp.GlobalArg("in_1", dtype=np.int32, shape=(192000, 1)),
            lp.GlobalArg("in_2", dtype=np.int32, shape=(192000, 15)),
        ],
    )
    _pt_t_unit_3 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 191999 and 0 <= _1 <= 14 }",
        "out[_0, _1] = _in0[_0, _1]*_in1[_0, 0]",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(192000, 15)),
            lp.GlobalArg("_in0", dtype=np.float64, shape=(192000, 15)),
            lp.GlobalArg("_in1", dtype=np.float64, shape=(192000, 1)),
        ],
    )
    _pt_t_unit_4 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 191999 and 0 <= _1 <= 14 }",
        "out[_0, _1] = in_0[0, _1]",
        [
            lp.GlobalArg("out", dtype=np.int32, shape=(192000, 15)),
            lp.GlobalArg("in_0", dtype=np.int32, shape=(1, 15)),
            lp.GlobalArg("in_1", dtype=np.int8, shape=(192000,)),
        ],
    )
    _pt_t_unit_5 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 47999 and 0 <= _1 <= 34 }",
        "out[_0, _1] = _in0[_0, _1]*_in1[_0, 0]",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(48000, 35)),
            lp.GlobalArg("_in0", dtype=np.float64, shape=(48000, 35)),
            lp.GlobalArg("_in1", dtype=np.float64, shape=(48000, 1)),
        ],
    )
    _pt_t_unit_6 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 47999 and 0 <= _1 <= 34 }",
        "out[_0, _1] = in_0[_0, 0]",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(48000, 35)),
            lp.GlobalArg("in_0", dtype=np.float64, shape=(48000, 1)),
        ],
    )
    _pt_data = actx.thaw(npzfile["_pt_data"])
    _pt_data = actx.tag((PrefixNamed(prefix="area_el_vol"),), _pt_data)
    _pt_data = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_data)
    _pt_data = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_data)
    _pt_tmp_6 = 1.0 / _pt_data
    _pt_tmp_5 = _pt_tmp_6[:, 0]
    _pt_data_0 = actx.thaw(npzfile["_pt_data_0"])
    del _pt_tmp_6
    _pt_data_0 = actx.tag_axis(0, (DiscretizationDOFAxisTag(),), _pt_data_0)
    _pt_data_1 = actx.thaw(npzfile["_pt_data_1"])
    _pt_data_1 = actx.tag((PrefixNamed(prefix="inv_metric_deriv_vol"),), _pt_data_1)
    _pt_data_1 = actx.tag_axis(0, (DiscretizationAmbientDimAxisTag(),), _pt_data_1)
    _pt_data_1 = actx.tag_axis(
        1, (DiscretizationTopologicalDimAxisTag(),), _pt_data_1
    )
    _pt_data_1 = actx.tag_axis(2, (DiscretizationElementAxisTag(),), _pt_data_1)
    _pt_data_1 = actx.tag_axis(3, (DiscretizationDOFAxisTag(),), _pt_data_1)
    _pt_tmp_9 = _pt_data_1[:, :, :, 0]
    _pt_data_2 = actx.thaw(npzfile["_pt_data_2"])
    _pt_data_2 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_data_2)
    _actx_in_1_0_mass_0 = actx.tag_axis(
        0, (DiscretizationElementAxisTag(),), _actx_in_1_0_mass_0
    )
    _actx_in_1_0_mass_0 = actx.tag_axis(
        1, (DiscretizationDOFAxisTag(),), _actx_in_1_0_mass_0
    )
    _pt_tmp_20 = 0 * _actx_in_1_0_mass_0
    _pt_tmp_19 = _pt_tmp_20 + 1.0
    _pt_tmp_18 = 1e-05 * _pt_tmp_19
    del _pt_tmp_20
    _actx_in_1_0_momentum_0_0 = actx.tag_axis(
        0, (DiscretizationElementAxisTag(),), _actx_in_1_0_momentum_0_0
    )
    _actx_in_1_0_momentum_0_0 = actx.tag_axis(
        1, (DiscretizationDOFAxisTag(),), _actx_in_1_0_momentum_0_0
    )
    _pt_tmp_28 = actx.einsum(
        "ijk, jlm, km -> ikl", _pt_tmp_9, _pt_data_2, _actx_in_1_0_momentum_0_0
    )
    _pt_tmp_28 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_28)
    _pt_tmp_27 = _pt_tmp_28[0]
    _pt_data_3 = actx.thaw(npzfile["_pt_data_3"])
    _pt_data_3 = actx.tag_axis(0, (DiscretizationDOFAxisTag(),), _pt_data_3)
    _pt_data_3 = actx.tag_axis(2, (DiscretizationDOFAxisTag(),), _pt_data_3)
    _pt_data_4 = actx.thaw(npzfile["_pt_data_4"])
    _pt_data_4 = actx.tag(
        (PrefixNamed(prefix="area_el_b_face_restr_all"),), _pt_data_4
    )
    _pt_data_4 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_data_4)
    _pt_data_4 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_data_4)
    _pt_tmp_31 = actx.np.reshape(_pt_data_4, (4, 48000, 1))
    _pt_tmp_31 = actx.tag_axis(1, (DiscretizationElementAxisTag(),), _pt_tmp_31)
    _pt_tmp_30 = _pt_tmp_31[:, :, 0]
    _pt_data_5 = actx.thaw(npzfile["_pt_data_5"])
    del _pt_tmp_31
    _pt_data_5 = actx.tag((PrefixNamed(prefix="from_el_indices"),), _pt_data_5)
    _pt_tmp_44 = actx.np.reshape(_pt_data_5, (192000, 1))
    _pt_tmp_44 = actx.tag((PrefixNamed(prefix="from_el_indices"),), _pt_tmp_44)
    _pt_data_6 = actx.thaw(npzfile["_pt_data_6"])
    _pt_data_6 = actx.tag((PrefixNamed(prefix="dof_pick_lists"),), _pt_data_6)
    _pt_data_7 = actx.thaw(npzfile["_pt_data_7"])
    _pt_data_7 = actx.tag((PrefixNamed(prefix="dof_pick_list_indices"),), _pt_data_7)
    _pt_tmp_45 = (
        _pt_data_6[_pt_data_7]
        if actx.permits_advanced_indexing
        else actx.call_loopy(_pt_t_unit, in_0=_pt_data_6, in_1=_pt_data_7)["out"]
    )
    _pt_tmp_43 = (
        _actx_in_1_0_momentum_0_0[_pt_tmp_44, _pt_tmp_45]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0,
            in_0=_actx_in_1_0_momentum_0_0,
            in_1=_pt_tmp_44,
            in_2=_pt_tmp_45,
        )["out"]
    )
    _pt_tmp_43 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_43)
    _pt_tmp_43 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_43)
    _pt_tmp_42 = 0 + _pt_tmp_43
    _pt_data_8 = actx.thaw(npzfile["_pt_data_8"])
    del _pt_tmp_43
    _pt_data_8 = actx.tag((PrefixNamed(prefix="from_el_indices"),), _pt_data_8)
    _pt_tmp_46 = actx.np.reshape(_pt_data_8, (192000, 1))
    _pt_tmp_46 = actx.tag((PrefixNamed(prefix="from_el_indices"),), _pt_tmp_46)
    _pt_data_9 = actx.thaw(npzfile["_pt_data_9"])
    _pt_data_9 = actx.tag((PrefixNamed(prefix="dof_pick_lists"),), _pt_data_9)
    _pt_data_10 = actx.thaw(npzfile["_pt_data_10"])
    _pt_data_10 = actx.tag(
        (PrefixNamed(prefix="dof_pick_list_indices"),), _pt_data_10
    )
    _pt_tmp_47 = (
        _pt_data_9[_pt_data_10]
        if actx.permits_advanced_indexing
        else actx.call_loopy(_pt_t_unit_1, in_0=_pt_data_9, in_1=_pt_data_10)["out"]
    )
    _pt_tmp_41 = (
        _pt_tmp_42[_pt_tmp_46, _pt_tmp_47]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_42, in_1=_pt_tmp_46, in_2=_pt_tmp_47
        )["out"]
    )
    _pt_tmp_41 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_41)
    _pt_tmp_41 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_41)
    _pt_tmp_40 = 0 + _pt_tmp_41
    _pt_tmp_39 = _pt_tmp_40 + _pt_tmp_42
    del _pt_tmp_41
    _pt_tmp_38 = _pt_tmp_39 / 2
    _pt_data_11 = actx.thaw(npzfile["_pt_data_11"])
    del _pt_tmp_39
    _pt_data_11 = actx.tag(
        (PrefixNamed(prefix="normal_1_b_face_restr_interior"),), _pt_data_11
    )
    _pt_data_11 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_data_11)
    _pt_data_11 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_data_11)
    _pt_tmp_37 = (
        _pt_tmp_38 * _pt_data_11
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_38, _in1=_pt_data_11)["out"]
    )
    _pt_data_12 = actx.thaw(npzfile["_pt_data_12"])
    _pt_data_12 = actx.tag((PrefixNamed(prefix="from_el_indices"),), _pt_data_12)
    _pt_tmp_48 = actx.np.reshape(_pt_data_12, (192000, 1))
    _pt_tmp_48 = actx.tag((PrefixNamed(prefix="from_el_indices"),), _pt_tmp_48)
    _pt_data_13 = actx.thaw(npzfile["_pt_data_13"])
    _pt_data_13 = actx.tag((PrefixNamed(prefix="dof_pick_lists"),), _pt_data_13)
    _pt_data_14 = actx.thaw(npzfile["_pt_data_14"])
    _pt_data_14 = actx.tag(
        (PrefixNamed(prefix="dof_pick_list_indices"),), _pt_data_14
    )
    _pt_tmp_49 = (
        _pt_data_13[_pt_data_14]
        if actx.permits_advanced_indexing
        else actx.call_loopy(_pt_t_unit_4, in_0=_pt_data_13, in_1=_pt_data_14)["out"]
    )
    _pt_tmp_36 = (
        _pt_tmp_37[_pt_tmp_48, _pt_tmp_49]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_37, in_1=_pt_tmp_48, in_2=_pt_tmp_49
        )["out"]
    )
    _pt_tmp_36 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_36)
    del _pt_tmp_37
    _pt_tmp_36 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_36)
    _pt_tmp_35 = 0 + _pt_tmp_36
    _pt_tmp_34 = 0 + _pt_tmp_35
    del _pt_tmp_36
    _pt_tmp_33 = 0 + _pt_tmp_34
    del _pt_tmp_35
    _pt_tmp_32 = actx.np.reshape(_pt_tmp_33, (4, 48000, 15))
    del _pt_tmp_34
    _pt_tmp_32 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_32)
    del _pt_tmp_33
    _pt_tmp_29 = actx.einsum(
        "ijk, jl, jlk -> li", _pt_data_3, _pt_tmp_30, _pt_tmp_32
    )
    _pt_tmp_29 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_29)
    del _pt_tmp_32
    _pt_tmp_26 = _pt_tmp_27 - _pt_tmp_29
    _pt_tmp_25 = actx.einsum("i, jk, ik -> ij", _pt_tmp_5, _pt_data_0, _pt_tmp_26)
    del _pt_tmp_27, _pt_tmp_29
    _pt_tmp_25 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_25)
    del _pt_tmp_26
    _pt_tmp_24 = -1 * _pt_tmp_25
    _pt_tmp_51 = _actx_in_1_0_momentum_0_0 / _actx_in_1_0_mass_0
    del _pt_tmp_25
    _pt_tmp_56 = actx.einsum(
        "ijk, jlm, km -> ikl", _pt_tmp_9, _pt_data_2, _actx_in_1_0_mass_0
    )
    _pt_tmp_56 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_56)
    _pt_tmp_55 = _pt_tmp_56[0]
    _pt_tmp_69 = (
        _actx_in_1_0_mass_0[_pt_tmp_44, _pt_tmp_45]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0,
            in_0=_actx_in_1_0_mass_0,
            in_1=_pt_tmp_44,
            in_2=_pt_tmp_45,
        )["out"]
    )
    _pt_tmp_69 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_69)
    _pt_tmp_69 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_69)
    _pt_tmp_68 = 0 + _pt_tmp_69
    _pt_tmp_67 = (
        _pt_tmp_68[_pt_tmp_46, _pt_tmp_47]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_68, in_1=_pt_tmp_46, in_2=_pt_tmp_47
        )["out"]
    )
    del _pt_tmp_69
    _pt_tmp_67 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_67)
    _pt_tmp_67 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_67)
    _pt_tmp_66 = 0 + _pt_tmp_67
    _pt_tmp_65 = _pt_tmp_66 + _pt_tmp_68
    del _pt_tmp_67
    _pt_tmp_64 = _pt_tmp_65 / 2
    _pt_tmp_63 = (
        _pt_tmp_64 * _pt_data_11
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_64, _in1=_pt_data_11)["out"]
    )
    del _pt_tmp_65
    _pt_tmp_62 = (
        _pt_tmp_63[_pt_tmp_48, _pt_tmp_49]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_63, in_1=_pt_tmp_48, in_2=_pt_tmp_49
        )["out"]
    )
    _pt_tmp_62 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_62)
    del _pt_tmp_63
    _pt_tmp_62 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_62)
    _pt_tmp_61 = 0 + _pt_tmp_62
    _pt_tmp_60 = 0 + _pt_tmp_61
    del _pt_tmp_62
    _pt_tmp_59 = 0 + _pt_tmp_60
    del _pt_tmp_61
    _pt_tmp_58 = actx.np.reshape(_pt_tmp_59, (4, 48000, 15))
    del _pt_tmp_60
    _pt_tmp_58 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_58)
    del _pt_tmp_59
    _pt_tmp_57 = actx.einsum(
        "ijk, jl, jlk -> li", _pt_data_3, _pt_tmp_30, _pt_tmp_58
    )
    _pt_tmp_57 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_57)
    del _pt_tmp_58
    _pt_tmp_54 = _pt_tmp_55 - _pt_tmp_57
    _pt_tmp_53 = actx.einsum("i, jk, ik -> ij", _pt_tmp_5, _pt_data_0, _pt_tmp_54)
    del _pt_tmp_55, _pt_tmp_57
    _pt_tmp_53 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_53)
    del _pt_tmp_54
    _pt_tmp_52 = -1 * _pt_tmp_53
    _pt_tmp_50 = _pt_tmp_51 * _pt_tmp_52
    del _pt_tmp_53
    _pt_tmp_23 = _pt_tmp_24 - _pt_tmp_50
    _pt_tmp_22 = _pt_tmp_23 / _actx_in_1_0_mass_0
    del _pt_tmp_50
    _pt_tmp_21 = _pt_tmp_22 + _pt_tmp_22
    del _pt_tmp_23
    _pt_tmp_17 = _pt_tmp_18 * _pt_tmp_21
    _pt_tmp_73 = 0 * _pt_tmp_19
    del _pt_tmp_21
    _pt_tmp_75 = 2 * _pt_tmp_18
    _pt_tmp_74 = _pt_tmp_75 / 3
    _pt_tmp_72 = _pt_tmp_73 - _pt_tmp_74
    del _pt_tmp_75
    _actx_in_1_0_momentum_1_0 = actx.tag_axis(
        0, (DiscretizationElementAxisTag(),), _actx_in_1_0_momentum_1_0
    )
    del _pt_tmp_73, _pt_tmp_74
    _actx_in_1_0_momentum_1_0 = actx.tag_axis(
        1, (DiscretizationDOFAxisTag(),), _actx_in_1_0_momentum_1_0
    )
    _pt_tmp_84 = actx.einsum(
        "ijk, jlm, km -> ikl", _pt_tmp_9, _pt_data_2, _actx_in_1_0_momentum_1_0
    )
    _pt_tmp_84 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_84)
    _pt_tmp_83 = _pt_tmp_84[1]
    _pt_tmp_97 = (
        _actx_in_1_0_momentum_1_0[_pt_tmp_44, _pt_tmp_45]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0,
            in_0=_actx_in_1_0_momentum_1_0,
            in_1=_pt_tmp_44,
            in_2=_pt_tmp_45,
        )["out"]
    )
    _pt_tmp_97 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_97)
    _pt_tmp_97 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_97)
    _pt_tmp_96 = 0 + _pt_tmp_97
    _pt_tmp_95 = (
        _pt_tmp_96[_pt_tmp_46, _pt_tmp_47]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_96, in_1=_pt_tmp_46, in_2=_pt_tmp_47
        )["out"]
    )
    del _pt_tmp_97
    _pt_tmp_95 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_95)
    _pt_tmp_95 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_95)
    _pt_tmp_94 = 0 + _pt_tmp_95
    _pt_tmp_93 = _pt_tmp_94 + _pt_tmp_96
    del _pt_tmp_95
    _pt_tmp_92 = _pt_tmp_93 / 2
    _pt_data_15 = actx.thaw(npzfile["_pt_data_15"])
    del _pt_tmp_93
    _pt_data_15 = actx.tag(
        (PrefixNamed(prefix="normal_2_b_face_restr_interior"),), _pt_data_15
    )
    _pt_data_15 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_data_15)
    _pt_data_15 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_data_15)
    _pt_tmp_91 = (
        _pt_tmp_92 * _pt_data_15
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_92, _in1=_pt_data_15)["out"]
    )
    _pt_tmp_90 = (
        _pt_tmp_91[_pt_tmp_48, _pt_tmp_49]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_91, in_1=_pt_tmp_48, in_2=_pt_tmp_49
        )["out"]
    )
    _pt_tmp_90 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_90)
    del _pt_tmp_91
    _pt_tmp_90 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_90)
    _pt_tmp_89 = 0 + _pt_tmp_90
    _pt_tmp_88 = 0 + _pt_tmp_89
    del _pt_tmp_90
    _pt_tmp_87 = 0 + _pt_tmp_88
    del _pt_tmp_89
    _pt_tmp_86 = actx.np.reshape(_pt_tmp_87, (4, 48000, 15))
    del _pt_tmp_88
    _pt_tmp_86 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_86)
    del _pt_tmp_87
    _pt_tmp_85 = actx.einsum(
        "ijk, jl, jlk -> li", _pt_data_3, _pt_tmp_30, _pt_tmp_86
    )
    _pt_tmp_85 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_85)
    del _pt_tmp_86
    _pt_tmp_82 = _pt_tmp_83 - _pt_tmp_85
    _pt_tmp_81 = actx.einsum("i, jk, ik -> ij", _pt_tmp_5, _pt_data_0, _pt_tmp_82)
    del _pt_tmp_83, _pt_tmp_85
    _pt_tmp_81 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_81)
    del _pt_tmp_82
    _pt_tmp_80 = -1 * _pt_tmp_81
    _pt_tmp_99 = _actx_in_1_0_momentum_1_0 / _actx_in_1_0_mass_0
    del _pt_tmp_81
    _pt_tmp_103 = _pt_tmp_56[1]
    _pt_tmp_110 = (
        _pt_tmp_64 * _pt_data_15
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_64, _in1=_pt_data_15)["out"]
    )
    _pt_tmp_109 = (
        _pt_tmp_110[_pt_tmp_48, _pt_tmp_49]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_110, in_1=_pt_tmp_48, in_2=_pt_tmp_49
        )["out"]
    )
    _pt_tmp_109 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_109)
    del _pt_tmp_110
    _pt_tmp_109 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_109)
    _pt_tmp_108 = 0 + _pt_tmp_109
    _pt_tmp_107 = 0 + _pt_tmp_108
    del _pt_tmp_109
    _pt_tmp_106 = 0 + _pt_tmp_107
    del _pt_tmp_108
    _pt_tmp_105 = actx.np.reshape(_pt_tmp_106, (4, 48000, 15))
    del _pt_tmp_107
    _pt_tmp_105 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_105)
    del _pt_tmp_106
    _pt_tmp_104 = actx.einsum(
        "ijk, jl, jlk -> li", _pt_data_3, _pt_tmp_30, _pt_tmp_105
    )
    _pt_tmp_104 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_104)
    del _pt_tmp_105
    _pt_tmp_102 = _pt_tmp_103 - _pt_tmp_104
    _pt_tmp_101 = actx.einsum("i, jk, ik -> ij", _pt_tmp_5, _pt_data_0, _pt_tmp_102)
    del _pt_tmp_103, _pt_tmp_104
    _pt_tmp_101 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_101)
    del _pt_tmp_102
    _pt_tmp_100 = -1 * _pt_tmp_101
    _pt_tmp_98 = _pt_tmp_99 * _pt_tmp_100
    del _pt_tmp_101
    _pt_tmp_79 = _pt_tmp_80 - _pt_tmp_98
    _pt_tmp_78 = _pt_tmp_79 / _actx_in_1_0_mass_0
    del _pt_tmp_98
    _pt_tmp_77 = _pt_tmp_22 + _pt_tmp_78
    del _pt_tmp_79
    _actx_in_1_0_momentum_2_0 = actx.tag_axis(
        0, (DiscretizationElementAxisTag(),), _actx_in_1_0_momentum_2_0
    )
    del _pt_tmp_22
    _actx_in_1_0_momentum_2_0 = actx.tag_axis(
        1, (DiscretizationDOFAxisTag(),), _actx_in_1_0_momentum_2_0
    )
    _pt_tmp_117 = actx.einsum(
        "ijk, jlm, km -> ikl", _pt_tmp_9, _pt_data_2, _actx_in_1_0_momentum_2_0
    )
    _pt_tmp_117 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_117)
    _pt_tmp_116 = _pt_tmp_117[2]
    _pt_tmp_130 = (
        _actx_in_1_0_momentum_2_0[_pt_tmp_44, _pt_tmp_45]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0,
            in_0=_actx_in_1_0_momentum_2_0,
            in_1=_pt_tmp_44,
            in_2=_pt_tmp_45,
        )["out"]
    )
    _pt_tmp_130 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_130)
    _pt_tmp_130 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_130)
    _pt_tmp_129 = 0 + _pt_tmp_130
    _pt_tmp_128 = (
        _pt_tmp_129[_pt_tmp_46, _pt_tmp_47]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_129, in_1=_pt_tmp_46, in_2=_pt_tmp_47
        )["out"]
    )
    del _pt_tmp_130
    _pt_tmp_128 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_128)
    _pt_tmp_128 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_128)
    _pt_tmp_127 = 0 + _pt_tmp_128
    _pt_tmp_126 = _pt_tmp_127 + _pt_tmp_129
    del _pt_tmp_128
    _pt_tmp_125 = _pt_tmp_126 / 2
    _pt_data_16 = actx.thaw(npzfile["_pt_data_16"])
    del _pt_tmp_126
    _pt_data_16 = actx.tag(
        (PrefixNamed(prefix="normal_4_b_face_restr_interior"),), _pt_data_16
    )
    _pt_data_16 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_data_16)
    _pt_data_16 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_data_16)
    _pt_tmp_124 = (
        _pt_tmp_125 * _pt_data_16
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_125, _in1=_pt_data_16)["out"]
    )
    _pt_tmp_123 = (
        _pt_tmp_124[_pt_tmp_48, _pt_tmp_49]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_124, in_1=_pt_tmp_48, in_2=_pt_tmp_49
        )["out"]
    )
    _pt_tmp_123 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_123)
    del _pt_tmp_124
    _pt_tmp_123 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_123)
    _pt_tmp_122 = 0 + _pt_tmp_123
    _pt_tmp_121 = 0 + _pt_tmp_122
    del _pt_tmp_123
    _pt_tmp_120 = 0 + _pt_tmp_121
    del _pt_tmp_122
    _pt_tmp_119 = actx.np.reshape(_pt_tmp_120, (4, 48000, 15))
    del _pt_tmp_121
    _pt_tmp_119 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_119)
    del _pt_tmp_120
    _pt_tmp_118 = actx.einsum(
        "ijk, jl, jlk -> li", _pt_data_3, _pt_tmp_30, _pt_tmp_119
    )
    _pt_tmp_118 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_118)
    del _pt_tmp_119
    _pt_tmp_115 = _pt_tmp_116 - _pt_tmp_118
    _pt_tmp_114 = actx.einsum("i, jk, ik -> ij", _pt_tmp_5, _pt_data_0, _pt_tmp_115)
    del _pt_tmp_116, _pt_tmp_118
    _pt_tmp_114 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_114)
    del _pt_tmp_115
    _pt_tmp_113 = -1 * _pt_tmp_114
    _pt_tmp_132 = _actx_in_1_0_momentum_2_0 / _actx_in_1_0_mass_0
    del _pt_tmp_114
    _pt_tmp_136 = _pt_tmp_56[2]
    _pt_tmp_143 = (
        _pt_tmp_64 * _pt_data_16
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_64, _in1=_pt_data_16)["out"]
    )
    del _pt_tmp_56
    _pt_tmp_142 = (
        _pt_tmp_143[_pt_tmp_48, _pt_tmp_49]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_143, in_1=_pt_tmp_48, in_2=_pt_tmp_49
        )["out"]
    )
    del _pt_tmp_64
    _pt_tmp_142 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_142)
    del _pt_tmp_143
    _pt_tmp_142 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_142)
    _pt_tmp_141 = 0 + _pt_tmp_142
    _pt_tmp_140 = 0 + _pt_tmp_141
    del _pt_tmp_142
    _pt_tmp_139 = 0 + _pt_tmp_140
    del _pt_tmp_141
    _pt_tmp_138 = actx.np.reshape(_pt_tmp_139, (4, 48000, 15))
    del _pt_tmp_140
    _pt_tmp_138 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_138)
    del _pt_tmp_139
    _pt_tmp_137 = actx.einsum(
        "ijk, jl, jlk -> li", _pt_data_3, _pt_tmp_30, _pt_tmp_138
    )
    _pt_tmp_137 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_137)
    del _pt_tmp_138
    _pt_tmp_135 = _pt_tmp_136 - _pt_tmp_137
    _pt_tmp_134 = actx.einsum("i, jk, ik -> ij", _pt_tmp_5, _pt_data_0, _pt_tmp_135)
    del _pt_tmp_136, _pt_tmp_137
    _pt_tmp_134 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_134)
    del _pt_tmp_135
    _pt_tmp_133 = -1 * _pt_tmp_134
    _pt_tmp_131 = _pt_tmp_132 * _pt_tmp_133
    del _pt_tmp_134
    _pt_tmp_112 = _pt_tmp_113 - _pt_tmp_131
    _pt_tmp_111 = _pt_tmp_112 / _actx_in_1_0_mass_0
    del _pt_tmp_131
    _pt_tmp_76 = _pt_tmp_77 + _pt_tmp_111
    del _pt_tmp_112
    _pt_tmp_71 = _pt_tmp_72 * _pt_tmp_76
    del _pt_tmp_77
    _pt_tmp_70 = _pt_tmp_71 * 1.0
    del _pt_tmp_72, _pt_tmp_76
    _pt_tmp_16 = _pt_tmp_17 + _pt_tmp_70
    _pt_tmp_15 = _pt_tmp_16 * _pt_tmp_51
    del _pt_tmp_17
    _pt_tmp_153 = _pt_tmp_28[1]
    _pt_tmp_160 = (
        _pt_tmp_38 * _pt_data_15
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_38, _in1=_pt_data_15)["out"]
    )
    _pt_tmp_159 = (
        _pt_tmp_160[_pt_tmp_48, _pt_tmp_49]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_160, in_1=_pt_tmp_48, in_2=_pt_tmp_49
        )["out"]
    )
    _pt_tmp_159 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_159)
    del _pt_tmp_160
    _pt_tmp_159 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_159)
    _pt_tmp_158 = 0 + _pt_tmp_159
    _pt_tmp_157 = 0 + _pt_tmp_158
    del _pt_tmp_159
    _pt_tmp_156 = 0 + _pt_tmp_157
    del _pt_tmp_158
    _pt_tmp_155 = actx.np.reshape(_pt_tmp_156, (4, 48000, 15))
    del _pt_tmp_157
    _pt_tmp_155 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_155)
    del _pt_tmp_156
    _pt_tmp_154 = actx.einsum(
        "ijk, jl, jlk -> li", _pt_data_3, _pt_tmp_30, _pt_tmp_155
    )
    _pt_tmp_154 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_154)
    del _pt_tmp_155
    _pt_tmp_152 = _pt_tmp_153 - _pt_tmp_154
    _pt_tmp_151 = actx.einsum("i, jk, ik -> ij", _pt_tmp_5, _pt_data_0, _pt_tmp_152)
    del _pt_tmp_153, _pt_tmp_154
    _pt_tmp_151 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_151)
    del _pt_tmp_152
    _pt_tmp_150 = -1 * _pt_tmp_151
    _pt_tmp_161 = _pt_tmp_51 * _pt_tmp_100
    del _pt_tmp_151
    _pt_tmp_149 = _pt_tmp_150 - _pt_tmp_161
    _pt_tmp_148 = _pt_tmp_149 / _actx_in_1_0_mass_0
    del _pt_tmp_161
    _pt_tmp_167 = _pt_tmp_84[0]
    del _pt_tmp_149
    _pt_tmp_174 = (
        _pt_tmp_92 * _pt_data_11
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_92, _in1=_pt_data_11)["out"]
    )
    _pt_tmp_173 = (
        _pt_tmp_174[_pt_tmp_48, _pt_tmp_49]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_174, in_1=_pt_tmp_48, in_2=_pt_tmp_49
        )["out"]
    )
    _pt_tmp_173 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_173)
    del _pt_tmp_174
    _pt_tmp_173 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_173)
    _pt_tmp_172 = 0 + _pt_tmp_173
    _pt_tmp_171 = 0 + _pt_tmp_172
    del _pt_tmp_173
    _pt_tmp_170 = 0 + _pt_tmp_171
    del _pt_tmp_172
    _pt_tmp_169 = actx.np.reshape(_pt_tmp_170, (4, 48000, 15))
    del _pt_tmp_171
    _pt_tmp_169 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_169)
    del _pt_tmp_170
    _pt_tmp_168 = actx.einsum(
        "ijk, jl, jlk -> li", _pt_data_3, _pt_tmp_30, _pt_tmp_169
    )
    _pt_tmp_168 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_168)
    del _pt_tmp_169
    _pt_tmp_166 = _pt_tmp_167 - _pt_tmp_168
    _pt_tmp_165 = actx.einsum("i, jk, ik -> ij", _pt_tmp_5, _pt_data_0, _pt_tmp_166)
    del _pt_tmp_167, _pt_tmp_168
    _pt_tmp_165 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_165)
    del _pt_tmp_166
    _pt_tmp_164 = -1 * _pt_tmp_165
    _pt_tmp_175 = _pt_tmp_99 * _pt_tmp_52
    del _pt_tmp_165
    _pt_tmp_163 = _pt_tmp_164 - _pt_tmp_175
    _pt_tmp_162 = _pt_tmp_163 / _actx_in_1_0_mass_0
    del _pt_tmp_175
    _pt_tmp_147 = _pt_tmp_148 + _pt_tmp_162
    del _pt_tmp_163
    _pt_tmp_146 = _pt_tmp_18 * _pt_tmp_147
    _pt_tmp_176 = _pt_tmp_71 * 0.0
    del _pt_tmp_147
    _pt_tmp_145 = _pt_tmp_146 + _pt_tmp_176
    del _pt_tmp_71
    _pt_tmp_144 = _pt_tmp_145 * _pt_tmp_99
    del _pt_tmp_146
    _pt_tmp_14 = _pt_tmp_15 + _pt_tmp_144
    _pt_tmp_186 = _pt_tmp_28[2]
    del _pt_tmp_144, _pt_tmp_15
    _pt_tmp_193 = (
        _pt_tmp_38 * _pt_data_16
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_38, _in1=_pt_data_16)["out"]
    )
    del _pt_tmp_28
    _pt_tmp_192 = (
        _pt_tmp_193[_pt_tmp_48, _pt_tmp_49]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_193, in_1=_pt_tmp_48, in_2=_pt_tmp_49
        )["out"]
    )
    del _pt_tmp_38
    _pt_tmp_192 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_192)
    del _pt_tmp_193
    _pt_tmp_192 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_192)
    _pt_tmp_191 = 0 + _pt_tmp_192
    _pt_tmp_190 = 0 + _pt_tmp_191
    del _pt_tmp_192
    _pt_tmp_189 = 0 + _pt_tmp_190
    del _pt_tmp_191
    _pt_tmp_188 = actx.np.reshape(_pt_tmp_189, (4, 48000, 15))
    del _pt_tmp_190
    _pt_tmp_188 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_188)
    del _pt_tmp_189
    _pt_tmp_187 = actx.einsum(
        "ijk, jl, jlk -> li", _pt_data_3, _pt_tmp_30, _pt_tmp_188
    )
    _pt_tmp_187 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_187)
    del _pt_tmp_188
    _pt_tmp_185 = _pt_tmp_186 - _pt_tmp_187
    _pt_tmp_184 = actx.einsum("i, jk, ik -> ij", _pt_tmp_5, _pt_data_0, _pt_tmp_185)
    del _pt_tmp_186, _pt_tmp_187
    _pt_tmp_184 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_184)
    del _pt_tmp_185
    _pt_tmp_183 = -1 * _pt_tmp_184
    _pt_tmp_194 = _pt_tmp_51 * _pt_tmp_133
    del _pt_tmp_184
    _pt_tmp_182 = _pt_tmp_183 - _pt_tmp_194
    _pt_tmp_181 = _pt_tmp_182 / _actx_in_1_0_mass_0
    del _pt_tmp_194
    _pt_tmp_200 = _pt_tmp_117[0]
    del _pt_tmp_182
    _pt_tmp_207 = (
        _pt_tmp_125 * _pt_data_11
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_125, _in1=_pt_data_11)["out"]
    )
    _pt_tmp_206 = (
        _pt_tmp_207[_pt_tmp_48, _pt_tmp_49]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_207, in_1=_pt_tmp_48, in_2=_pt_tmp_49
        )["out"]
    )
    _pt_tmp_206 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_206)
    del _pt_tmp_207
    _pt_tmp_206 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_206)
    _pt_tmp_205 = 0 + _pt_tmp_206
    _pt_tmp_204 = 0 + _pt_tmp_205
    del _pt_tmp_206
    _pt_tmp_203 = 0 + _pt_tmp_204
    del _pt_tmp_205
    _pt_tmp_202 = actx.np.reshape(_pt_tmp_203, (4, 48000, 15))
    del _pt_tmp_204
    _pt_tmp_202 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_202)
    del _pt_tmp_203
    _pt_tmp_201 = actx.einsum(
        "ijk, jl, jlk -> li", _pt_data_3, _pt_tmp_30, _pt_tmp_202
    )
    _pt_tmp_201 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_201)
    del _pt_tmp_202
    _pt_tmp_199 = _pt_tmp_200 - _pt_tmp_201
    _pt_tmp_198 = actx.einsum("i, jk, ik -> ij", _pt_tmp_5, _pt_data_0, _pt_tmp_199)
    del _pt_tmp_200, _pt_tmp_201
    _pt_tmp_198 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_198)
    del _pt_tmp_199
    _pt_tmp_197 = -1 * _pt_tmp_198
    _pt_tmp_208 = _pt_tmp_132 * _pt_tmp_52
    del _pt_tmp_198
    _pt_tmp_196 = _pt_tmp_197 - _pt_tmp_208
    _pt_tmp_195 = _pt_tmp_196 / _actx_in_1_0_mass_0
    del _pt_tmp_208
    _pt_tmp_180 = _pt_tmp_181 + _pt_tmp_195
    del _pt_tmp_196
    _pt_tmp_179 = _pt_tmp_18 * _pt_tmp_180
    _pt_tmp_178 = _pt_tmp_179 + _pt_tmp_176
    del _pt_tmp_180
    _pt_tmp_177 = _pt_tmp_178 * _pt_tmp_132
    del _pt_tmp_179
    _pt_tmp_13 = _pt_tmp_14 + _pt_tmp_177
    _pt_tmp_12 = _pt_tmp_13 + 0
    del _pt_tmp_14, _pt_tmp_177
    _pt_tmp_211 = 10 * _pt_tmp_19
    del _pt_tmp_13
    _pt_tmp_210 = -1 * _pt_tmp_211
    del _pt_tmp_19
    _actx_in_1_0_energy_0 = actx.tag_axis(
        0, (DiscretizationElementAxisTag(),), _actx_in_1_0_energy_0
    )
    del _pt_tmp_211
    _actx_in_1_0_energy_0 = actx.tag_axis(
        1, (DiscretizationDOFAxisTag(),), _actx_in_1_0_energy_0
    )
    _pt_tmp_224 = _actx_in_1_0_momentum_0_0 * _actx_in_1_0_momentum_0_0
    _pt_tmp_225 = _actx_in_1_0_momentum_1_0 * _actx_in_1_0_momentum_1_0
    _pt_tmp_223 = _pt_tmp_224 + _pt_tmp_225
    _pt_tmp_226 = _actx_in_1_0_momentum_2_0 * _actx_in_1_0_momentum_2_0
    del _pt_tmp_224, _pt_tmp_225
    _pt_tmp_222 = _pt_tmp_223 + _pt_tmp_226
    _pt_tmp_221 = 0.5 * _pt_tmp_222
    del _pt_tmp_223, _pt_tmp_226
    _pt_tmp_220 = _pt_tmp_221 / _actx_in_1_0_mass_0
    del _pt_tmp_222
    _pt_tmp_219 = _actx_in_1_0_energy_0 - _pt_tmp_220
    del _pt_tmp_221
    _pt_tmp_218 = 0.001393242772553117 * _pt_tmp_219
    del _pt_tmp_220
    _pt_tmp_217 = _pt_tmp_218 / _actx_in_1_0_mass_0
    _pt_tmp_216 = actx.einsum(
        "ijk, jlm, km -> ikl", _pt_tmp_9, _pt_data_2, _pt_tmp_217
    )
    del _pt_tmp_218
    _pt_tmp_216 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_216)
    _pt_tmp_215 = _pt_tmp_216[0]
    _pt_tmp_242 = (
        _actx_in_1_0_energy_0[_pt_tmp_44, _pt_tmp_45]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0,
            in_0=_actx_in_1_0_energy_0,
            in_1=_pt_tmp_44,
            in_2=_pt_tmp_45,
        )["out"]
    )
    _pt_tmp_242 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_242)
    _pt_tmp_242 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_242)
    _pt_tmp_241 = 0 + _pt_tmp_242
    _pt_tmp_240 = (
        _pt_tmp_241[_pt_tmp_46, _pt_tmp_47]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_241, in_1=_pt_tmp_46, in_2=_pt_tmp_47
        )["out"]
    )
    del _pt_tmp_242
    _pt_tmp_240 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_240)
    _pt_tmp_240 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_240)
    _pt_tmp_239 = 0 + _pt_tmp_240
    _pt_tmp_247 = _pt_tmp_40 * _pt_tmp_40
    del _pt_tmp_240
    _pt_tmp_248 = _pt_tmp_94 * _pt_tmp_94
    _pt_tmp_246 = _pt_tmp_247 + _pt_tmp_248
    _pt_tmp_249 = _pt_tmp_127 * _pt_tmp_127
    del _pt_tmp_247, _pt_tmp_248
    _pt_tmp_245 = _pt_tmp_246 + _pt_tmp_249
    _pt_tmp_244 = 0.5 * _pt_tmp_245
    del _pt_tmp_246, _pt_tmp_249
    _pt_tmp_243 = _pt_tmp_244 / _pt_tmp_66
    del _pt_tmp_245
    _pt_tmp_238 = _pt_tmp_239 - _pt_tmp_243
    del _pt_tmp_244
    _pt_tmp_237 = 0.001393242772553117 * _pt_tmp_238
    del _pt_tmp_243
    _pt_tmp_236 = _pt_tmp_237 / _pt_tmp_66
    _pt_tmp_257 = _pt_tmp_42 * _pt_tmp_42
    del _pt_tmp_237
    _pt_tmp_258 = _pt_tmp_96 * _pt_tmp_96
    _pt_tmp_256 = _pt_tmp_257 + _pt_tmp_258
    _pt_tmp_259 = _pt_tmp_129 * _pt_tmp_129
    del _pt_tmp_257, _pt_tmp_258
    _pt_tmp_255 = _pt_tmp_256 + _pt_tmp_259
    _pt_tmp_254 = 0.5 * _pt_tmp_255
    del _pt_tmp_256, _pt_tmp_259
    _pt_tmp_253 = _pt_tmp_254 / _pt_tmp_68
    del _pt_tmp_255
    _pt_tmp_252 = _pt_tmp_241 - _pt_tmp_253
    del _pt_tmp_254
    _pt_tmp_251 = 0.001393242772553117 * _pt_tmp_252
    del _pt_tmp_253
    _pt_tmp_250 = _pt_tmp_251 / _pt_tmp_68
    _pt_tmp_235 = _pt_tmp_236 + _pt_tmp_250
    del _pt_tmp_251
    _pt_tmp_234 = _pt_tmp_235 / 2
    del _pt_tmp_236, _pt_tmp_250
    _pt_tmp_233 = (
        _pt_tmp_234 * _pt_data_11
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_234, _in1=_pt_data_11)["out"]
    )
    del _pt_tmp_235
    _pt_tmp_232 = (
        _pt_tmp_233[_pt_tmp_48, _pt_tmp_49]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_233, in_1=_pt_tmp_48, in_2=_pt_tmp_49
        )["out"]
    )
    _pt_tmp_232 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_232)
    del _pt_tmp_233
    _pt_tmp_232 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_232)
    _pt_tmp_231 = 0 + _pt_tmp_232
    _pt_tmp_230 = 0 + _pt_tmp_231
    del _pt_tmp_232
    _pt_tmp_229 = 0 + _pt_tmp_230
    del _pt_tmp_231
    _pt_tmp_228 = actx.np.reshape(_pt_tmp_229, (4, 48000, 15))
    del _pt_tmp_230
    _pt_tmp_228 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_228)
    del _pt_tmp_229
    _pt_tmp_227 = actx.einsum(
        "ijk, jl, jlk -> li", _pt_data_3, _pt_tmp_30, _pt_tmp_228
    )
    _pt_tmp_227 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_227)
    del _pt_tmp_228
    _pt_tmp_214 = _pt_tmp_215 - _pt_tmp_227
    _pt_tmp_213 = actx.einsum("i, jk, ik -> ij", _pt_tmp_5, _pt_data_0, _pt_tmp_214)
    del _pt_tmp_215, _pt_tmp_227
    _pt_tmp_213 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_213)
    del _pt_tmp_214
    _pt_tmp_212 = -1 * _pt_tmp_213
    _pt_tmp_209 = _pt_tmp_210 * _pt_tmp_212
    del _pt_tmp_213
    _pt_tmp_11 = _pt_tmp_12 - _pt_tmp_209
    _pt_tmp_262 = _pt_tmp_219 * 0.3999999999999999
    del _pt_tmp_12, _pt_tmp_209
    _pt_tmp_261 = _actx_in_1_0_energy_0 + _pt_tmp_262
    del _pt_tmp_219
    _pt_tmp_260 = _pt_tmp_51 * _pt_tmp_261
    _pt_tmp_10 = _pt_tmp_11 - _pt_tmp_260
    _pt_tmp_271 = _pt_tmp_162 + _pt_tmp_148
    del _pt_tmp_11, _pt_tmp_260
    _pt_tmp_270 = _pt_tmp_18 * _pt_tmp_271
    del _pt_tmp_148, _pt_tmp_162
    _pt_tmp_269 = _pt_tmp_270 + _pt_tmp_176
    del _pt_tmp_271
    _pt_tmp_268 = _pt_tmp_269 * _pt_tmp_51
    del _pt_tmp_270
    _pt_tmp_275 = _pt_tmp_78 + _pt_tmp_78
    _pt_tmp_274 = _pt_tmp_18 * _pt_tmp_275
    del _pt_tmp_78
    _pt_tmp_273 = _pt_tmp_274 + _pt_tmp_70
    del _pt_tmp_275
    _pt_tmp_272 = _pt_tmp_273 * _pt_tmp_99
    del _pt_tmp_274
    _pt_tmp_267 = _pt_tmp_268 + _pt_tmp_272
    _pt_tmp_285 = _pt_tmp_84[2]
    del _pt_tmp_268, _pt_tmp_272
    _pt_tmp_292 = (
        _pt_tmp_92 * _pt_data_16
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_92, _in1=_pt_data_16)["out"]
    )
    del _pt_tmp_84
    _pt_tmp_291 = (
        _pt_tmp_292[_pt_tmp_48, _pt_tmp_49]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_292, in_1=_pt_tmp_48, in_2=_pt_tmp_49
        )["out"]
    )
    del _pt_tmp_92
    _pt_tmp_291 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_291)
    del _pt_tmp_292
    _pt_tmp_291 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_291)
    _pt_tmp_290 = 0 + _pt_tmp_291
    _pt_tmp_289 = 0 + _pt_tmp_290
    del _pt_tmp_291
    _pt_tmp_288 = 0 + _pt_tmp_289
    del _pt_tmp_290
    _pt_tmp_287 = actx.np.reshape(_pt_tmp_288, (4, 48000, 15))
    del _pt_tmp_289
    _pt_tmp_287 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_287)
    del _pt_tmp_288
    _pt_tmp_286 = actx.einsum(
        "ijk, jl, jlk -> li", _pt_data_3, _pt_tmp_30, _pt_tmp_287
    )
    _pt_tmp_286 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_286)
    del _pt_tmp_287
    _pt_tmp_284 = _pt_tmp_285 - _pt_tmp_286
    _pt_tmp_283 = actx.einsum("i, jk, ik -> ij", _pt_tmp_5, _pt_data_0, _pt_tmp_284)
    del _pt_tmp_285, _pt_tmp_286
    _pt_tmp_283 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_283)
    del _pt_tmp_284
    _pt_tmp_282 = -1 * _pt_tmp_283
    _pt_tmp_293 = _pt_tmp_99 * _pt_tmp_133
    del _pt_tmp_283
    _pt_tmp_281 = _pt_tmp_282 - _pt_tmp_293
    _pt_tmp_280 = _pt_tmp_281 / _actx_in_1_0_mass_0
    del _pt_tmp_293
    _pt_tmp_299 = _pt_tmp_117[1]
    del _pt_tmp_281
    _pt_tmp_306 = (
        _pt_tmp_125 * _pt_data_15
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_125, _in1=_pt_data_15)["out"]
    )
    del _pt_tmp_117
    _pt_tmp_305 = (
        _pt_tmp_306[_pt_tmp_48, _pt_tmp_49]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_306, in_1=_pt_tmp_48, in_2=_pt_tmp_49
        )["out"]
    )
    del _pt_tmp_125
    _pt_tmp_305 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_305)
    del _pt_tmp_306
    _pt_tmp_305 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_305)
    _pt_tmp_304 = 0 + _pt_tmp_305
    _pt_tmp_303 = 0 + _pt_tmp_304
    del _pt_tmp_305
    _pt_tmp_302 = 0 + _pt_tmp_303
    del _pt_tmp_304
    _pt_tmp_301 = actx.np.reshape(_pt_tmp_302, (4, 48000, 15))
    del _pt_tmp_303
    _pt_tmp_301 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_301)
    del _pt_tmp_302
    _pt_tmp_300 = actx.einsum(
        "ijk, jl, jlk -> li", _pt_data_3, _pt_tmp_30, _pt_tmp_301
    )
    _pt_tmp_300 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_300)
    del _pt_tmp_301
    _pt_tmp_298 = _pt_tmp_299 - _pt_tmp_300
    _pt_tmp_297 = actx.einsum("i, jk, ik -> ij", _pt_tmp_5, _pt_data_0, _pt_tmp_298)
    del _pt_tmp_299, _pt_tmp_300
    _pt_tmp_297 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_297)
    del _pt_tmp_298
    _pt_tmp_296 = -1 * _pt_tmp_297
    _pt_tmp_307 = _pt_tmp_132 * _pt_tmp_100
    del _pt_tmp_297
    _pt_tmp_295 = _pt_tmp_296 - _pt_tmp_307
    _pt_tmp_294 = _pt_tmp_295 / _actx_in_1_0_mass_0
    del _pt_tmp_307
    _pt_tmp_279 = _pt_tmp_280 + _pt_tmp_294
    del _pt_tmp_295
    _pt_tmp_278 = _pt_tmp_18 * _pt_tmp_279
    _pt_tmp_277 = _pt_tmp_278 + _pt_tmp_176
    del _pt_tmp_279
    _pt_tmp_276 = _pt_tmp_277 * _pt_tmp_132
    del _pt_tmp_278
    _pt_tmp_266 = _pt_tmp_267 + _pt_tmp_276
    _pt_tmp_265 = _pt_tmp_266 + 0
    del _pt_tmp_267, _pt_tmp_276
    _pt_tmp_312 = _pt_tmp_216[1]
    del _pt_tmp_266
    _pt_tmp_319 = (
        _pt_tmp_234 * _pt_data_15
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_234, _in1=_pt_data_15)["out"]
    )
    _pt_tmp_318 = (
        _pt_tmp_319[_pt_tmp_48, _pt_tmp_49]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_319, in_1=_pt_tmp_48, in_2=_pt_tmp_49
        )["out"]
    )
    _pt_tmp_318 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_318)
    del _pt_tmp_319
    _pt_tmp_318 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_318)
    _pt_tmp_317 = 0 + _pt_tmp_318
    _pt_tmp_316 = 0 + _pt_tmp_317
    del _pt_tmp_318
    _pt_tmp_315 = 0 + _pt_tmp_316
    del _pt_tmp_317
    _pt_tmp_314 = actx.np.reshape(_pt_tmp_315, (4, 48000, 15))
    del _pt_tmp_316
    _pt_tmp_314 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_314)
    del _pt_tmp_315
    _pt_tmp_313 = actx.einsum(
        "ijk, jl, jlk -> li", _pt_data_3, _pt_tmp_30, _pt_tmp_314
    )
    _pt_tmp_313 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_313)
    del _pt_tmp_314
    _pt_tmp_311 = _pt_tmp_312 - _pt_tmp_313
    _pt_tmp_310 = actx.einsum("i, jk, ik -> ij", _pt_tmp_5, _pt_data_0, _pt_tmp_311)
    del _pt_tmp_312, _pt_tmp_313
    _pt_tmp_310 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_310)
    del _pt_tmp_311
    _pt_tmp_309 = -1 * _pt_tmp_310
    _pt_tmp_308 = _pt_tmp_210 * _pt_tmp_309
    del _pt_tmp_310
    _pt_tmp_264 = _pt_tmp_265 - _pt_tmp_308
    _pt_tmp_320 = _pt_tmp_99 * _pt_tmp_261
    del _pt_tmp_265, _pt_tmp_308
    _pt_tmp_263 = _pt_tmp_264 - _pt_tmp_320
    _pt_tmp_329 = _pt_tmp_195 + _pt_tmp_181
    del _pt_tmp_264, _pt_tmp_320
    _pt_tmp_328 = _pt_tmp_18 * _pt_tmp_329
    del _pt_tmp_181, _pt_tmp_195
    _pt_tmp_327 = _pt_tmp_328 + _pt_tmp_176
    del _pt_tmp_329
    _pt_tmp_326 = _pt_tmp_327 * _pt_tmp_51
    del _pt_tmp_328
    _pt_tmp_333 = _pt_tmp_294 + _pt_tmp_280
    _pt_tmp_332 = _pt_tmp_18 * _pt_tmp_333
    del _pt_tmp_280, _pt_tmp_294
    _pt_tmp_331 = _pt_tmp_332 + _pt_tmp_176
    del _pt_tmp_333
    _pt_tmp_330 = _pt_tmp_331 * _pt_tmp_99
    del _pt_tmp_176, _pt_tmp_332
    _pt_tmp_325 = _pt_tmp_326 + _pt_tmp_330
    _pt_tmp_337 = _pt_tmp_111 + _pt_tmp_111
    del _pt_tmp_326, _pt_tmp_330
    _pt_tmp_336 = _pt_tmp_18 * _pt_tmp_337
    del _pt_tmp_111
    _pt_tmp_335 = _pt_tmp_336 + _pt_tmp_70
    del _pt_tmp_18, _pt_tmp_337
    _pt_tmp_334 = _pt_tmp_335 * _pt_tmp_132
    del _pt_tmp_336, _pt_tmp_70
    _pt_tmp_324 = _pt_tmp_325 + _pt_tmp_334
    _pt_tmp_323 = _pt_tmp_324 + 0
    del _pt_tmp_325, _pt_tmp_334
    _pt_tmp_342 = _pt_tmp_216[2]
    del _pt_tmp_324
    _pt_tmp_349 = (
        _pt_tmp_234 * _pt_data_16
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_234, _in1=_pt_data_16)["out"]
    )
    del _pt_tmp_216
    _pt_tmp_348 = (
        _pt_tmp_349[_pt_tmp_48, _pt_tmp_49]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_349, in_1=_pt_tmp_48, in_2=_pt_tmp_49
        )["out"]
    )
    del _pt_tmp_234
    _pt_tmp_348 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_348)
    del _pt_tmp_349
    _pt_tmp_348 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_348)
    _pt_tmp_347 = 0 + _pt_tmp_348
    _pt_tmp_346 = 0 + _pt_tmp_347
    del _pt_tmp_348
    _pt_tmp_345 = 0 + _pt_tmp_346
    del _pt_tmp_347
    _pt_tmp_344 = actx.np.reshape(_pt_tmp_345, (4, 48000, 15))
    del _pt_tmp_346
    _pt_tmp_344 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_344)
    del _pt_tmp_345
    _pt_tmp_343 = actx.einsum(
        "ijk, jl, jlk -> li", _pt_data_3, _pt_tmp_30, _pt_tmp_344
    )
    _pt_tmp_343 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_343)
    del _pt_tmp_344
    _pt_tmp_341 = _pt_tmp_342 - _pt_tmp_343
    _pt_tmp_340 = actx.einsum("i, jk, ik -> ij", _pt_tmp_5, _pt_data_0, _pt_tmp_341)
    del _pt_tmp_342, _pt_tmp_343
    _pt_tmp_340 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_340)
    del _pt_tmp_341
    _pt_tmp_339 = -1 * _pt_tmp_340
    _pt_tmp_338 = _pt_tmp_210 * _pt_tmp_339
    del _pt_tmp_340
    _pt_tmp_322 = _pt_tmp_323 - _pt_tmp_338
    del _pt_tmp_210
    _pt_tmp_350 = _pt_tmp_132 * _pt_tmp_261
    del _pt_tmp_323, _pt_tmp_338
    _pt_tmp_321 = _pt_tmp_322 - _pt_tmp_350
    del _pt_tmp_261
    _pt_tmp_351 = actx.np.stack([_pt_tmp_10, _pt_tmp_263, _pt_tmp_321], axis=0)
    del _pt_tmp_322, _pt_tmp_350
    _pt_tmp_8 = actx.einsum(
        "ijk, jlm, ikm -> kl", _pt_tmp_9, _pt_data_2, _pt_tmp_351
    )
    del _pt_tmp_10, _pt_tmp_263, _pt_tmp_321
    _pt_tmp_8 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_8)
    del _pt_tmp_351
    _pt_tmp_373 = 0 * _pt_tmp_66
    _pt_tmp_372 = _pt_tmp_373 + 1.0
    _pt_tmp_371 = 1e-05 * _pt_tmp_372
    del _pt_tmp_373
    _pt_tmp_380 = (
        _pt_tmp_24[_pt_tmp_44, _pt_tmp_45]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_pt_tmp_24, in_1=_pt_tmp_44, in_2=_pt_tmp_45
        )["out"]
    )
    _pt_tmp_380 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_380)
    _pt_tmp_380 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_380)
    _pt_tmp_379 = 0 + _pt_tmp_380
    _pt_tmp_378 = (
        _pt_tmp_379[_pt_tmp_46, _pt_tmp_47]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_379, in_1=_pt_tmp_46, in_2=_pt_tmp_47
        )["out"]
    )
    del _pt_tmp_380
    _pt_tmp_378 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_378)
    _pt_tmp_378 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_378)
    _pt_tmp_377 = 0 + _pt_tmp_378
    _pt_tmp_382 = _pt_tmp_40 / _pt_tmp_66
    del _pt_tmp_378
    _pt_tmp_386 = (
        _pt_tmp_52[_pt_tmp_44, _pt_tmp_45]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_pt_tmp_52, in_1=_pt_tmp_44, in_2=_pt_tmp_45
        )["out"]
    )
    _pt_tmp_386 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_386)
    _pt_tmp_386 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_386)
    _pt_tmp_385 = 0 + _pt_tmp_386
    _pt_tmp_384 = (
        _pt_tmp_385[_pt_tmp_46, _pt_tmp_47]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_385, in_1=_pt_tmp_46, in_2=_pt_tmp_47
        )["out"]
    )
    del _pt_tmp_386
    _pt_tmp_384 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_384)
    _pt_tmp_384 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_384)
    _pt_tmp_383 = 0 + _pt_tmp_384
    _pt_tmp_381 = _pt_tmp_382 * _pt_tmp_383
    del _pt_tmp_384
    _pt_tmp_376 = _pt_tmp_377 - _pt_tmp_381
    _pt_tmp_375 = _pt_tmp_376 / _pt_tmp_66
    del _pt_tmp_377, _pt_tmp_381
    _pt_tmp_374 = _pt_tmp_375 + _pt_tmp_375
    del _pt_tmp_376
    _pt_tmp_370 = _pt_tmp_371 * _pt_tmp_374
    _pt_tmp_390 = 0 * _pt_tmp_372
    del _pt_tmp_374
    _pt_tmp_392 = 2 * _pt_tmp_371
    _pt_tmp_391 = _pt_tmp_392 / 3
    _pt_tmp_389 = _pt_tmp_390 - _pt_tmp_391
    del _pt_tmp_392
    _pt_tmp_400 = (
        _pt_tmp_80[_pt_tmp_44, _pt_tmp_45]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_pt_tmp_80, in_1=_pt_tmp_44, in_2=_pt_tmp_45
        )["out"]
    )
    del _pt_tmp_390, _pt_tmp_391
    _pt_tmp_400 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_400)
    _pt_tmp_400 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_400)
    _pt_tmp_399 = 0 + _pt_tmp_400
    _pt_tmp_398 = (
        _pt_tmp_399[_pt_tmp_46, _pt_tmp_47]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_399, in_1=_pt_tmp_46, in_2=_pt_tmp_47
        )["out"]
    )
    del _pt_tmp_400
    _pt_tmp_398 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_398)
    _pt_tmp_398 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_398)
    _pt_tmp_397 = 0 + _pt_tmp_398
    _pt_tmp_402 = _pt_tmp_94 / _pt_tmp_66
    del _pt_tmp_398
    _pt_tmp_406 = (
        _pt_tmp_100[_pt_tmp_44, _pt_tmp_45]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_pt_tmp_100, in_1=_pt_tmp_44, in_2=_pt_tmp_45
        )["out"]
    )
    _pt_tmp_406 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_406)
    _pt_tmp_406 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_406)
    _pt_tmp_405 = 0 + _pt_tmp_406
    _pt_tmp_404 = (
        _pt_tmp_405[_pt_tmp_46, _pt_tmp_47]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_405, in_1=_pt_tmp_46, in_2=_pt_tmp_47
        )["out"]
    )
    del _pt_tmp_406
    _pt_tmp_404 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_404)
    _pt_tmp_404 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_404)
    _pt_tmp_403 = 0 + _pt_tmp_404
    _pt_tmp_401 = _pt_tmp_402 * _pt_tmp_403
    del _pt_tmp_404
    _pt_tmp_396 = _pt_tmp_397 - _pt_tmp_401
    _pt_tmp_395 = _pt_tmp_396 / _pt_tmp_66
    del _pt_tmp_397, _pt_tmp_401
    _pt_tmp_394 = _pt_tmp_375 + _pt_tmp_395
    del _pt_tmp_396
    _pt_tmp_412 = (
        _pt_tmp_113[_pt_tmp_44, _pt_tmp_45]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_pt_tmp_113, in_1=_pt_tmp_44, in_2=_pt_tmp_45
        )["out"]
    )
    del _pt_tmp_375
    _pt_tmp_412 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_412)
    _pt_tmp_412 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_412)
    _pt_tmp_411 = 0 + _pt_tmp_412
    _pt_tmp_410 = (
        _pt_tmp_411[_pt_tmp_46, _pt_tmp_47]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_411, in_1=_pt_tmp_46, in_2=_pt_tmp_47
        )["out"]
    )
    del _pt_tmp_412
    _pt_tmp_410 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_410)
    _pt_tmp_410 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_410)
    _pt_tmp_409 = 0 + _pt_tmp_410
    _pt_tmp_414 = _pt_tmp_127 / _pt_tmp_66
    del _pt_tmp_410
    _pt_tmp_418 = (
        _pt_tmp_133[_pt_tmp_44, _pt_tmp_45]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_pt_tmp_133, in_1=_pt_tmp_44, in_2=_pt_tmp_45
        )["out"]
    )
    _pt_tmp_418 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_418)
    _pt_tmp_418 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_418)
    _pt_tmp_417 = 0 + _pt_tmp_418
    _pt_tmp_416 = (
        _pt_tmp_417[_pt_tmp_46, _pt_tmp_47]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_417, in_1=_pt_tmp_46, in_2=_pt_tmp_47
        )["out"]
    )
    del _pt_tmp_418
    _pt_tmp_416 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_416)
    _pt_tmp_416 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_416)
    _pt_tmp_415 = 0 + _pt_tmp_416
    _pt_tmp_413 = _pt_tmp_414 * _pt_tmp_415
    del _pt_tmp_416
    _pt_tmp_408 = _pt_tmp_409 - _pt_tmp_413
    _pt_tmp_407 = _pt_tmp_408 / _pt_tmp_66
    del _pt_tmp_409, _pt_tmp_413
    _pt_tmp_393 = _pt_tmp_394 + _pt_tmp_407
    del _pt_tmp_408
    _pt_tmp_388 = _pt_tmp_389 * _pt_tmp_393
    del _pt_tmp_394
    _pt_tmp_387 = _pt_tmp_388 * 1.0
    del _pt_tmp_389, _pt_tmp_393
    _pt_tmp_369 = _pt_tmp_370 + _pt_tmp_387
    _pt_tmp_368 = _pt_tmp_369 * _pt_tmp_382
    del _pt_tmp_370
    _pt_tmp_428 = (
        _pt_tmp_150[_pt_tmp_44, _pt_tmp_45]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_pt_tmp_150, in_1=_pt_tmp_44, in_2=_pt_tmp_45
        )["out"]
    )
    _pt_tmp_428 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_428)
    _pt_tmp_428 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_428)
    _pt_tmp_427 = 0 + _pt_tmp_428
    _pt_tmp_426 = (
        _pt_tmp_427[_pt_tmp_46, _pt_tmp_47]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_427, in_1=_pt_tmp_46, in_2=_pt_tmp_47
        )["out"]
    )
    del _pt_tmp_428
    _pt_tmp_426 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_426)
    _pt_tmp_426 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_426)
    _pt_tmp_425 = 0 + _pt_tmp_426
    _pt_tmp_429 = _pt_tmp_382 * _pt_tmp_403
    del _pt_tmp_426
    _pt_tmp_424 = _pt_tmp_425 - _pt_tmp_429
    _pt_tmp_423 = _pt_tmp_424 / _pt_tmp_66
    del _pt_tmp_425, _pt_tmp_429
    _pt_tmp_435 = (
        _pt_tmp_164[_pt_tmp_44, _pt_tmp_45]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_pt_tmp_164, in_1=_pt_tmp_44, in_2=_pt_tmp_45
        )["out"]
    )
    del _pt_tmp_424
    _pt_tmp_435 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_435)
    _pt_tmp_435 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_435)
    _pt_tmp_434 = 0 + _pt_tmp_435
    _pt_tmp_433 = (
        _pt_tmp_434[_pt_tmp_46, _pt_tmp_47]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_434, in_1=_pt_tmp_46, in_2=_pt_tmp_47
        )["out"]
    )
    del _pt_tmp_435
    _pt_tmp_433 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_433)
    _pt_tmp_433 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_433)
    _pt_tmp_432 = 0 + _pt_tmp_433
    _pt_tmp_436 = _pt_tmp_402 * _pt_tmp_383
    del _pt_tmp_433
    _pt_tmp_431 = _pt_tmp_432 - _pt_tmp_436
    _pt_tmp_430 = _pt_tmp_431 / _pt_tmp_66
    del _pt_tmp_432, _pt_tmp_436
    _pt_tmp_422 = _pt_tmp_423 + _pt_tmp_430
    del _pt_tmp_431
    _pt_tmp_421 = _pt_tmp_371 * _pt_tmp_422
    _pt_tmp_437 = _pt_tmp_388 * 0.0
    del _pt_tmp_422
    _pt_tmp_420 = _pt_tmp_421 + _pt_tmp_437
    del _pt_tmp_388
    _pt_tmp_419 = _pt_tmp_420 * _pt_tmp_402
    del _pt_tmp_421
    _pt_tmp_367 = _pt_tmp_368 + _pt_tmp_419
    _pt_tmp_447 = (
        _pt_tmp_183[_pt_tmp_44, _pt_tmp_45]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_pt_tmp_183, in_1=_pt_tmp_44, in_2=_pt_tmp_45
        )["out"]
    )
    del _pt_tmp_368, _pt_tmp_419
    _pt_tmp_447 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_447)
    _pt_tmp_447 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_447)
    _pt_tmp_446 = 0 + _pt_tmp_447
    _pt_tmp_445 = (
        _pt_tmp_446[_pt_tmp_46, _pt_tmp_47]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_446, in_1=_pt_tmp_46, in_2=_pt_tmp_47
        )["out"]
    )
    del _pt_tmp_447
    _pt_tmp_445 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_445)
    _pt_tmp_445 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_445)
    _pt_tmp_444 = 0 + _pt_tmp_445
    _pt_tmp_448 = _pt_tmp_382 * _pt_tmp_415
    del _pt_tmp_445
    _pt_tmp_443 = _pt_tmp_444 - _pt_tmp_448
    _pt_tmp_442 = _pt_tmp_443 / _pt_tmp_66
    del _pt_tmp_444, _pt_tmp_448
    _pt_tmp_454 = (
        _pt_tmp_197[_pt_tmp_44, _pt_tmp_45]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_pt_tmp_197, in_1=_pt_tmp_44, in_2=_pt_tmp_45
        )["out"]
    )
    del _pt_tmp_443
    _pt_tmp_454 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_454)
    _pt_tmp_454 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_454)
    _pt_tmp_453 = 0 + _pt_tmp_454
    _pt_tmp_452 = (
        _pt_tmp_453[_pt_tmp_46, _pt_tmp_47]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_453, in_1=_pt_tmp_46, in_2=_pt_tmp_47
        )["out"]
    )
    del _pt_tmp_454
    _pt_tmp_452 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_452)
    _pt_tmp_452 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_452)
    _pt_tmp_451 = 0 + _pt_tmp_452
    _pt_tmp_455 = _pt_tmp_414 * _pt_tmp_383
    del _pt_tmp_452
    _pt_tmp_450 = _pt_tmp_451 - _pt_tmp_455
    del _pt_tmp_383
    _pt_tmp_449 = _pt_tmp_450 / _pt_tmp_66
    del _pt_tmp_451, _pt_tmp_455
    _pt_tmp_441 = _pt_tmp_442 + _pt_tmp_449
    del _pt_tmp_450
    _pt_tmp_440 = _pt_tmp_371 * _pt_tmp_441
    _pt_tmp_439 = _pt_tmp_440 + _pt_tmp_437
    del _pt_tmp_441
    _pt_tmp_438 = _pt_tmp_439 * _pt_tmp_414
    del _pt_tmp_440
    _pt_tmp_366 = _pt_tmp_367 + _pt_tmp_438
    _pt_tmp_365 = _pt_tmp_366 + 0
    del _pt_tmp_367, _pt_tmp_438
    _pt_tmp_458 = 10 * _pt_tmp_372
    del _pt_tmp_366
    _pt_tmp_457 = -1 * _pt_tmp_458
    del _pt_tmp_372
    _pt_tmp_462 = (
        _pt_tmp_212[_pt_tmp_44, _pt_tmp_45]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_pt_tmp_212, in_1=_pt_tmp_44, in_2=_pt_tmp_45
        )["out"]
    )
    del _pt_tmp_458
    _pt_tmp_462 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_462)
    del _pt_tmp_212
    _pt_tmp_462 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_462)
    _pt_tmp_461 = 0 + _pt_tmp_462
    _pt_tmp_460 = (
        _pt_tmp_461[_pt_tmp_46, _pt_tmp_47]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_461, in_1=_pt_tmp_46, in_2=_pt_tmp_47
        )["out"]
    )
    del _pt_tmp_462
    _pt_tmp_460 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_460)
    _pt_tmp_460 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_460)
    _pt_tmp_459 = 0 + _pt_tmp_460
    _pt_tmp_456 = _pt_tmp_457 * _pt_tmp_459
    del _pt_tmp_460
    _pt_tmp_364 = _pt_tmp_365 - _pt_tmp_456
    del _pt_tmp_459
    _pt_tmp_472 = 0 * _pt_tmp_68
    del _pt_tmp_365, _pt_tmp_456
    _pt_tmp_471 = _pt_tmp_472 + 1.0
    _pt_tmp_470 = 1e-05 * _pt_tmp_471
    del _pt_tmp_472
    _pt_tmp_477 = _pt_tmp_42 / _pt_tmp_68
    _pt_tmp_476 = _pt_tmp_477 * _pt_tmp_385
    _pt_tmp_475 = _pt_tmp_379 - _pt_tmp_476
    _pt_tmp_474 = _pt_tmp_475 / _pt_tmp_68
    del _pt_tmp_379, _pt_tmp_476
    _pt_tmp_473 = _pt_tmp_474 + _pt_tmp_474
    del _pt_tmp_475
    _pt_tmp_469 = _pt_tmp_470 * _pt_tmp_473
    _pt_tmp_481 = 0 * _pt_tmp_471
    del _pt_tmp_473
    _pt_tmp_483 = 2 * _pt_tmp_470
    _pt_tmp_482 = _pt_tmp_483 / 3
    _pt_tmp_480 = _pt_tmp_481 - _pt_tmp_482
    del _pt_tmp_483
    _pt_tmp_489 = _pt_tmp_96 / _pt_tmp_68
    del _pt_tmp_481, _pt_tmp_482
    _pt_tmp_488 = _pt_tmp_489 * _pt_tmp_405
    _pt_tmp_487 = _pt_tmp_399 - _pt_tmp_488
    _pt_tmp_486 = _pt_tmp_487 / _pt_tmp_68
    del _pt_tmp_399, _pt_tmp_488
    _pt_tmp_485 = _pt_tmp_474 + _pt_tmp_486
    del _pt_tmp_487
    _pt_tmp_493 = _pt_tmp_129 / _pt_tmp_68
    del _pt_tmp_474
    _pt_tmp_492 = _pt_tmp_493 * _pt_tmp_417
    _pt_tmp_491 = _pt_tmp_411 - _pt_tmp_492
    _pt_tmp_490 = _pt_tmp_491 / _pt_tmp_68
    del _pt_tmp_411, _pt_tmp_492
    _pt_tmp_484 = _pt_tmp_485 + _pt_tmp_490
    del _pt_tmp_491
    _pt_tmp_479 = _pt_tmp_480 * _pt_tmp_484
    del _pt_tmp_485
    _pt_tmp_478 = _pt_tmp_479 * 1.0
    del _pt_tmp_480, _pt_tmp_484
    _pt_tmp_468 = _pt_tmp_469 + _pt_tmp_478
    _pt_tmp_467 = _pt_tmp_468 * _pt_tmp_477
    del _pt_tmp_469
    _pt_tmp_500 = _pt_tmp_477 * _pt_tmp_405
    _pt_tmp_499 = _pt_tmp_427 - _pt_tmp_500
    _pt_tmp_498 = _pt_tmp_499 / _pt_tmp_68
    del _pt_tmp_427, _pt_tmp_500
    _pt_tmp_503 = _pt_tmp_489 * _pt_tmp_385
    del _pt_tmp_499
    _pt_tmp_502 = _pt_tmp_434 - _pt_tmp_503
    _pt_tmp_501 = _pt_tmp_502 / _pt_tmp_68
    del _pt_tmp_434, _pt_tmp_503
    _pt_tmp_497 = _pt_tmp_498 + _pt_tmp_501
    del _pt_tmp_502
    _pt_tmp_496 = _pt_tmp_470 * _pt_tmp_497
    _pt_tmp_504 = _pt_tmp_479 * 0.0
    del _pt_tmp_497
    _pt_tmp_495 = _pt_tmp_496 + _pt_tmp_504
    del _pt_tmp_479
    _pt_tmp_494 = _pt_tmp_495 * _pt_tmp_489
    del _pt_tmp_496
    _pt_tmp_466 = _pt_tmp_467 + _pt_tmp_494
    _pt_tmp_511 = _pt_tmp_477 * _pt_tmp_417
    del _pt_tmp_467, _pt_tmp_494
    _pt_tmp_510 = _pt_tmp_446 - _pt_tmp_511
    _pt_tmp_509 = _pt_tmp_510 / _pt_tmp_68
    del _pt_tmp_446, _pt_tmp_511
    _pt_tmp_514 = _pt_tmp_493 * _pt_tmp_385
    del _pt_tmp_510
    _pt_tmp_513 = _pt_tmp_453 - _pt_tmp_514
    del _pt_tmp_385
    _pt_tmp_512 = _pt_tmp_513 / _pt_tmp_68
    del _pt_tmp_453, _pt_tmp_514
    _pt_tmp_508 = _pt_tmp_509 + _pt_tmp_512
    del _pt_tmp_513
    _pt_tmp_507 = _pt_tmp_470 * _pt_tmp_508
    _pt_tmp_506 = _pt_tmp_507 + _pt_tmp_504
    del _pt_tmp_508
    _pt_tmp_505 = _pt_tmp_506 * _pt_tmp_493
    del _pt_tmp_507
    _pt_tmp_465 = _pt_tmp_466 + _pt_tmp_505
    _pt_tmp_464 = _pt_tmp_465 + 0
    del _pt_tmp_466, _pt_tmp_505
    _pt_tmp_517 = 10 * _pt_tmp_471
    del _pt_tmp_465
    _pt_tmp_516 = -1 * _pt_tmp_517
    del _pt_tmp_471
    _pt_tmp_515 = _pt_tmp_516 * _pt_tmp_461
    del _pt_tmp_517
    _pt_tmp_463 = _pt_tmp_464 - _pt_tmp_515
    del _pt_tmp_461
    _pt_tmp_363 = _pt_tmp_364 + _pt_tmp_463
    del _pt_tmp_464, _pt_tmp_515
    _pt_tmp_362 = _pt_tmp_363 / 2
    del _pt_tmp_364, _pt_tmp_463
    _pt_tmp_361 = (
        _pt_tmp_362 * _pt_data_11
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_362, _in1=_pt_data_11)["out"]
    )
    del _pt_tmp_363
    _pt_tmp_528 = _pt_tmp_430 + _pt_tmp_423
    del _pt_tmp_362
    _pt_tmp_527 = _pt_tmp_371 * _pt_tmp_528
    del _pt_tmp_423, _pt_tmp_430
    _pt_tmp_526 = _pt_tmp_527 + _pt_tmp_437
    del _pt_tmp_528
    _pt_tmp_525 = _pt_tmp_526 * _pt_tmp_382
    del _pt_tmp_527
    _pt_tmp_532 = _pt_tmp_395 + _pt_tmp_395
    _pt_tmp_531 = _pt_tmp_371 * _pt_tmp_532
    del _pt_tmp_395
    _pt_tmp_530 = _pt_tmp_531 + _pt_tmp_387
    del _pt_tmp_532
    _pt_tmp_529 = _pt_tmp_530 * _pt_tmp_402
    del _pt_tmp_531
    _pt_tmp_524 = _pt_tmp_525 + _pt_tmp_529
    _pt_tmp_542 = (
        _pt_tmp_282[_pt_tmp_44, _pt_tmp_45]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_pt_tmp_282, in_1=_pt_tmp_44, in_2=_pt_tmp_45
        )["out"]
    )
    del _pt_tmp_525, _pt_tmp_529
    _pt_tmp_542 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_542)
    _pt_tmp_542 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_542)
    _pt_tmp_541 = 0 + _pt_tmp_542
    _pt_tmp_540 = (
        _pt_tmp_541[_pt_tmp_46, _pt_tmp_47]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_541, in_1=_pt_tmp_46, in_2=_pt_tmp_47
        )["out"]
    )
    del _pt_tmp_542
    _pt_tmp_540 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_540)
    _pt_tmp_540 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_540)
    _pt_tmp_539 = 0 + _pt_tmp_540
    _pt_tmp_543 = _pt_tmp_402 * _pt_tmp_415
    del _pt_tmp_540
    _pt_tmp_538 = _pt_tmp_539 - _pt_tmp_543
    del _pt_tmp_415
    _pt_tmp_537 = _pt_tmp_538 / _pt_tmp_66
    del _pt_tmp_539, _pt_tmp_543
    _pt_tmp_549 = (
        _pt_tmp_296[_pt_tmp_44, _pt_tmp_45]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_pt_tmp_296, in_1=_pt_tmp_44, in_2=_pt_tmp_45
        )["out"]
    )
    del _pt_tmp_538
    _pt_tmp_549 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_549)
    _pt_tmp_549 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_549)
    _pt_tmp_548 = 0 + _pt_tmp_549
    _pt_tmp_547 = (
        _pt_tmp_548[_pt_tmp_46, _pt_tmp_47]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_548, in_1=_pt_tmp_46, in_2=_pt_tmp_47
        )["out"]
    )
    del _pt_tmp_549
    _pt_tmp_547 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_547)
    _pt_tmp_547 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_547)
    _pt_tmp_546 = 0 + _pt_tmp_547
    _pt_tmp_550 = _pt_tmp_414 * _pt_tmp_403
    del _pt_tmp_547
    _pt_tmp_545 = _pt_tmp_546 - _pt_tmp_550
    del _pt_tmp_403
    _pt_tmp_544 = _pt_tmp_545 / _pt_tmp_66
    del _pt_tmp_546, _pt_tmp_550
    _pt_tmp_536 = _pt_tmp_537 + _pt_tmp_544
    del _pt_tmp_545
    _pt_tmp_535 = _pt_tmp_371 * _pt_tmp_536
    _pt_tmp_534 = _pt_tmp_535 + _pt_tmp_437
    del _pt_tmp_536
    _pt_tmp_533 = _pt_tmp_534 * _pt_tmp_414
    del _pt_tmp_535
    _pt_tmp_523 = _pt_tmp_524 + _pt_tmp_533
    _pt_tmp_522 = _pt_tmp_523 + 0
    del _pt_tmp_524, _pt_tmp_533
    _pt_tmp_555 = (
        _pt_tmp_309[_pt_tmp_44, _pt_tmp_45]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_pt_tmp_309, in_1=_pt_tmp_44, in_2=_pt_tmp_45
        )["out"]
    )
    del _pt_tmp_523
    _pt_tmp_555 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_555)
    del _pt_tmp_309
    _pt_tmp_555 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_555)
    _pt_tmp_554 = 0 + _pt_tmp_555
    _pt_tmp_553 = (
        _pt_tmp_554[_pt_tmp_46, _pt_tmp_47]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_554, in_1=_pt_tmp_46, in_2=_pt_tmp_47
        )["out"]
    )
    del _pt_tmp_555
    _pt_tmp_553 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_553)
    _pt_tmp_553 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_553)
    _pt_tmp_552 = 0 + _pt_tmp_553
    _pt_tmp_551 = _pt_tmp_457 * _pt_tmp_552
    del _pt_tmp_553
    _pt_tmp_521 = _pt_tmp_522 - _pt_tmp_551
    del _pt_tmp_552
    _pt_tmp_563 = _pt_tmp_501 + _pt_tmp_498
    del _pt_tmp_522, _pt_tmp_551
    _pt_tmp_562 = _pt_tmp_470 * _pt_tmp_563
    del _pt_tmp_498, _pt_tmp_501
    _pt_tmp_561 = _pt_tmp_562 + _pt_tmp_504
    del _pt_tmp_563
    _pt_tmp_560 = _pt_tmp_561 * _pt_tmp_477
    del _pt_tmp_562
    _pt_tmp_567 = _pt_tmp_486 + _pt_tmp_486
    _pt_tmp_566 = _pt_tmp_470 * _pt_tmp_567
    del _pt_tmp_486
    _pt_tmp_565 = _pt_tmp_566 + _pt_tmp_478
    del _pt_tmp_567
    _pt_tmp_564 = _pt_tmp_565 * _pt_tmp_489
    del _pt_tmp_566
    _pt_tmp_559 = _pt_tmp_560 + _pt_tmp_564
    _pt_tmp_574 = _pt_tmp_489 * _pt_tmp_417
    del _pt_tmp_560, _pt_tmp_564
    _pt_tmp_573 = _pt_tmp_541 - _pt_tmp_574
    del _pt_tmp_417
    _pt_tmp_572 = _pt_tmp_573 / _pt_tmp_68
    del _pt_tmp_541, _pt_tmp_574
    _pt_tmp_577 = _pt_tmp_493 * _pt_tmp_405
    del _pt_tmp_573
    _pt_tmp_576 = _pt_tmp_548 - _pt_tmp_577
    del _pt_tmp_405
    _pt_tmp_575 = _pt_tmp_576 / _pt_tmp_68
    del _pt_tmp_548, _pt_tmp_577
    _pt_tmp_571 = _pt_tmp_572 + _pt_tmp_575
    del _pt_tmp_576
    _pt_tmp_570 = _pt_tmp_470 * _pt_tmp_571
    _pt_tmp_569 = _pt_tmp_570 + _pt_tmp_504
    del _pt_tmp_571
    _pt_tmp_568 = _pt_tmp_569 * _pt_tmp_493
    del _pt_tmp_570
    _pt_tmp_558 = _pt_tmp_559 + _pt_tmp_568
    _pt_tmp_557 = _pt_tmp_558 + 0
    del _pt_tmp_559, _pt_tmp_568
    _pt_tmp_578 = _pt_tmp_516 * _pt_tmp_554
    del _pt_tmp_558
    _pt_tmp_556 = _pt_tmp_557 - _pt_tmp_578
    del _pt_tmp_554
    _pt_tmp_520 = _pt_tmp_521 + _pt_tmp_556
    del _pt_tmp_557, _pt_tmp_578
    _pt_tmp_519 = _pt_tmp_520 / 2
    del _pt_tmp_521, _pt_tmp_556
    _pt_tmp_518 = (
        _pt_tmp_519 * _pt_data_15
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_519, _in1=_pt_data_15)["out"]
    )
    del _pt_tmp_520
    _pt_tmp_360 = _pt_tmp_361 + _pt_tmp_518
    del _pt_tmp_519
    _pt_tmp_589 = _pt_tmp_449 + _pt_tmp_442
    del _pt_tmp_361, _pt_tmp_518
    _pt_tmp_588 = _pt_tmp_371 * _pt_tmp_589
    del _pt_tmp_442, _pt_tmp_449
    _pt_tmp_587 = _pt_tmp_588 + _pt_tmp_437
    del _pt_tmp_589
    _pt_tmp_586 = _pt_tmp_587 * _pt_tmp_382
    del _pt_tmp_588
    _pt_tmp_593 = _pt_tmp_544 + _pt_tmp_537
    _pt_tmp_592 = _pt_tmp_371 * _pt_tmp_593
    del _pt_tmp_537, _pt_tmp_544
    _pt_tmp_591 = _pt_tmp_592 + _pt_tmp_437
    del _pt_tmp_593
    _pt_tmp_590 = _pt_tmp_591 * _pt_tmp_402
    del _pt_tmp_437, _pt_tmp_592
    _pt_tmp_585 = _pt_tmp_586 + _pt_tmp_590
    _pt_tmp_597 = _pt_tmp_407 + _pt_tmp_407
    del _pt_tmp_586, _pt_tmp_590
    _pt_tmp_596 = _pt_tmp_371 * _pt_tmp_597
    del _pt_tmp_407
    _pt_tmp_595 = _pt_tmp_596 + _pt_tmp_387
    del _pt_tmp_371, _pt_tmp_597
    _pt_tmp_594 = _pt_tmp_595 * _pt_tmp_414
    del _pt_tmp_387, _pt_tmp_596
    _pt_tmp_584 = _pt_tmp_585 + _pt_tmp_594
    _pt_tmp_583 = _pt_tmp_584 + 0
    del _pt_tmp_585, _pt_tmp_594
    _pt_tmp_602 = (
        _pt_tmp_339[_pt_tmp_44, _pt_tmp_45]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_pt_tmp_339, in_1=_pt_tmp_44, in_2=_pt_tmp_45
        )["out"]
    )
    del _pt_tmp_584
    _pt_tmp_602 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_602)
    del _pt_tmp_339
    _pt_tmp_602 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_602)
    _pt_tmp_601 = 0 + _pt_tmp_602
    _pt_tmp_600 = (
        _pt_tmp_601[_pt_tmp_46, _pt_tmp_47]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_601, in_1=_pt_tmp_46, in_2=_pt_tmp_47
        )["out"]
    )
    del _pt_tmp_602
    _pt_tmp_600 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_600)
    _pt_tmp_600 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_600)
    _pt_tmp_599 = 0 + _pt_tmp_600
    _pt_tmp_598 = _pt_tmp_457 * _pt_tmp_599
    del _pt_tmp_600
    _pt_tmp_582 = _pt_tmp_583 - _pt_tmp_598
    del _pt_tmp_457, _pt_tmp_599
    _pt_tmp_610 = _pt_tmp_512 + _pt_tmp_509
    del _pt_tmp_583, _pt_tmp_598
    _pt_tmp_609 = _pt_tmp_470 * _pt_tmp_610
    del _pt_tmp_509, _pt_tmp_512
    _pt_tmp_608 = _pt_tmp_609 + _pt_tmp_504
    del _pt_tmp_610
    _pt_tmp_607 = _pt_tmp_608 * _pt_tmp_477
    del _pt_tmp_609
    _pt_tmp_614 = _pt_tmp_575 + _pt_tmp_572
    _pt_tmp_613 = _pt_tmp_470 * _pt_tmp_614
    del _pt_tmp_572, _pt_tmp_575
    _pt_tmp_612 = _pt_tmp_613 + _pt_tmp_504
    del _pt_tmp_614
    _pt_tmp_611 = _pt_tmp_612 * _pt_tmp_489
    del _pt_tmp_504, _pt_tmp_613
    _pt_tmp_606 = _pt_tmp_607 + _pt_tmp_611
    _pt_tmp_618 = _pt_tmp_490 + _pt_tmp_490
    del _pt_tmp_607, _pt_tmp_611
    _pt_tmp_617 = _pt_tmp_470 * _pt_tmp_618
    del _pt_tmp_490
    _pt_tmp_616 = _pt_tmp_617 + _pt_tmp_478
    del _pt_tmp_470, _pt_tmp_618
    _pt_tmp_615 = _pt_tmp_616 * _pt_tmp_493
    del _pt_tmp_478, _pt_tmp_617
    _pt_tmp_605 = _pt_tmp_606 + _pt_tmp_615
    _pt_tmp_604 = _pt_tmp_605 + 0
    del _pt_tmp_606, _pt_tmp_615
    _pt_tmp_619 = _pt_tmp_516 * _pt_tmp_601
    del _pt_tmp_605
    _pt_tmp_603 = _pt_tmp_604 - _pt_tmp_619
    del _pt_tmp_516, _pt_tmp_601
    _pt_tmp_581 = _pt_tmp_582 + _pt_tmp_603
    del _pt_tmp_604, _pt_tmp_619
    _pt_tmp_580 = _pt_tmp_581 / 2
    del _pt_tmp_582, _pt_tmp_603
    _pt_tmp_579 = (
        _pt_tmp_580 * _pt_data_16
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_580, _in1=_pt_data_16)["out"]
    )
    del _pt_tmp_581
    _pt_tmp_359 = _pt_tmp_360 + _pt_tmp_579
    del _pt_tmp_580
    _pt_tmp_358 = (
        _pt_tmp_359[_pt_tmp_48, _pt_tmp_49]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_359, in_1=_pt_tmp_48, in_2=_pt_tmp_49
        )["out"]
    )
    del _pt_tmp_360, _pt_tmp_579
    _pt_tmp_358 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_358)
    del _pt_tmp_359
    _pt_tmp_358 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_358)
    _pt_tmp_357 = 0 + _pt_tmp_358
    _pt_tmp_356 = 0 + _pt_tmp_357
    del _pt_tmp_358
    _pt_tmp_355 = 0 + _pt_tmp_356
    del _pt_tmp_357
    _pt_tmp_632 = _pt_tmp_252 * 0.3999999999999999
    del _pt_tmp_356
    _pt_tmp_631 = _pt_tmp_241 + _pt_tmp_632
    del _pt_tmp_252
    _pt_tmp_630 = _pt_tmp_477 * _pt_tmp_631
    _pt_tmp_629 = (
        _pt_tmp_630 * _pt_data_11
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_630, _in1=_pt_data_11)["out"]
    )
    _pt_tmp_634 = _pt_tmp_489 * _pt_tmp_631
    del _pt_tmp_630
    _pt_tmp_633 = (
        _pt_tmp_634 * _pt_data_15
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_634, _in1=_pt_data_15)["out"]
    )
    _pt_tmp_628 = _pt_tmp_629 + _pt_tmp_633
    del _pt_tmp_634
    _pt_tmp_636 = _pt_tmp_493 * _pt_tmp_631
    del _pt_tmp_629, _pt_tmp_633
    _pt_tmp_635 = (
        _pt_tmp_636 * _pt_data_16
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_636, _in1=_pt_data_16)["out"]
    )
    del _pt_tmp_631
    _pt_tmp_627 = _pt_tmp_628 + _pt_tmp_635
    del _pt_tmp_636
    _pt_tmp_642 = _pt_tmp_238 * 0.3999999999999999
    del _pt_tmp_628, _pt_tmp_635
    _pt_tmp_641 = _pt_tmp_239 + _pt_tmp_642
    del _pt_tmp_238
    _pt_tmp_640 = _pt_tmp_382 * _pt_tmp_641
    _pt_tmp_639 = (
        _pt_tmp_640 * _pt_data_11
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_640, _in1=_pt_data_11)["out"]
    )
    _pt_tmp_644 = _pt_tmp_402 * _pt_tmp_641
    del _pt_tmp_640
    _pt_tmp_643 = (
        _pt_tmp_644 * _pt_data_15
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_644, _in1=_pt_data_15)["out"]
    )
    _pt_tmp_638 = _pt_tmp_639 + _pt_tmp_643
    del _pt_tmp_644
    _pt_tmp_646 = _pt_tmp_414 * _pt_tmp_641
    del _pt_tmp_639, _pt_tmp_643
    _pt_tmp_645 = (
        _pt_tmp_646 * _pt_data_16
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_646, _in1=_pt_data_16)["out"]
    )
    del _pt_tmp_641
    _pt_tmp_637 = _pt_tmp_638 + _pt_tmp_645
    del _pt_tmp_646
    _pt_tmp_626 = _pt_tmp_627 + _pt_tmp_637
    del _pt_tmp_638, _pt_tmp_645
    _pt_tmp_655 = _pt_tmp_477 * _pt_tmp_477
    del _pt_tmp_627, _pt_tmp_637
    _pt_tmp_656 = _pt_tmp_489 * _pt_tmp_489
    _pt_tmp_654 = _pt_tmp_655 + _pt_tmp_656
    _pt_tmp_657 = _pt_tmp_493 * _pt_tmp_493
    _pt_tmp_653 = _pt_tmp_654 + _pt_tmp_657
    _pt_tmp_652 = actx.np.sqrt(_pt_tmp_653)
    del _pt_tmp_654
    _pt_tmp_660 = 1.4 / _pt_tmp_68
    del _pt_tmp_653
    _pt_tmp_659 = _pt_tmp_660 * _pt_tmp_632
    _pt_tmp_658 = actx.np.sqrt(_pt_tmp_659)
    del _pt_tmp_660
    _pt_tmp_651 = _pt_tmp_652 + _pt_tmp_658
    del _pt_tmp_659
    _pt_tmp_650 = actx.np.isnan(_pt_tmp_651)
    del _pt_tmp_652, _pt_tmp_658
    _pt_tmp_666 = _pt_tmp_382 * _pt_tmp_382
    _pt_tmp_667 = _pt_tmp_402 * _pt_tmp_402
    _pt_tmp_665 = _pt_tmp_666 + _pt_tmp_667
    _pt_tmp_668 = _pt_tmp_414 * _pt_tmp_414
    _pt_tmp_664 = _pt_tmp_665 + _pt_tmp_668
    _pt_tmp_663 = actx.np.sqrt(_pt_tmp_664)
    del _pt_tmp_665
    _pt_tmp_671 = 1.4 / _pt_tmp_66
    del _pt_tmp_664
    _pt_tmp_670 = _pt_tmp_671 * _pt_tmp_642
    _pt_tmp_669 = actx.np.sqrt(_pt_tmp_670)
    del _pt_tmp_671
    _pt_tmp_662 = _pt_tmp_663 + _pt_tmp_669
    del _pt_tmp_670
    _pt_tmp_661 = actx.np.isnan(_pt_tmp_662)
    del _pt_tmp_663, _pt_tmp_669
    _pt_tmp_649 = actx.np.logical_or(_pt_tmp_650, _pt_tmp_661)
    _pt_tmp_673 = actx.np.greater(_pt_tmp_651, _pt_tmp_662)
    del _pt_tmp_650, _pt_tmp_661
    _pt_tmp_672 = actx.np.where(_pt_tmp_673, _pt_tmp_651, _pt_tmp_662)
    _pt_tmp_648 = actx.np.where(_pt_tmp_649, np.float64("nan"), _pt_tmp_672)
    del _pt_tmp_651, _pt_tmp_662, _pt_tmp_673
    _pt_tmp_674 = _pt_tmp_241 - _pt_tmp_239
    del _pt_tmp_649, _pt_tmp_672
    _pt_tmp_647 = _pt_tmp_648 * _pt_tmp_674
    _pt_tmp_625 = _pt_tmp_626 + _pt_tmp_647
    del _pt_tmp_674
    _pt_tmp_624 = _pt_tmp_625 / 2
    del _pt_tmp_626, _pt_tmp_647
    _pt_tmp_623 = (
        _pt_tmp_624[_pt_tmp_48, _pt_tmp_49]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_624, in_1=_pt_tmp_48, in_2=_pt_tmp_49
        )["out"]
    )
    del _pt_tmp_625
    _pt_tmp_623 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_623)
    del _pt_tmp_624
    _pt_tmp_623 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_623)
    _pt_tmp_622 = 0 + _pt_tmp_623
    _pt_tmp_621 = 0 + _pt_tmp_622
    del _pt_tmp_623
    _pt_tmp_620 = _pt_tmp_621 + 0
    del _pt_tmp_622
    _pt_tmp_354 = _pt_tmp_355 - _pt_tmp_620
    del _pt_tmp_621
    _pt_tmp_353 = actx.np.reshape(_pt_tmp_354, (4, 48000, 15))
    del _pt_tmp_355, _pt_tmp_620
    _pt_tmp_353 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_353)
    del _pt_tmp_354
    _pt_tmp_352 = actx.einsum(
        "ijk, jl, jlk -> li", _pt_data_3, _pt_tmp_30, _pt_tmp_353
    )
    _pt_tmp_352 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_352)
    del _pt_tmp_353
    _pt_tmp_7 = _pt_tmp_8 - _pt_tmp_352
    _pt_tmp_4 = actx.einsum("i, jk, ik -> ij", _pt_tmp_5, _pt_data_0, _pt_tmp_7)
    del _pt_tmp_352, _pt_tmp_8
    _pt_tmp_4 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_4)
    del _pt_tmp_7
    _pt_tmp_3 = -1 * _pt_tmp_4
    _pt_tmp_675 = 0 * _pt_tmp_3
    del _pt_tmp_4
    _pt_tmp_2 = _pt_tmp_3 + _pt_tmp_675
    _pt_tmp_689 = _pt_tmp_51 * _pt_tmp_51
    del _pt_tmp_3, _pt_tmp_675
    _pt_tmp_690 = _pt_tmp_99 * _pt_tmp_99
    _pt_tmp_688 = _pt_tmp_689 + _pt_tmp_690
    _pt_tmp_691 = _pt_tmp_132 * _pt_tmp_132
    _pt_tmp_687 = _pt_tmp_688 + _pt_tmp_691
    _pt_tmp_686 = actx.np.sqrt(_pt_tmp_687)
    del _pt_tmp_688
    _pt_tmp_685 = 0.5 * _pt_tmp_686
    del _pt_tmp_687
    _pt_data_17 = actx.thaw(npzfile["_pt_data_17"])
    del _pt_tmp_686
    _pt_data_17 = actx.tag((PrefixNamed(prefix="char_lscales"),), _pt_data_17)
    _pt_data_17 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_data_17)
    _pt_data_17 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_data_17)
    _pt_tmp_684 = (
        _pt_tmp_685 * _pt_data_17
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_5, _in0=_pt_tmp_685, _in1=_pt_data_17)["out"]
    )
    _pt_tmp_683 = -1 * _pt_tmp_684
    del _pt_tmp_685
    _pt_data_18 = actx.thaw(npzfile["_pt_data_18"])
    del _pt_tmp_684
    _pt_data_18 = actx.tag_axis(0, (DiscretizationDOFAxisTag(),), _pt_data_18)
    _pt_tmp_701 = actx.einsum("ij, kj -> ki", _pt_data_18, _actx_in_1_0_mass_0)
    _pt_tmp_701 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_701)
    _pt_tmp_700 = _pt_tmp_701**2
    _pt_data_19 = actx.thaw(npzfile["_pt_data_19"])
    del _pt_tmp_701
    _pt_tmp_699 = actx.einsum("ij, j -> i", _pt_tmp_700, _pt_data_19)
    _pt_tmp_703 = _pt_tmp_700 + 2.857142857142857e-14
    _pt_tmp_702 = actx.einsum("ij -> i", _pt_tmp_703)
    del _pt_tmp_700
    _pt_tmp_698 = _pt_tmp_699 / _pt_tmp_702
    del _pt_tmp_703
    _pt_tmp_697 = actx.np.reshape(_pt_tmp_698, (48000, 1))
    del _pt_tmp_699, _pt_tmp_702
    _pt_tmp_696 = (
        actx.np.broadcast_to(_pt_tmp_697, (48000, 35))
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_6, in_0=_pt_tmp_697)["out"]
    )
    del _pt_tmp_698
    _pt_tmp_696 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_696)
    del _pt_tmp_697
    _pt_tmp_695 = _pt_tmp_696 + 1e-12
    _pt_tmp_694 = actx.np.log10(_pt_tmp_695)
    del _pt_tmp_696
    _pt_tmp_693 = actx.np.greater(_pt_tmp_694, -4.5)
    del _pt_tmp_695
    _pt_tmp_705 = actx.np.greater(_pt_tmp_694, -5.5)
    _pt_tmp_711 = _pt_tmp_694 + 5.0
    _pt_tmp_710 = 3.141592653589793 * _pt_tmp_711
    _pt_tmp_709 = _pt_tmp_710 / 1.0
    del _pt_tmp_711
    _pt_tmp_708 = actx.np.sin(_pt_tmp_709)
    del _pt_tmp_710
    _pt_tmp_707 = 1.0 + _pt_tmp_708
    del _pt_tmp_709
    _pt_tmp_706 = 0.5 * _pt_tmp_707
    del _pt_tmp_708
    _pt_tmp_712 = 0.0 * _pt_tmp_694
    del _pt_tmp_707
    _pt_tmp_704 = actx.np.where(_pt_tmp_705, _pt_tmp_706, _pt_tmp_712)
    del _pt_tmp_694
    _pt_tmp_692 = actx.np.where(_pt_tmp_693, 1.0, _pt_tmp_704)
    del _pt_tmp_705, _pt_tmp_706, _pt_tmp_712
    _pt_tmp_682 = _pt_tmp_683 * _pt_tmp_692
    del _pt_tmp_693, _pt_tmp_704
    _pt_tmp_717 = actx.einsum(
        "ijk, jlm, km -> ikl", _pt_tmp_9, _pt_data_2, _actx_in_1_0_energy_0
    )
    del _pt_tmp_683, _pt_tmp_692
    _pt_tmp_717 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_717)
    _pt_tmp_716 = _pt_tmp_717[0]
    _pt_tmp_726 = _pt_tmp_239 + _pt_tmp_241
    _pt_tmp_725 = _pt_tmp_726 / 2
    del _pt_tmp_239, _pt_tmp_241
    _pt_tmp_724 = (
        _pt_tmp_725 * _pt_data_11
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_725, _in1=_pt_data_11)["out"]
    )
    del _pt_tmp_726
    _pt_tmp_723 = (
        _pt_tmp_724[_pt_tmp_48, _pt_tmp_49]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_724, in_1=_pt_tmp_48, in_2=_pt_tmp_49
        )["out"]
    )
    _pt_tmp_723 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_723)
    del _pt_tmp_724
    _pt_tmp_723 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_723)
    _pt_tmp_722 = 0 + _pt_tmp_723
    _pt_tmp_721 = 0 + _pt_tmp_722
    del _pt_tmp_723
    _pt_tmp_720 = 0 + _pt_tmp_721
    del _pt_tmp_722
    _pt_tmp_719 = actx.np.reshape(_pt_tmp_720, (4, 48000, 15))
    del _pt_tmp_721
    _pt_tmp_719 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_719)
    del _pt_tmp_720
    _pt_tmp_718 = actx.einsum(
        "ijk, jl, jlk -> li", _pt_data_3, _pt_tmp_30, _pt_tmp_719
    )
    _pt_tmp_718 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_718)
    del _pt_tmp_719
    _pt_tmp_715 = _pt_tmp_716 - _pt_tmp_718
    _pt_tmp_714 = actx.einsum("i, jk, ik -> ij", _pt_tmp_5, _pt_data_0, _pt_tmp_715)
    del _pt_tmp_716, _pt_tmp_718
    _pt_tmp_714 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_714)
    del _pt_tmp_715
    _pt_tmp_713 = -1 * _pt_tmp_714
    _pt_tmp_681 = _pt_tmp_682 * _pt_tmp_713
    del _pt_tmp_714
    _pt_tmp_731 = _pt_tmp_717[1]
    del _pt_tmp_713
    _pt_tmp_738 = (
        _pt_tmp_725 * _pt_data_15
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_725, _in1=_pt_data_15)["out"]
    )
    _pt_tmp_737 = (
        _pt_tmp_738[_pt_tmp_48, _pt_tmp_49]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_738, in_1=_pt_tmp_48, in_2=_pt_tmp_49
        )["out"]
    )
    _pt_tmp_737 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_737)
    del _pt_tmp_738
    _pt_tmp_737 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_737)
    _pt_tmp_736 = 0 + _pt_tmp_737
    _pt_tmp_735 = 0 + _pt_tmp_736
    del _pt_tmp_737
    _pt_tmp_734 = 0 + _pt_tmp_735
    del _pt_tmp_736
    _pt_tmp_733 = actx.np.reshape(_pt_tmp_734, (4, 48000, 15))
    del _pt_tmp_735
    _pt_tmp_733 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_733)
    del _pt_tmp_734
    _pt_tmp_732 = actx.einsum(
        "ijk, jl, jlk -> li", _pt_data_3, _pt_tmp_30, _pt_tmp_733
    )
    _pt_tmp_732 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_732)
    del _pt_tmp_733
    _pt_tmp_730 = _pt_tmp_731 - _pt_tmp_732
    _pt_tmp_729 = actx.einsum("i, jk, ik -> ij", _pt_tmp_5, _pt_data_0, _pt_tmp_730)
    del _pt_tmp_731, _pt_tmp_732
    _pt_tmp_729 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_729)
    del _pt_tmp_730
    _pt_tmp_728 = -1 * _pt_tmp_729
    _pt_tmp_727 = _pt_tmp_682 * _pt_tmp_728
    del _pt_tmp_729
    _pt_tmp_743 = _pt_tmp_717[2]
    del _pt_tmp_728
    _pt_tmp_750 = (
        _pt_tmp_725 * _pt_data_16
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_725, _in1=_pt_data_16)["out"]
    )
    del _pt_tmp_717
    _pt_tmp_749 = (
        _pt_tmp_750[_pt_tmp_48, _pt_tmp_49]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_750, in_1=_pt_tmp_48, in_2=_pt_tmp_49
        )["out"]
    )
    del _pt_tmp_725
    _pt_tmp_749 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_749)
    del _pt_tmp_750
    _pt_tmp_749 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_749)
    _pt_tmp_748 = 0 + _pt_tmp_749
    _pt_tmp_747 = 0 + _pt_tmp_748
    del _pt_tmp_749
    _pt_tmp_746 = 0 + _pt_tmp_747
    del _pt_tmp_748
    _pt_tmp_745 = actx.np.reshape(_pt_tmp_746, (4, 48000, 15))
    del _pt_tmp_747
    _pt_tmp_745 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_745)
    del _pt_tmp_746
    _pt_tmp_744 = actx.einsum(
        "ijk, jl, jlk -> li", _pt_data_3, _pt_tmp_30, _pt_tmp_745
    )
    _pt_tmp_744 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_744)
    del _pt_tmp_745
    _pt_tmp_742 = _pt_tmp_743 - _pt_tmp_744
    _pt_tmp_741 = actx.einsum("i, jk, ik -> ij", _pt_tmp_5, _pt_data_0, _pt_tmp_742)
    del _pt_tmp_743, _pt_tmp_744
    _pt_tmp_741 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_741)
    del _pt_tmp_742
    _pt_tmp_740 = -1 * _pt_tmp_741
    _pt_tmp_739 = _pt_tmp_682 * _pt_tmp_740
    del _pt_tmp_741
    _pt_tmp_751 = actx.np.stack([_pt_tmp_681, _pt_tmp_727, _pt_tmp_739], axis=0)
    del _pt_tmp_740
    _pt_tmp_680 = actx.einsum(
        "ijk, jlm, ikm -> kl", _pt_tmp_9, _pt_data_2, _pt_tmp_751
    )
    _pt_tmp_680 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_680)
    del _pt_tmp_751
    _pt_tmp_766 = (
        _pt_tmp_681[_pt_tmp_44, _pt_tmp_45]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_pt_tmp_681, in_1=_pt_tmp_44, in_2=_pt_tmp_45
        )["out"]
    )
    _pt_tmp_766 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_766)
    del _pt_tmp_681
    _pt_tmp_766 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_766)
    _pt_tmp_765 = 0 + _pt_tmp_766
    _pt_tmp_764 = (
        _pt_tmp_765[_pt_tmp_46, _pt_tmp_47]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_765, in_1=_pt_tmp_46, in_2=_pt_tmp_47
        )["out"]
    )
    del _pt_tmp_766
    _pt_tmp_764 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_764)
    _pt_tmp_764 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_764)
    _pt_tmp_763 = 0 + _pt_tmp_764
    _pt_tmp_762 = _pt_tmp_763 + _pt_tmp_765
    del _pt_tmp_764
    _pt_tmp_761 = _pt_tmp_762 / 2
    del _pt_tmp_763, _pt_tmp_765
    _pt_tmp_760 = (
        _pt_tmp_761 * _pt_data_11
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_761, _in1=_pt_data_11)["out"]
    )
    del _pt_tmp_762
    _pt_tmp_773 = (
        _pt_tmp_727[_pt_tmp_44, _pt_tmp_45]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_pt_tmp_727, in_1=_pt_tmp_44, in_2=_pt_tmp_45
        )["out"]
    )
    del _pt_tmp_761
    _pt_tmp_773 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_773)
    del _pt_tmp_727
    _pt_tmp_773 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_773)
    _pt_tmp_772 = 0 + _pt_tmp_773
    _pt_tmp_771 = (
        _pt_tmp_772[_pt_tmp_46, _pt_tmp_47]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_772, in_1=_pt_tmp_46, in_2=_pt_tmp_47
        )["out"]
    )
    del _pt_tmp_773
    _pt_tmp_771 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_771)
    _pt_tmp_771 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_771)
    _pt_tmp_770 = 0 + _pt_tmp_771
    _pt_tmp_769 = _pt_tmp_770 + _pt_tmp_772
    del _pt_tmp_771
    _pt_tmp_768 = _pt_tmp_769 / 2
    del _pt_tmp_770, _pt_tmp_772
    _pt_tmp_767 = (
        _pt_tmp_768 * _pt_data_15
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_768, _in1=_pt_data_15)["out"]
    )
    del _pt_tmp_769
    _pt_tmp_759 = _pt_tmp_760 + _pt_tmp_767
    del _pt_tmp_768
    _pt_tmp_780 = (
        _pt_tmp_739[_pt_tmp_44, _pt_tmp_45]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_pt_tmp_739, in_1=_pt_tmp_44, in_2=_pt_tmp_45
        )["out"]
    )
    del _pt_tmp_760, _pt_tmp_767
    _pt_tmp_780 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_780)
    del _pt_tmp_739
    _pt_tmp_780 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_780)
    _pt_tmp_779 = 0 + _pt_tmp_780
    _pt_tmp_778 = (
        _pt_tmp_779[_pt_tmp_46, _pt_tmp_47]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_779, in_1=_pt_tmp_46, in_2=_pt_tmp_47
        )["out"]
    )
    del _pt_tmp_780
    _pt_tmp_778 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_778)
    _pt_tmp_778 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_778)
    _pt_tmp_777 = 0 + _pt_tmp_778
    _pt_tmp_776 = _pt_tmp_777 + _pt_tmp_779
    del _pt_tmp_778
    _pt_tmp_775 = _pt_tmp_776 / 2
    del _pt_tmp_777, _pt_tmp_779
    _pt_tmp_774 = (
        _pt_tmp_775 * _pt_data_16
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_775, _in1=_pt_data_16)["out"]
    )
    del _pt_tmp_776
    _pt_tmp_758 = _pt_tmp_759 + _pt_tmp_774
    del _pt_tmp_775
    _pt_tmp_757 = (
        _pt_tmp_758[_pt_tmp_48, _pt_tmp_49]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_758, in_1=_pt_tmp_48, in_2=_pt_tmp_49
        )["out"]
    )
    del _pt_tmp_759, _pt_tmp_774
    _pt_tmp_757 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_757)
    del _pt_tmp_758
    _pt_tmp_757 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_757)
    _pt_tmp_756 = 0 + _pt_tmp_757
    _pt_tmp_755 = 0 + _pt_tmp_756
    del _pt_tmp_757
    _pt_tmp_754 = _pt_tmp_755 + 0
    del _pt_tmp_756
    _pt_tmp_753 = actx.np.reshape(_pt_tmp_754, (4, 48000, 15))
    del _pt_tmp_755
    _pt_tmp_753 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_753)
    del _pt_tmp_754
    _pt_tmp_752 = actx.einsum(
        "ijk, jl, jlk -> li", _pt_data_3, _pt_tmp_30, _pt_tmp_753
    )
    _pt_tmp_752 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_752)
    del _pt_tmp_753
    _pt_tmp_679 = _pt_tmp_680 - _pt_tmp_752
    _pt_tmp_678 = actx.einsum("i, jk, ik -> ij", _pt_tmp_5, _pt_data_0, _pt_tmp_679)
    del _pt_tmp_680, _pt_tmp_752
    _pt_tmp_678 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_678)
    del _pt_tmp_679
    _pt_tmp_677 = -1 * _pt_tmp_678
    _pt_tmp_676 = -1 * _pt_tmp_677
    del _pt_tmp_678
    _pt_tmp_1 = _pt_tmp_2 + _pt_tmp_676
    del _pt_tmp_677
    _pt_data_20 = actx.thaw(npzfile["_pt_data_20"])
    del _pt_tmp_2, _pt_tmp_676
    _pt_data_20 = actx.tag(
        (FirstAxisIsElementsTag(), PrefixNamed(prefix="nodes0_3d")), _pt_data_20
    )
    _pt_data_20 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_data_20)
    _pt_data_20 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_data_20)
    _pt_tmp_786 = 0 * _pt_data_20
    _pt_tmp_785 = _pt_tmp_786 + 0.9
    _pt_tmp_784 = actx.np.greater(_pt_data_20, _pt_tmp_785)
    _pt_tmp_790 = _pt_data_20 + -0.9
    del _pt_tmp_785
    _pt_tmp_789 = _pt_tmp_790 / 0.09
    _pt_tmp_788 = _pt_tmp_789 * _pt_tmp_789
    del _pt_tmp_790
    _pt_tmp_787 = _pt_tmp_786 + _pt_tmp_788
    del _pt_tmp_789
    _pt_tmp_791 = _pt_tmp_786 + 0.0
    del _pt_tmp_788
    _pt_tmp_783 = actx.np.where(_pt_tmp_784, _pt_tmp_787, _pt_tmp_791)
    _pt_tmp_782 = 10000000000.0 * _pt_tmp_783
    del _pt_tmp_784, _pt_tmp_787, _pt_tmp_791
    _pt_tmp_799 = _pt_tmp_786 + 0.23397065362031969
    del _pt_tmp_783
    _pt_tmp_798 = 0.0 * _pt_tmp_799
    del _pt_tmp_786
    _pt_tmp_797 = _pt_tmp_798 * _pt_tmp_798
    _pt_tmp_796 = _pt_tmp_797 + _pt_tmp_797
    _pt_tmp_795 = _pt_tmp_796 + _pt_tmp_797
    _pt_tmp_800 = 2.0 * _pt_tmp_799
    del _pt_tmp_796, _pt_tmp_797
    _pt_tmp_794 = _pt_tmp_795 / _pt_tmp_800
    _pt_tmp_793 = 253312.50000000006 + _pt_tmp_794
    del _pt_tmp_795, _pt_tmp_800
    _pt_tmp_792 = _pt_tmp_793 - _actx_in_1_0_energy_0
    del _pt_tmp_794
    _pt_tmp_781 = _pt_tmp_782 * _pt_tmp_792
    del _pt_tmp_793
    _pt_tmp_0 = _pt_tmp_1 + _pt_tmp_781
    del _pt_tmp_792
    _pt_tmp_809 = 0 * _actx_in_1_0_momentum_0_0
    del _pt_tmp_1, _pt_tmp_781
    _pt_tmp_808 = _pt_tmp_809 - _actx_in_1_0_momentum_0_0
    _pt_tmp_811 = 0 * _actx_in_1_0_momentum_1_0
    del _pt_tmp_809
    _pt_tmp_810 = _pt_tmp_811 - _actx_in_1_0_momentum_1_0
    _pt_tmp_813 = 0 * _actx_in_1_0_momentum_2_0
    del _pt_tmp_811
    _pt_tmp_812 = _pt_tmp_813 - _actx_in_1_0_momentum_2_0
    _pt_tmp_814 = actx.np.stack([_pt_tmp_808, _pt_tmp_810, _pt_tmp_812], axis=0)
    del _pt_tmp_813
    _pt_tmp_807 = actx.einsum(
        "ijk, jlm, ikm -> kl", _pt_tmp_9, _pt_data_2, _pt_tmp_814
    )
    del _pt_tmp_808, _pt_tmp_810, _pt_tmp_812
    _pt_tmp_807 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_807)
    del _pt_tmp_814
    _pt_tmp_827 = 0 * _pt_tmp_40
    _pt_tmp_828 = 0 * _pt_tmp_42
    _pt_tmp_826 = _pt_tmp_827 + _pt_tmp_828
    _pt_tmp_825 = _pt_tmp_826 / 2
    del _pt_tmp_827, _pt_tmp_828
    _pt_tmp_824 = (
        _pt_tmp_825 * _pt_data_11
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_825, _in1=_pt_data_11)["out"]
    )
    del _pt_tmp_826
    _pt_tmp_832 = 0 * _pt_tmp_94
    del _pt_tmp_825
    _pt_tmp_833 = 0 * _pt_tmp_96
    _pt_tmp_831 = _pt_tmp_832 + _pt_tmp_833
    _pt_tmp_830 = _pt_tmp_831 / 2
    del _pt_tmp_832, _pt_tmp_833
    _pt_tmp_829 = (
        _pt_tmp_830 * _pt_data_15
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_830, _in1=_pt_data_15)["out"]
    )
    del _pt_tmp_831
    _pt_tmp_823 = _pt_tmp_824 + _pt_tmp_829
    del _pt_tmp_830
    _pt_tmp_837 = 0 * _pt_tmp_127
    del _pt_tmp_824, _pt_tmp_829
    _pt_tmp_838 = 0 * _pt_tmp_129
    _pt_tmp_836 = _pt_tmp_837 + _pt_tmp_838
    _pt_tmp_835 = _pt_tmp_836 / 2
    del _pt_tmp_837, _pt_tmp_838
    _pt_tmp_834 = (
        _pt_tmp_835 * _pt_data_16
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_835, _in1=_pt_data_16)["out"]
    )
    del _pt_tmp_836
    _pt_tmp_822 = _pt_tmp_823 + _pt_tmp_834
    del _pt_tmp_835
    _pt_tmp_821 = (
        _pt_tmp_822[_pt_tmp_48, _pt_tmp_49]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_822, in_1=_pt_tmp_48, in_2=_pt_tmp_49
        )["out"]
    )
    del _pt_tmp_823, _pt_tmp_834
    _pt_tmp_821 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_821)
    del _pt_tmp_822
    _pt_tmp_821 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_821)
    _pt_tmp_820 = 0 + _pt_tmp_821
    _pt_tmp_819 = 0 + _pt_tmp_820
    del _pt_tmp_821
    _pt_tmp_818 = 0 + _pt_tmp_819
    del _pt_tmp_820
    _pt_tmp_848 = (
        _pt_tmp_42 * _pt_data_11
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_42, _in1=_pt_data_11)["out"]
    )
    del _pt_tmp_819
    _pt_tmp_849 = (
        _pt_tmp_96 * _pt_data_15
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_96, _in1=_pt_data_15)["out"]
    )
    _pt_tmp_847 = _pt_tmp_848 + _pt_tmp_849
    _pt_tmp_850 = (
        _pt_tmp_129 * _pt_data_16
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_129, _in1=_pt_data_16)["out"]
    )
    del _pt_tmp_848, _pt_tmp_849
    _pt_tmp_846 = _pt_tmp_847 + _pt_tmp_850
    _pt_tmp_853 = (
        _pt_tmp_40 * _pt_data_11
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_40, _in1=_pt_data_11)["out"]
    )
    del _pt_tmp_847, _pt_tmp_850
    _pt_tmp_854 = (
        _pt_tmp_94 * _pt_data_15
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_94, _in1=_pt_data_15)["out"]
    )
    _pt_tmp_852 = _pt_tmp_853 + _pt_tmp_854
    _pt_tmp_855 = (
        _pt_tmp_127 * _pt_data_16
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_127, _in1=_pt_data_16)["out"]
    )
    del _pt_tmp_853, _pt_tmp_854
    _pt_tmp_851 = _pt_tmp_852 + _pt_tmp_855
    _pt_tmp_845 = _pt_tmp_846 + _pt_tmp_851
    del _pt_tmp_852, _pt_tmp_855
    _pt_tmp_857 = _pt_tmp_68 - _pt_tmp_66
    del _pt_tmp_846, _pt_tmp_851
    _pt_tmp_856 = _pt_tmp_648 * _pt_tmp_857
    _pt_tmp_844 = _pt_tmp_845 + _pt_tmp_856
    del _pt_tmp_857
    _pt_tmp_843 = _pt_tmp_844 / 2
    del _pt_tmp_845, _pt_tmp_856
    _pt_tmp_842 = (
        _pt_tmp_843[_pt_tmp_48, _pt_tmp_49]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_843, in_1=_pt_tmp_48, in_2=_pt_tmp_49
        )["out"]
    )
    del _pt_tmp_844
    _pt_tmp_842 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_842)
    del _pt_tmp_843
    _pt_tmp_842 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_842)
    _pt_tmp_841 = 0 + _pt_tmp_842
    _pt_tmp_840 = 0 + _pt_tmp_841
    del _pt_tmp_842
    _pt_tmp_839 = _pt_tmp_840 + 0
    del _pt_tmp_841
    _pt_tmp_817 = _pt_tmp_818 - _pt_tmp_839
    del _pt_tmp_840
    _pt_tmp_816 = actx.np.reshape(_pt_tmp_817, (4, 48000, 15))
    del _pt_tmp_818, _pt_tmp_839
    _pt_tmp_816 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_816)
    del _pt_tmp_817
    _pt_tmp_815 = actx.einsum(
        "ijk, jl, jlk -> li", _pt_data_3, _pt_tmp_30, _pt_tmp_816
    )
    _pt_tmp_815 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_815)
    del _pt_tmp_816
    _pt_tmp_806 = _pt_tmp_807 - _pt_tmp_815
    _pt_tmp_805 = actx.einsum("i, jk, ik -> ij", _pt_tmp_5, _pt_data_0, _pt_tmp_806)
    del _pt_tmp_807, _pt_tmp_815
    _pt_tmp_805 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_805)
    del _pt_tmp_806
    _pt_tmp_804 = -1 * _pt_tmp_805
    _pt_tmp_858 = 0 * _pt_tmp_804
    del _pt_tmp_805
    _pt_tmp_803 = _pt_tmp_804 + _pt_tmp_858
    _pt_tmp_864 = _pt_tmp_682 * _pt_tmp_52
    del _pt_tmp_804, _pt_tmp_858
    _pt_tmp_865 = _pt_tmp_682 * _pt_tmp_100
    del _pt_tmp_52
    _pt_tmp_866 = _pt_tmp_682 * _pt_tmp_133
    del _pt_tmp_100
    _pt_tmp_867 = actx.np.stack([_pt_tmp_864, _pt_tmp_865, _pt_tmp_866], axis=0)
    del _pt_tmp_133
    _pt_tmp_863 = actx.einsum(
        "ijk, jlm, ikm -> kl", _pt_tmp_9, _pt_data_2, _pt_tmp_867
    )
    _pt_tmp_863 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_863)
    del _pt_tmp_867
    _pt_tmp_882 = (
        _pt_tmp_864[_pt_tmp_44, _pt_tmp_45]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_pt_tmp_864, in_1=_pt_tmp_44, in_2=_pt_tmp_45
        )["out"]
    )
    _pt_tmp_882 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_882)
    del _pt_tmp_864
    _pt_tmp_882 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_882)
    _pt_tmp_881 = 0 + _pt_tmp_882
    _pt_tmp_880 = (
        _pt_tmp_881[_pt_tmp_46, _pt_tmp_47]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_881, in_1=_pt_tmp_46, in_2=_pt_tmp_47
        )["out"]
    )
    del _pt_tmp_882
    _pt_tmp_880 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_880)
    _pt_tmp_880 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_880)
    _pt_tmp_879 = 0 + _pt_tmp_880
    _pt_tmp_878 = _pt_tmp_879 + _pt_tmp_881
    del _pt_tmp_880
    _pt_tmp_877 = _pt_tmp_878 / 2
    del _pt_tmp_879, _pt_tmp_881
    _pt_tmp_876 = (
        _pt_tmp_877 * _pt_data_11
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_877, _in1=_pt_data_11)["out"]
    )
    del _pt_tmp_878
    _pt_tmp_889 = (
        _pt_tmp_865[_pt_tmp_44, _pt_tmp_45]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_pt_tmp_865, in_1=_pt_tmp_44, in_2=_pt_tmp_45
        )["out"]
    )
    del _pt_tmp_877
    _pt_tmp_889 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_889)
    del _pt_tmp_865
    _pt_tmp_889 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_889)
    _pt_tmp_888 = 0 + _pt_tmp_889
    _pt_tmp_887 = (
        _pt_tmp_888[_pt_tmp_46, _pt_tmp_47]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_888, in_1=_pt_tmp_46, in_2=_pt_tmp_47
        )["out"]
    )
    del _pt_tmp_889
    _pt_tmp_887 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_887)
    _pt_tmp_887 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_887)
    _pt_tmp_886 = 0 + _pt_tmp_887
    _pt_tmp_885 = _pt_tmp_886 + _pt_tmp_888
    del _pt_tmp_887
    _pt_tmp_884 = _pt_tmp_885 / 2
    del _pt_tmp_886, _pt_tmp_888
    _pt_tmp_883 = (
        _pt_tmp_884 * _pt_data_15
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_884, _in1=_pt_data_15)["out"]
    )
    del _pt_tmp_885
    _pt_tmp_875 = _pt_tmp_876 + _pt_tmp_883
    del _pt_tmp_884
    _pt_tmp_896 = (
        _pt_tmp_866[_pt_tmp_44, _pt_tmp_45]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_pt_tmp_866, in_1=_pt_tmp_44, in_2=_pt_tmp_45
        )["out"]
    )
    del _pt_tmp_876, _pt_tmp_883
    _pt_tmp_896 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_896)
    del _pt_tmp_866
    _pt_tmp_896 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_896)
    _pt_tmp_895 = 0 + _pt_tmp_896
    _pt_tmp_894 = (
        _pt_tmp_895[_pt_tmp_46, _pt_tmp_47]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_895, in_1=_pt_tmp_46, in_2=_pt_tmp_47
        )["out"]
    )
    del _pt_tmp_896
    _pt_tmp_894 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_894)
    _pt_tmp_894 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_894)
    _pt_tmp_893 = 0 + _pt_tmp_894
    _pt_tmp_892 = _pt_tmp_893 + _pt_tmp_895
    del _pt_tmp_894
    _pt_tmp_891 = _pt_tmp_892 / 2
    del _pt_tmp_893, _pt_tmp_895
    _pt_tmp_890 = (
        _pt_tmp_891 * _pt_data_16
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_891, _in1=_pt_data_16)["out"]
    )
    del _pt_tmp_892
    _pt_tmp_874 = _pt_tmp_875 + _pt_tmp_890
    del _pt_tmp_891
    _pt_tmp_873 = (
        _pt_tmp_874[_pt_tmp_48, _pt_tmp_49]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_874, in_1=_pt_tmp_48, in_2=_pt_tmp_49
        )["out"]
    )
    del _pt_tmp_875, _pt_tmp_890
    _pt_tmp_873 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_873)
    del _pt_tmp_874
    _pt_tmp_873 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_873)
    _pt_tmp_872 = 0 + _pt_tmp_873
    _pt_tmp_871 = 0 + _pt_tmp_872
    del _pt_tmp_873
    _pt_tmp_870 = _pt_tmp_871 + 0
    del _pt_tmp_872
    _pt_tmp_869 = actx.np.reshape(_pt_tmp_870, (4, 48000, 15))
    del _pt_tmp_871
    _pt_tmp_869 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_869)
    del _pt_tmp_870
    _pt_tmp_868 = actx.einsum(
        "ijk, jl, jlk -> li", _pt_data_3, _pt_tmp_30, _pt_tmp_869
    )
    _pt_tmp_868 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_868)
    del _pt_tmp_869
    _pt_tmp_862 = _pt_tmp_863 - _pt_tmp_868
    _pt_tmp_861 = actx.einsum("i, jk, ik -> ij", _pt_tmp_5, _pt_data_0, _pt_tmp_862)
    del _pt_tmp_863, _pt_tmp_868
    _pt_tmp_861 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_861)
    del _pt_tmp_862
    _pt_tmp_860 = -1 * _pt_tmp_861
    _pt_tmp_859 = -1 * _pt_tmp_860
    del _pt_tmp_861
    _pt_tmp_802 = _pt_tmp_803 + _pt_tmp_859
    del _pt_tmp_860
    _pt_tmp_898 = _pt_tmp_799 - _actx_in_1_0_mass_0
    del _pt_tmp_803, _pt_tmp_859
    _pt_tmp_897 = _pt_tmp_782 * _pt_tmp_898
    del _pt_tmp_799
    _pt_tmp_801 = _pt_tmp_802 + _pt_tmp_897
    del _pt_tmp_898
    _pt_tmp_908 = _actx_in_1_0_mass_0 * _pt_tmp_689
    del _pt_tmp_802, _pt_tmp_897
    _pt_tmp_909 = 1.0 * _pt_tmp_262
    del _pt_tmp_689
    _pt_tmp_907 = _pt_tmp_908 + _pt_tmp_909
    _pt_tmp_906 = _pt_tmp_16 - _pt_tmp_907
    del _pt_tmp_908
    _pt_tmp_913 = _pt_tmp_51 * _pt_tmp_99
    del _pt_tmp_16, _pt_tmp_907
    _pt_tmp_912 = _actx_in_1_0_mass_0 * _pt_tmp_913
    _pt_tmp_914 = 0.0 * _pt_tmp_262
    del _pt_tmp_913
    _pt_tmp_911 = _pt_tmp_912 + _pt_tmp_914
    del _pt_tmp_262
    _pt_tmp_910 = _pt_tmp_145 - _pt_tmp_911
    del _pt_tmp_912
    _pt_tmp_918 = _pt_tmp_51 * _pt_tmp_132
    del _pt_tmp_145, _pt_tmp_911
    _pt_tmp_917 = _actx_in_1_0_mass_0 * _pt_tmp_918
    _pt_tmp_916 = _pt_tmp_917 + _pt_tmp_914
    del _pt_tmp_918
    _pt_tmp_915 = _pt_tmp_178 - _pt_tmp_916
    del _pt_tmp_917
    _pt_tmp_919 = actx.np.stack([_pt_tmp_906, _pt_tmp_910, _pt_tmp_915], axis=0)
    del _pt_tmp_178, _pt_tmp_916
    _pt_tmp_905 = actx.einsum(
        "ijk, jlm, ikm -> kl", _pt_tmp_9, _pt_data_2, _pt_tmp_919
    )
    del _pt_tmp_906, _pt_tmp_910, _pt_tmp_915
    _pt_tmp_905 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_905)
    del _pt_tmp_919
    _pt_tmp_931 = _pt_tmp_369 + _pt_tmp_468
    _pt_tmp_930 = _pt_tmp_931 / 2
    del _pt_tmp_369, _pt_tmp_468
    _pt_tmp_929 = (
        _pt_tmp_930 * _pt_data_11
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_930, _in1=_pt_data_11)["out"]
    )
    del _pt_tmp_931
    _pt_tmp_934 = _pt_tmp_420 + _pt_tmp_495
    del _pt_tmp_930
    _pt_tmp_933 = _pt_tmp_934 / 2
    del _pt_tmp_420, _pt_tmp_495
    _pt_tmp_932 = (
        _pt_tmp_933 * _pt_data_15
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_933, _in1=_pt_data_15)["out"]
    )
    del _pt_tmp_934
    _pt_tmp_928 = _pt_tmp_929 + _pt_tmp_932
    del _pt_tmp_933
    _pt_tmp_937 = _pt_tmp_439 + _pt_tmp_506
    del _pt_tmp_929, _pt_tmp_932
    _pt_tmp_936 = _pt_tmp_937 / 2
    del _pt_tmp_439, _pt_tmp_506
    _pt_tmp_935 = (
        _pt_tmp_936 * _pt_data_16
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_936, _in1=_pt_data_16)["out"]
    )
    del _pt_tmp_937
    _pt_tmp_927 = _pt_tmp_928 + _pt_tmp_935
    del _pt_tmp_936
    _pt_tmp_926 = (
        _pt_tmp_927[_pt_tmp_48, _pt_tmp_49]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_927, in_1=_pt_tmp_48, in_2=_pt_tmp_49
        )["out"]
    )
    del _pt_tmp_928, _pt_tmp_935
    _pt_tmp_926 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_926)
    del _pt_tmp_927
    _pt_tmp_926 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_926)
    _pt_tmp_925 = 0 + _pt_tmp_926
    _pt_tmp_924 = 0 + _pt_tmp_925
    del _pt_tmp_926
    _pt_tmp_923 = 0 + _pt_tmp_924
    del _pt_tmp_925
    _pt_tmp_949 = _pt_tmp_68 * _pt_tmp_655
    del _pt_tmp_924
    _pt_tmp_950 = 1.0 * _pt_tmp_632
    del _pt_tmp_655
    _pt_tmp_948 = _pt_tmp_949 + _pt_tmp_950
    _pt_tmp_947 = (
        _pt_tmp_948 * _pt_data_11
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_948, _in1=_pt_data_11)["out"]
    )
    del _pt_tmp_949
    _pt_tmp_954 = _pt_tmp_477 * _pt_tmp_489
    del _pt_tmp_948
    _pt_tmp_953 = _pt_tmp_68 * _pt_tmp_954
    _pt_tmp_955 = 0.0 * _pt_tmp_632
    del _pt_tmp_954
    _pt_tmp_952 = _pt_tmp_953 + _pt_tmp_955
    del _pt_tmp_632
    _pt_tmp_951 = (
        _pt_tmp_952 * _pt_data_15
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_952, _in1=_pt_data_15)["out"]
    )
    del _pt_tmp_953
    _pt_tmp_946 = _pt_tmp_947 + _pt_tmp_951
    del _pt_tmp_952
    _pt_tmp_959 = _pt_tmp_477 * _pt_tmp_493
    del _pt_tmp_947, _pt_tmp_951
    _pt_tmp_958 = _pt_tmp_68 * _pt_tmp_959
    _pt_tmp_957 = _pt_tmp_958 + _pt_tmp_955
    del _pt_tmp_959
    _pt_tmp_956 = (
        _pt_tmp_957 * _pt_data_16
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_957, _in1=_pt_data_16)["out"]
    )
    del _pt_tmp_958
    _pt_tmp_945 = _pt_tmp_946 + _pt_tmp_956
    del _pt_tmp_957
    _pt_tmp_964 = _pt_tmp_66 * _pt_tmp_666
    del _pt_tmp_946, _pt_tmp_956
    _pt_tmp_965 = 1.0 * _pt_tmp_642
    del _pt_tmp_666
    _pt_tmp_963 = _pt_tmp_964 + _pt_tmp_965
    _pt_tmp_962 = (
        _pt_tmp_963 * _pt_data_11
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_963, _in1=_pt_data_11)["out"]
    )
    del _pt_tmp_964
    _pt_tmp_969 = _pt_tmp_382 * _pt_tmp_402
    del _pt_tmp_963
    _pt_tmp_968 = _pt_tmp_66 * _pt_tmp_969
    _pt_tmp_970 = 0.0 * _pt_tmp_642
    del _pt_tmp_969
    _pt_tmp_967 = _pt_tmp_968 + _pt_tmp_970
    del _pt_tmp_642
    _pt_tmp_966 = (
        _pt_tmp_967 * _pt_data_15
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_967, _in1=_pt_data_15)["out"]
    )
    del _pt_tmp_968
    _pt_tmp_961 = _pt_tmp_962 + _pt_tmp_966
    del _pt_tmp_967
    _pt_tmp_974 = _pt_tmp_382 * _pt_tmp_414
    del _pt_tmp_962, _pt_tmp_966
    _pt_tmp_973 = _pt_tmp_66 * _pt_tmp_974
    _pt_tmp_972 = _pt_tmp_973 + _pt_tmp_970
    del _pt_tmp_974
    _pt_tmp_971 = (
        _pt_tmp_972 * _pt_data_16
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_972, _in1=_pt_data_16)["out"]
    )
    del _pt_tmp_973
    _pt_tmp_960 = _pt_tmp_961 + _pt_tmp_971
    del _pt_tmp_972
    _pt_tmp_944 = _pt_tmp_945 + _pt_tmp_960
    del _pt_tmp_961, _pt_tmp_971
    _pt_tmp_976 = _pt_tmp_42 - _pt_tmp_40
    del _pt_tmp_945, _pt_tmp_960
    _pt_tmp_975 = _pt_tmp_648 * _pt_tmp_976
    del _pt_tmp_40, _pt_tmp_42
    _pt_tmp_943 = _pt_tmp_944 + _pt_tmp_975
    del _pt_tmp_976
    _pt_tmp_942 = _pt_tmp_943 / 2
    del _pt_tmp_944, _pt_tmp_975
    _pt_tmp_941 = (
        _pt_tmp_942[_pt_tmp_48, _pt_tmp_49]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_942, in_1=_pt_tmp_48, in_2=_pt_tmp_49
        )["out"]
    )
    del _pt_tmp_943
    _pt_tmp_941 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_941)
    del _pt_tmp_942
    _pt_tmp_941 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_941)
    _pt_tmp_940 = 0 + _pt_tmp_941
    _pt_tmp_939 = 0 + _pt_tmp_940
    del _pt_tmp_941
    _pt_tmp_938 = _pt_tmp_939 + 0
    del _pt_tmp_940
    _pt_tmp_922 = _pt_tmp_923 - _pt_tmp_938
    del _pt_tmp_939
    _pt_tmp_921 = actx.np.reshape(_pt_tmp_922, (4, 48000, 15))
    del _pt_tmp_923, _pt_tmp_938
    _pt_tmp_921 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_921)
    del _pt_tmp_922
    _pt_tmp_920 = actx.einsum(
        "ijk, jl, jlk -> li", _pt_data_3, _pt_tmp_30, _pt_tmp_921
    )
    _pt_tmp_920 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_920)
    del _pt_tmp_921
    _pt_tmp_904 = _pt_tmp_905 - _pt_tmp_920
    _pt_tmp_903 = actx.einsum("i, jk, ik -> ij", _pt_tmp_5, _pt_data_0, _pt_tmp_904)
    del _pt_tmp_905, _pt_tmp_920
    _pt_tmp_903 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_903)
    del _pt_tmp_904
    _pt_tmp_902 = -1 * _pt_tmp_903
    _pt_tmp_977 = 0 * _pt_tmp_902
    del _pt_tmp_903
    _pt_tmp_901 = _pt_tmp_902 + _pt_tmp_977
    _pt_tmp_983 = _pt_tmp_682 * _pt_tmp_24
    del _pt_tmp_902, _pt_tmp_977
    _pt_tmp_984 = _pt_tmp_682 * _pt_tmp_150
    del _pt_tmp_24
    _pt_tmp_985 = _pt_tmp_682 * _pt_tmp_183
    del _pt_tmp_150
    _pt_tmp_986 = actx.np.stack([_pt_tmp_983, _pt_tmp_984, _pt_tmp_985], axis=0)
    del _pt_tmp_183
    _pt_tmp_982 = actx.einsum(
        "ijk, jlm, ikm -> kl", _pt_tmp_9, _pt_data_2, _pt_tmp_986
    )
    _pt_tmp_982 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_982)
    del _pt_tmp_986
    _pt_tmp_1001 = (
        _pt_tmp_983[_pt_tmp_44, _pt_tmp_45]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_pt_tmp_983, in_1=_pt_tmp_44, in_2=_pt_tmp_45
        )["out"]
    )
    _pt_tmp_1001 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_1001)
    del _pt_tmp_983
    _pt_tmp_1001 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_1001)
    _pt_tmp_1000 = 0 + _pt_tmp_1001
    _pt_tmp_999 = (
        _pt_tmp_1000[_pt_tmp_46, _pt_tmp_47]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_1000, in_1=_pt_tmp_46, in_2=_pt_tmp_47
        )["out"]
    )
    del _pt_tmp_1001
    _pt_tmp_999 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_999)
    _pt_tmp_999 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_999)
    _pt_tmp_998 = 0 + _pt_tmp_999
    _pt_tmp_997 = _pt_tmp_998 + _pt_tmp_1000
    del _pt_tmp_999
    _pt_tmp_996 = _pt_tmp_997 / 2
    del _pt_tmp_1000, _pt_tmp_998
    _pt_tmp_995 = (
        _pt_tmp_996 * _pt_data_11
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_996, _in1=_pt_data_11)["out"]
    )
    del _pt_tmp_997
    _pt_tmp_1008 = (
        _pt_tmp_984[_pt_tmp_44, _pt_tmp_45]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_pt_tmp_984, in_1=_pt_tmp_44, in_2=_pt_tmp_45
        )["out"]
    )
    del _pt_tmp_996
    _pt_tmp_1008 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_1008)
    del _pt_tmp_984
    _pt_tmp_1008 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_1008)
    _pt_tmp_1007 = 0 + _pt_tmp_1008
    _pt_tmp_1006 = (
        _pt_tmp_1007[_pt_tmp_46, _pt_tmp_47]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_1007, in_1=_pt_tmp_46, in_2=_pt_tmp_47
        )["out"]
    )
    del _pt_tmp_1008
    _pt_tmp_1006 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_1006)
    _pt_tmp_1006 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_1006)
    _pt_tmp_1005 = 0 + _pt_tmp_1006
    _pt_tmp_1004 = _pt_tmp_1005 + _pt_tmp_1007
    del _pt_tmp_1006
    _pt_tmp_1003 = _pt_tmp_1004 / 2
    del _pt_tmp_1005, _pt_tmp_1007
    _pt_tmp_1002 = (
        _pt_tmp_1003 * _pt_data_15
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_1003, _in1=_pt_data_15)[
            "out"
        ]
    )
    del _pt_tmp_1004
    _pt_tmp_994 = _pt_tmp_995 + _pt_tmp_1002
    del _pt_tmp_1003
    _pt_tmp_1015 = (
        _pt_tmp_985[_pt_tmp_44, _pt_tmp_45]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_pt_tmp_985, in_1=_pt_tmp_44, in_2=_pt_tmp_45
        )["out"]
    )
    del _pt_tmp_1002, _pt_tmp_995
    _pt_tmp_1015 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_1015)
    del _pt_tmp_985
    _pt_tmp_1015 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_1015)
    _pt_tmp_1014 = 0 + _pt_tmp_1015
    _pt_tmp_1013 = (
        _pt_tmp_1014[_pt_tmp_46, _pt_tmp_47]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_1014, in_1=_pt_tmp_46, in_2=_pt_tmp_47
        )["out"]
    )
    del _pt_tmp_1015
    _pt_tmp_1013 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_1013)
    _pt_tmp_1013 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_1013)
    _pt_tmp_1012 = 0 + _pt_tmp_1013
    _pt_tmp_1011 = _pt_tmp_1012 + _pt_tmp_1014
    del _pt_tmp_1013
    _pt_tmp_1010 = _pt_tmp_1011 / 2
    del _pt_tmp_1012, _pt_tmp_1014
    _pt_tmp_1009 = (
        _pt_tmp_1010 * _pt_data_16
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_1010, _in1=_pt_data_16)[
            "out"
        ]
    )
    del _pt_tmp_1011
    _pt_tmp_993 = _pt_tmp_994 + _pt_tmp_1009
    del _pt_tmp_1010
    _pt_tmp_992 = (
        _pt_tmp_993[_pt_tmp_48, _pt_tmp_49]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_993, in_1=_pt_tmp_48, in_2=_pt_tmp_49
        )["out"]
    )
    del _pt_tmp_1009, _pt_tmp_994
    _pt_tmp_992 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_992)
    del _pt_tmp_993
    _pt_tmp_992 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_992)
    _pt_tmp_991 = 0 + _pt_tmp_992
    _pt_tmp_990 = 0 + _pt_tmp_991
    del _pt_tmp_992
    _pt_tmp_989 = _pt_tmp_990 + 0
    del _pt_tmp_991
    _pt_tmp_988 = actx.np.reshape(_pt_tmp_989, (4, 48000, 15))
    del _pt_tmp_990
    _pt_tmp_988 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_988)
    del _pt_tmp_989
    _pt_tmp_987 = actx.einsum(
        "ijk, jl, jlk -> li", _pt_data_3, _pt_tmp_30, _pt_tmp_988
    )
    _pt_tmp_987 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_987)
    del _pt_tmp_988
    _pt_tmp_981 = _pt_tmp_982 - _pt_tmp_987
    _pt_tmp_980 = actx.einsum("i, jk, ik -> ij", _pt_tmp_5, _pt_data_0, _pt_tmp_981)
    del _pt_tmp_982, _pt_tmp_987
    _pt_tmp_980 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_980)
    del _pt_tmp_981
    _pt_tmp_979 = -1 * _pt_tmp_980
    _pt_tmp_978 = -1 * _pt_tmp_979
    del _pt_tmp_980
    _pt_tmp_900 = _pt_tmp_901 + _pt_tmp_978
    del _pt_tmp_979
    _pt_tmp_1017 = _pt_tmp_798 - _actx_in_1_0_momentum_0_0
    del _pt_tmp_901, _pt_tmp_978
    _pt_tmp_1016 = _pt_tmp_782 * _pt_tmp_1017
    _pt_tmp_899 = _pt_tmp_900 + _pt_tmp_1016
    del _pt_tmp_1017
    _pt_tmp_1028 = _pt_tmp_99 * _pt_tmp_51
    del _pt_tmp_1016, _pt_tmp_900
    _pt_tmp_1027 = _actx_in_1_0_mass_0 * _pt_tmp_1028
    _pt_tmp_1026 = _pt_tmp_1027 + _pt_tmp_914
    del _pt_tmp_1028
    _pt_tmp_1025 = _pt_tmp_269 - _pt_tmp_1026
    del _pt_tmp_1027
    _pt_tmp_1031 = _actx_in_1_0_mass_0 * _pt_tmp_690
    del _pt_tmp_1026, _pt_tmp_269
    _pt_tmp_1030 = _pt_tmp_1031 + _pt_tmp_909
    del _pt_tmp_690
    _pt_tmp_1029 = _pt_tmp_273 - _pt_tmp_1030
    del _pt_tmp_1031
    _pt_tmp_1035 = _pt_tmp_99 * _pt_tmp_132
    del _pt_tmp_1030, _pt_tmp_273
    _pt_tmp_1034 = _actx_in_1_0_mass_0 * _pt_tmp_1035
    _pt_tmp_1033 = _pt_tmp_1034 + _pt_tmp_914
    del _pt_tmp_1035
    _pt_tmp_1032 = _pt_tmp_277 - _pt_tmp_1033
    del _pt_tmp_1034
    _pt_tmp_1036 = actx.np.stack([_pt_tmp_1025, _pt_tmp_1029, _pt_tmp_1032], axis=0)
    del _pt_tmp_1033, _pt_tmp_277
    _pt_tmp_1024 = actx.einsum(
        "ijk, jlm, ikm -> kl", _pt_tmp_9, _pt_data_2, _pt_tmp_1036
    )
    del _pt_tmp_1025, _pt_tmp_1029, _pt_tmp_1032
    _pt_tmp_1024 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_1024)
    del _pt_tmp_1036
    _pt_tmp_1048 = _pt_tmp_526 + _pt_tmp_561
    _pt_tmp_1047 = _pt_tmp_1048 / 2
    del _pt_tmp_526, _pt_tmp_561
    _pt_tmp_1046 = (
        _pt_tmp_1047 * _pt_data_11
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_1047, _in1=_pt_data_11)[
            "out"
        ]
    )
    del _pt_tmp_1048
    _pt_tmp_1051 = _pt_tmp_530 + _pt_tmp_565
    del _pt_tmp_1047
    _pt_tmp_1050 = _pt_tmp_1051 / 2
    del _pt_tmp_530, _pt_tmp_565
    _pt_tmp_1049 = (
        _pt_tmp_1050 * _pt_data_15
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_1050, _in1=_pt_data_15)[
            "out"
        ]
    )
    del _pt_tmp_1051
    _pt_tmp_1045 = _pt_tmp_1046 + _pt_tmp_1049
    del _pt_tmp_1050
    _pt_tmp_1054 = _pt_tmp_534 + _pt_tmp_569
    del _pt_tmp_1046, _pt_tmp_1049
    _pt_tmp_1053 = _pt_tmp_1054 / 2
    del _pt_tmp_534, _pt_tmp_569
    _pt_tmp_1052 = (
        _pt_tmp_1053 * _pt_data_16
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_1053, _in1=_pt_data_16)[
            "out"
        ]
    )
    del _pt_tmp_1054
    _pt_tmp_1044 = _pt_tmp_1045 + _pt_tmp_1052
    del _pt_tmp_1053
    _pt_tmp_1043 = (
        _pt_tmp_1044[_pt_tmp_48, _pt_tmp_49]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_1044, in_1=_pt_tmp_48, in_2=_pt_tmp_49
        )["out"]
    )
    del _pt_tmp_1045, _pt_tmp_1052
    _pt_tmp_1043 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_1043)
    del _pt_tmp_1044
    _pt_tmp_1043 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_1043)
    _pt_tmp_1042 = 0 + _pt_tmp_1043
    _pt_tmp_1041 = 0 + _pt_tmp_1042
    del _pt_tmp_1043
    _pt_tmp_1040 = 0 + _pt_tmp_1041
    del _pt_tmp_1042
    _pt_tmp_1067 = _pt_tmp_489 * _pt_tmp_477
    del _pt_tmp_1041
    _pt_tmp_1066 = _pt_tmp_68 * _pt_tmp_1067
    _pt_tmp_1065 = _pt_tmp_1066 + _pt_tmp_955
    del _pt_tmp_1067
    _pt_tmp_1064 = (
        _pt_tmp_1065 * _pt_data_11
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_1065, _in1=_pt_data_11)[
            "out"
        ]
    )
    del _pt_tmp_1066
    _pt_tmp_1070 = _pt_tmp_68 * _pt_tmp_656
    del _pt_tmp_1065
    _pt_tmp_1069 = _pt_tmp_1070 + _pt_tmp_950
    del _pt_tmp_656
    _pt_tmp_1068 = (
        _pt_tmp_1069 * _pt_data_15
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_1069, _in1=_pt_data_15)[
            "out"
        ]
    )
    del _pt_tmp_1070
    _pt_tmp_1063 = _pt_tmp_1064 + _pt_tmp_1068
    del _pt_tmp_1069
    _pt_tmp_1074 = _pt_tmp_489 * _pt_tmp_493
    del _pt_tmp_1064, _pt_tmp_1068
    _pt_tmp_1073 = _pt_tmp_68 * _pt_tmp_1074
    _pt_tmp_1072 = _pt_tmp_1073 + _pt_tmp_955
    del _pt_tmp_1074
    _pt_tmp_1071 = (
        _pt_tmp_1072 * _pt_data_16
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_1072, _in1=_pt_data_16)[
            "out"
        ]
    )
    del _pt_tmp_1073
    _pt_tmp_1062 = _pt_tmp_1063 + _pt_tmp_1071
    del _pt_tmp_1072
    _pt_tmp_1080 = _pt_tmp_402 * _pt_tmp_382
    del _pt_tmp_1063, _pt_tmp_1071
    _pt_tmp_1079 = _pt_tmp_66 * _pt_tmp_1080
    _pt_tmp_1078 = _pt_tmp_1079 + _pt_tmp_970
    del _pt_tmp_1080
    _pt_tmp_1077 = (
        _pt_tmp_1078 * _pt_data_11
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_1078, _in1=_pt_data_11)[
            "out"
        ]
    )
    del _pt_tmp_1079
    _pt_tmp_1083 = _pt_tmp_66 * _pt_tmp_667
    del _pt_tmp_1078
    _pt_tmp_1082 = _pt_tmp_1083 + _pt_tmp_965
    del _pt_tmp_667
    _pt_tmp_1081 = (
        _pt_tmp_1082 * _pt_data_15
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_1082, _in1=_pt_data_15)[
            "out"
        ]
    )
    del _pt_tmp_1083
    _pt_tmp_1076 = _pt_tmp_1077 + _pt_tmp_1081
    del _pt_tmp_1082
    _pt_tmp_1087 = _pt_tmp_402 * _pt_tmp_414
    del _pt_tmp_1077, _pt_tmp_1081
    _pt_tmp_1086 = _pt_tmp_66 * _pt_tmp_1087
    _pt_tmp_1085 = _pt_tmp_1086 + _pt_tmp_970
    del _pt_tmp_1087
    _pt_tmp_1084 = (
        _pt_tmp_1085 * _pt_data_16
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_1085, _in1=_pt_data_16)[
            "out"
        ]
    )
    del _pt_tmp_1086
    _pt_tmp_1075 = _pt_tmp_1076 + _pt_tmp_1084
    del _pt_tmp_1085
    _pt_tmp_1061 = _pt_tmp_1062 + _pt_tmp_1075
    del _pt_tmp_1076, _pt_tmp_1084
    _pt_tmp_1089 = _pt_tmp_96 - _pt_tmp_94
    del _pt_tmp_1062, _pt_tmp_1075
    _pt_tmp_1088 = _pt_tmp_648 * _pt_tmp_1089
    del _pt_tmp_94, _pt_tmp_96
    _pt_tmp_1060 = _pt_tmp_1061 + _pt_tmp_1088
    del _pt_tmp_1089
    _pt_tmp_1059 = _pt_tmp_1060 / 2
    del _pt_tmp_1061, _pt_tmp_1088
    _pt_tmp_1058 = (
        _pt_tmp_1059[_pt_tmp_48, _pt_tmp_49]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_1059, in_1=_pt_tmp_48, in_2=_pt_tmp_49
        )["out"]
    )
    del _pt_tmp_1060
    _pt_tmp_1058 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_1058)
    del _pt_tmp_1059
    _pt_tmp_1058 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_1058)
    _pt_tmp_1057 = 0 + _pt_tmp_1058
    _pt_tmp_1056 = 0 + _pt_tmp_1057
    del _pt_tmp_1058
    _pt_tmp_1055 = _pt_tmp_1056 + 0
    del _pt_tmp_1057
    _pt_tmp_1039 = _pt_tmp_1040 - _pt_tmp_1055
    del _pt_tmp_1056
    _pt_tmp_1038 = actx.np.reshape(_pt_tmp_1039, (4, 48000, 15))
    del _pt_tmp_1040, _pt_tmp_1055
    _pt_tmp_1038 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_1038)
    del _pt_tmp_1039
    _pt_tmp_1037 = actx.einsum(
        "ijk, jl, jlk -> li", _pt_data_3, _pt_tmp_30, _pt_tmp_1038
    )
    _pt_tmp_1037 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_1037)
    del _pt_tmp_1038
    _pt_tmp_1023 = _pt_tmp_1024 - _pt_tmp_1037
    _pt_tmp_1022 = actx.einsum(
        "i, jk, ik -> ij", _pt_tmp_5, _pt_data_0, _pt_tmp_1023
    )
    del _pt_tmp_1024, _pt_tmp_1037
    _pt_tmp_1022 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_1022)
    del _pt_tmp_1023
    _pt_tmp_1021 = -1 * _pt_tmp_1022
    _pt_tmp_1090 = 0 * _pt_tmp_1021
    del _pt_tmp_1022
    _pt_tmp_1020 = _pt_tmp_1021 + _pt_tmp_1090
    _pt_tmp_1096 = _pt_tmp_682 * _pt_tmp_164
    del _pt_tmp_1021, _pt_tmp_1090
    _pt_tmp_1097 = _pt_tmp_682 * _pt_tmp_80
    del _pt_tmp_164
    _pt_tmp_1098 = _pt_tmp_682 * _pt_tmp_282
    del _pt_tmp_80
    _pt_tmp_1099 = actx.np.stack([_pt_tmp_1096, _pt_tmp_1097, _pt_tmp_1098], axis=0)
    del _pt_tmp_282
    _pt_tmp_1095 = actx.einsum(
        "ijk, jlm, ikm -> kl", _pt_tmp_9, _pt_data_2, _pt_tmp_1099
    )
    _pt_tmp_1095 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_1095)
    del _pt_tmp_1099
    _pt_tmp_1114 = (
        _pt_tmp_1096[_pt_tmp_44, _pt_tmp_45]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_pt_tmp_1096, in_1=_pt_tmp_44, in_2=_pt_tmp_45
        )["out"]
    )
    _pt_tmp_1114 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_1114)
    del _pt_tmp_1096
    _pt_tmp_1114 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_1114)
    _pt_tmp_1113 = 0 + _pt_tmp_1114
    _pt_tmp_1112 = (
        _pt_tmp_1113[_pt_tmp_46, _pt_tmp_47]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_1113, in_1=_pt_tmp_46, in_2=_pt_tmp_47
        )["out"]
    )
    del _pt_tmp_1114
    _pt_tmp_1112 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_1112)
    _pt_tmp_1112 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_1112)
    _pt_tmp_1111 = 0 + _pt_tmp_1112
    _pt_tmp_1110 = _pt_tmp_1111 + _pt_tmp_1113
    del _pt_tmp_1112
    _pt_tmp_1109 = _pt_tmp_1110 / 2
    del _pt_tmp_1111, _pt_tmp_1113
    _pt_tmp_1108 = (
        _pt_tmp_1109 * _pt_data_11
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_1109, _in1=_pt_data_11)[
            "out"
        ]
    )
    del _pt_tmp_1110
    _pt_tmp_1121 = (
        _pt_tmp_1097[_pt_tmp_44, _pt_tmp_45]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_pt_tmp_1097, in_1=_pt_tmp_44, in_2=_pt_tmp_45
        )["out"]
    )
    del _pt_tmp_1109
    _pt_tmp_1121 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_1121)
    del _pt_tmp_1097
    _pt_tmp_1121 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_1121)
    _pt_tmp_1120 = 0 + _pt_tmp_1121
    _pt_tmp_1119 = (
        _pt_tmp_1120[_pt_tmp_46, _pt_tmp_47]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_1120, in_1=_pt_tmp_46, in_2=_pt_tmp_47
        )["out"]
    )
    del _pt_tmp_1121
    _pt_tmp_1119 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_1119)
    _pt_tmp_1119 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_1119)
    _pt_tmp_1118 = 0 + _pt_tmp_1119
    _pt_tmp_1117 = _pt_tmp_1118 + _pt_tmp_1120
    del _pt_tmp_1119
    _pt_tmp_1116 = _pt_tmp_1117 / 2
    del _pt_tmp_1118, _pt_tmp_1120
    _pt_tmp_1115 = (
        _pt_tmp_1116 * _pt_data_15
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_1116, _in1=_pt_data_15)[
            "out"
        ]
    )
    del _pt_tmp_1117
    _pt_tmp_1107 = _pt_tmp_1108 + _pt_tmp_1115
    del _pt_tmp_1116
    _pt_tmp_1128 = (
        _pt_tmp_1098[_pt_tmp_44, _pt_tmp_45]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_pt_tmp_1098, in_1=_pt_tmp_44, in_2=_pt_tmp_45
        )["out"]
    )
    del _pt_tmp_1108, _pt_tmp_1115
    _pt_tmp_1128 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_1128)
    del _pt_tmp_1098
    _pt_tmp_1128 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_1128)
    _pt_tmp_1127 = 0 + _pt_tmp_1128
    _pt_tmp_1126 = (
        _pt_tmp_1127[_pt_tmp_46, _pt_tmp_47]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_1127, in_1=_pt_tmp_46, in_2=_pt_tmp_47
        )["out"]
    )
    del _pt_tmp_1128
    _pt_tmp_1126 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_1126)
    _pt_tmp_1126 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_1126)
    _pt_tmp_1125 = 0 + _pt_tmp_1126
    _pt_tmp_1124 = _pt_tmp_1125 + _pt_tmp_1127
    del _pt_tmp_1126
    _pt_tmp_1123 = _pt_tmp_1124 / 2
    del _pt_tmp_1125, _pt_tmp_1127
    _pt_tmp_1122 = (
        _pt_tmp_1123 * _pt_data_16
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_1123, _in1=_pt_data_16)[
            "out"
        ]
    )
    del _pt_tmp_1124
    _pt_tmp_1106 = _pt_tmp_1107 + _pt_tmp_1122
    del _pt_tmp_1123
    _pt_tmp_1105 = (
        _pt_tmp_1106[_pt_tmp_48, _pt_tmp_49]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_1106, in_1=_pt_tmp_48, in_2=_pt_tmp_49
        )["out"]
    )
    del _pt_tmp_1107, _pt_tmp_1122
    _pt_tmp_1105 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_1105)
    del _pt_tmp_1106
    _pt_tmp_1105 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_1105)
    _pt_tmp_1104 = 0 + _pt_tmp_1105
    _pt_tmp_1103 = 0 + _pt_tmp_1104
    del _pt_tmp_1105
    _pt_tmp_1102 = _pt_tmp_1103 + 0
    del _pt_tmp_1104
    _pt_tmp_1101 = actx.np.reshape(_pt_tmp_1102, (4, 48000, 15))
    del _pt_tmp_1103
    _pt_tmp_1101 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_1101)
    del _pt_tmp_1102
    _pt_tmp_1100 = actx.einsum(
        "ijk, jl, jlk -> li", _pt_data_3, _pt_tmp_30, _pt_tmp_1101
    )
    _pt_tmp_1100 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_1100)
    del _pt_tmp_1101
    _pt_tmp_1094 = _pt_tmp_1095 - _pt_tmp_1100
    _pt_tmp_1093 = actx.einsum(
        "i, jk, ik -> ij", _pt_tmp_5, _pt_data_0, _pt_tmp_1094
    )
    del _pt_tmp_1095, _pt_tmp_1100
    _pt_tmp_1093 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_1093)
    del _pt_tmp_1094
    _pt_tmp_1092 = -1 * _pt_tmp_1093
    _pt_tmp_1091 = -1 * _pt_tmp_1092
    del _pt_tmp_1093
    _pt_tmp_1019 = _pt_tmp_1020 + _pt_tmp_1091
    del _pt_tmp_1092
    _pt_tmp_1130 = _pt_tmp_798 - _actx_in_1_0_momentum_1_0
    del _pt_tmp_1020, _pt_tmp_1091
    _pt_tmp_1129 = _pt_tmp_782 * _pt_tmp_1130
    _pt_tmp_1018 = _pt_tmp_1019 + _pt_tmp_1129
    del _pt_tmp_1130
    _pt_tmp_1141 = _pt_tmp_132 * _pt_tmp_51
    del _pt_tmp_1019, _pt_tmp_1129
    _pt_tmp_1140 = _actx_in_1_0_mass_0 * _pt_tmp_1141
    del _pt_tmp_51
    _pt_tmp_1139 = _pt_tmp_1140 + _pt_tmp_914
    del _pt_tmp_1141
    _pt_tmp_1138 = _pt_tmp_327 - _pt_tmp_1139
    del _pt_tmp_1140
    _pt_tmp_1145 = _pt_tmp_132 * _pt_tmp_99
    del _pt_tmp_1139, _pt_tmp_327
    _pt_tmp_1144 = _actx_in_1_0_mass_0 * _pt_tmp_1145
    del _pt_tmp_132, _pt_tmp_99
    _pt_tmp_1143 = _pt_tmp_1144 + _pt_tmp_914
    del _pt_tmp_1145
    _pt_tmp_1142 = _pt_tmp_331 - _pt_tmp_1143
    del _pt_tmp_1144, _pt_tmp_914
    _pt_tmp_1148 = _actx_in_1_0_mass_0 * _pt_tmp_691
    del _pt_tmp_1143, _pt_tmp_331
    _pt_tmp_1147 = _pt_tmp_1148 + _pt_tmp_909
    del _pt_tmp_691
    _pt_tmp_1146 = _pt_tmp_335 - _pt_tmp_1147
    del _pt_tmp_1148, _pt_tmp_909
    _pt_tmp_1149 = actx.np.stack([_pt_tmp_1138, _pt_tmp_1142, _pt_tmp_1146], axis=0)
    del _pt_tmp_1147, _pt_tmp_335
    _pt_tmp_1137 = actx.einsum(
        "ijk, jlm, ikm -> kl", _pt_tmp_9, _pt_data_2, _pt_tmp_1149
    )
    del _pt_tmp_1138, _pt_tmp_1142, _pt_tmp_1146
    _pt_tmp_1137 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_1137)
    del _pt_tmp_1149
    _pt_tmp_1161 = _pt_tmp_587 + _pt_tmp_608
    _pt_tmp_1160 = _pt_tmp_1161 / 2
    del _pt_tmp_587, _pt_tmp_608
    _pt_tmp_1159 = (
        _pt_tmp_1160 * _pt_data_11
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_1160, _in1=_pt_data_11)[
            "out"
        ]
    )
    del _pt_tmp_1161
    _pt_tmp_1164 = _pt_tmp_591 + _pt_tmp_612
    del _pt_tmp_1160
    _pt_tmp_1163 = _pt_tmp_1164 / 2
    del _pt_tmp_591, _pt_tmp_612
    _pt_tmp_1162 = (
        _pt_tmp_1163 * _pt_data_15
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_1163, _in1=_pt_data_15)[
            "out"
        ]
    )
    del _pt_tmp_1164
    _pt_tmp_1158 = _pt_tmp_1159 + _pt_tmp_1162
    del _pt_tmp_1163
    _pt_tmp_1167 = _pt_tmp_595 + _pt_tmp_616
    del _pt_tmp_1159, _pt_tmp_1162
    _pt_tmp_1166 = _pt_tmp_1167 / 2
    del _pt_tmp_595, _pt_tmp_616
    _pt_tmp_1165 = (
        _pt_tmp_1166 * _pt_data_16
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_1166, _in1=_pt_data_16)[
            "out"
        ]
    )
    del _pt_tmp_1167
    _pt_tmp_1157 = _pt_tmp_1158 + _pt_tmp_1165
    del _pt_tmp_1166
    _pt_tmp_1156 = (
        _pt_tmp_1157[_pt_tmp_48, _pt_tmp_49]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_1157, in_1=_pt_tmp_48, in_2=_pt_tmp_49
        )["out"]
    )
    del _pt_tmp_1158, _pt_tmp_1165
    _pt_tmp_1156 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_1156)
    del _pt_tmp_1157
    _pt_tmp_1156 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_1156)
    _pt_tmp_1155 = 0 + _pt_tmp_1156
    _pt_tmp_1154 = 0 + _pt_tmp_1155
    del _pt_tmp_1156
    _pt_tmp_1153 = 0 + _pt_tmp_1154
    del _pt_tmp_1155
    _pt_tmp_1180 = _pt_tmp_493 * _pt_tmp_477
    del _pt_tmp_1154
    _pt_tmp_1179 = _pt_tmp_68 * _pt_tmp_1180
    del _pt_tmp_477
    _pt_tmp_1178 = _pt_tmp_1179 + _pt_tmp_955
    del _pt_tmp_1180
    _pt_tmp_1177 = (
        _pt_tmp_1178 * _pt_data_11
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_1178, _in1=_pt_data_11)[
            "out"
        ]
    )
    del _pt_tmp_1179
    _pt_tmp_1184 = _pt_tmp_493 * _pt_tmp_489
    del _pt_tmp_1178
    _pt_tmp_1183 = _pt_tmp_68 * _pt_tmp_1184
    del _pt_tmp_489, _pt_tmp_493
    _pt_tmp_1182 = _pt_tmp_1183 + _pt_tmp_955
    del _pt_tmp_1184
    _pt_tmp_1181 = (
        _pt_tmp_1182 * _pt_data_15
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_1182, _in1=_pt_data_15)[
            "out"
        ]
    )
    del _pt_tmp_1183, _pt_tmp_955
    _pt_tmp_1176 = _pt_tmp_1177 + _pt_tmp_1181
    del _pt_tmp_1182
    _pt_tmp_1187 = _pt_tmp_68 * _pt_tmp_657
    del _pt_tmp_1177, _pt_tmp_1181
    _pt_tmp_1186 = _pt_tmp_1187 + _pt_tmp_950
    del _pt_tmp_657, _pt_tmp_68
    _pt_tmp_1185 = (
        _pt_tmp_1186 * _pt_data_16
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_1186, _in1=_pt_data_16)[
            "out"
        ]
    )
    del _pt_tmp_1187, _pt_tmp_950
    _pt_tmp_1175 = _pt_tmp_1176 + _pt_tmp_1185
    del _pt_tmp_1186
    _pt_tmp_1193 = _pt_tmp_414 * _pt_tmp_382
    del _pt_tmp_1176, _pt_tmp_1185
    _pt_tmp_1192 = _pt_tmp_66 * _pt_tmp_1193
    del _pt_tmp_382
    _pt_tmp_1191 = _pt_tmp_1192 + _pt_tmp_970
    del _pt_tmp_1193
    _pt_tmp_1190 = (
        _pt_tmp_1191 * _pt_data_11
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_1191, _in1=_pt_data_11)[
            "out"
        ]
    )
    del _pt_tmp_1192
    _pt_tmp_1197 = _pt_tmp_414 * _pt_tmp_402
    del _pt_tmp_1191
    _pt_tmp_1196 = _pt_tmp_66 * _pt_tmp_1197
    del _pt_tmp_402, _pt_tmp_414
    _pt_tmp_1195 = _pt_tmp_1196 + _pt_tmp_970
    del _pt_tmp_1197
    _pt_tmp_1194 = (
        _pt_tmp_1195 * _pt_data_15
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_1195, _in1=_pt_data_15)[
            "out"
        ]
    )
    del _pt_tmp_1196, _pt_tmp_970
    _pt_tmp_1189 = _pt_tmp_1190 + _pt_tmp_1194
    del _pt_tmp_1195
    _pt_tmp_1200 = _pt_tmp_66 * _pt_tmp_668
    del _pt_tmp_1190, _pt_tmp_1194
    _pt_tmp_1199 = _pt_tmp_1200 + _pt_tmp_965
    del _pt_tmp_66, _pt_tmp_668
    _pt_tmp_1198 = (
        _pt_tmp_1199 * _pt_data_16
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_1199, _in1=_pt_data_16)[
            "out"
        ]
    )
    del _pt_tmp_1200, _pt_tmp_965
    _pt_tmp_1188 = _pt_tmp_1189 + _pt_tmp_1198
    del _pt_tmp_1199
    _pt_tmp_1174 = _pt_tmp_1175 + _pt_tmp_1188
    del _pt_tmp_1189, _pt_tmp_1198
    _pt_tmp_1202 = _pt_tmp_129 - _pt_tmp_127
    del _pt_tmp_1175, _pt_tmp_1188
    _pt_tmp_1201 = _pt_tmp_648 * _pt_tmp_1202
    del _pt_tmp_127, _pt_tmp_129
    _pt_tmp_1173 = _pt_tmp_1174 + _pt_tmp_1201
    del _pt_tmp_1202, _pt_tmp_648
    _pt_tmp_1172 = _pt_tmp_1173 / 2
    del _pt_tmp_1174, _pt_tmp_1201
    _pt_tmp_1171 = (
        _pt_tmp_1172[_pt_tmp_48, _pt_tmp_49]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_1172, in_1=_pt_tmp_48, in_2=_pt_tmp_49
        )["out"]
    )
    del _pt_tmp_1173
    _pt_tmp_1171 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_1171)
    del _pt_tmp_1172
    _pt_tmp_1171 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_1171)
    _pt_tmp_1170 = 0 + _pt_tmp_1171
    _pt_tmp_1169 = 0 + _pt_tmp_1170
    del _pt_tmp_1171
    _pt_tmp_1168 = _pt_tmp_1169 + 0
    del _pt_tmp_1170
    _pt_tmp_1152 = _pt_tmp_1153 - _pt_tmp_1168
    del _pt_tmp_1169
    _pt_tmp_1151 = actx.np.reshape(_pt_tmp_1152, (4, 48000, 15))
    del _pt_tmp_1153, _pt_tmp_1168
    _pt_tmp_1151 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_1151)
    del _pt_tmp_1152
    _pt_tmp_1150 = actx.einsum(
        "ijk, jl, jlk -> li", _pt_data_3, _pt_tmp_30, _pt_tmp_1151
    )
    _pt_tmp_1150 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_1150)
    del _pt_tmp_1151
    _pt_tmp_1136 = _pt_tmp_1137 - _pt_tmp_1150
    _pt_tmp_1135 = actx.einsum(
        "i, jk, ik -> ij", _pt_tmp_5, _pt_data_0, _pt_tmp_1136
    )
    del _pt_tmp_1137, _pt_tmp_1150
    _pt_tmp_1135 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_1135)
    del _pt_tmp_1136
    _pt_tmp_1134 = -1 * _pt_tmp_1135
    _pt_tmp_1203 = 0 * _pt_tmp_1134
    del _pt_tmp_1135
    _pt_tmp_1133 = _pt_tmp_1134 + _pt_tmp_1203
    _pt_tmp_1209 = _pt_tmp_682 * _pt_tmp_197
    del _pt_tmp_1134, _pt_tmp_1203
    _pt_tmp_1210 = _pt_tmp_682 * _pt_tmp_296
    del _pt_tmp_197
    _pt_tmp_1211 = _pt_tmp_682 * _pt_tmp_113
    del _pt_tmp_296
    _pt_tmp_1212 = actx.np.stack([_pt_tmp_1209, _pt_tmp_1210, _pt_tmp_1211], axis=0)
    del _pt_tmp_113, _pt_tmp_682
    _pt_tmp_1208 = actx.einsum(
        "ijk, jlm, ikm -> kl", _pt_tmp_9, _pt_data_2, _pt_tmp_1212
    )
    _pt_tmp_1208 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_1208)
    del _pt_tmp_1212, _pt_tmp_9
    _pt_tmp_1227 = (
        _pt_tmp_1209[_pt_tmp_44, _pt_tmp_45]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_pt_tmp_1209, in_1=_pt_tmp_44, in_2=_pt_tmp_45
        )["out"]
    )
    _pt_tmp_1227 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_1227)
    del _pt_tmp_1209
    _pt_tmp_1227 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_1227)
    _pt_tmp_1226 = 0 + _pt_tmp_1227
    _pt_tmp_1225 = (
        _pt_tmp_1226[_pt_tmp_46, _pt_tmp_47]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_1226, in_1=_pt_tmp_46, in_2=_pt_tmp_47
        )["out"]
    )
    del _pt_tmp_1227
    _pt_tmp_1225 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_1225)
    _pt_tmp_1225 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_1225)
    _pt_tmp_1224 = 0 + _pt_tmp_1225
    _pt_tmp_1223 = _pt_tmp_1224 + _pt_tmp_1226
    del _pt_tmp_1225
    _pt_tmp_1222 = _pt_tmp_1223 / 2
    del _pt_tmp_1224, _pt_tmp_1226
    _pt_tmp_1221 = (
        _pt_tmp_1222 * _pt_data_11
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_1222, _in1=_pt_data_11)[
            "out"
        ]
    )
    del _pt_tmp_1223
    _pt_tmp_1234 = (
        _pt_tmp_1210[_pt_tmp_44, _pt_tmp_45]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_pt_tmp_1210, in_1=_pt_tmp_44, in_2=_pt_tmp_45
        )["out"]
    )
    del _pt_tmp_1222
    _pt_tmp_1234 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_1234)
    del _pt_tmp_1210
    _pt_tmp_1234 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_1234)
    _pt_tmp_1233 = 0 + _pt_tmp_1234
    _pt_tmp_1232 = (
        _pt_tmp_1233[_pt_tmp_46, _pt_tmp_47]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_1233, in_1=_pt_tmp_46, in_2=_pt_tmp_47
        )["out"]
    )
    del _pt_tmp_1234
    _pt_tmp_1232 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_1232)
    _pt_tmp_1232 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_1232)
    _pt_tmp_1231 = 0 + _pt_tmp_1232
    _pt_tmp_1230 = _pt_tmp_1231 + _pt_tmp_1233
    del _pt_tmp_1232
    _pt_tmp_1229 = _pt_tmp_1230 / 2
    del _pt_tmp_1231, _pt_tmp_1233
    _pt_tmp_1228 = (
        _pt_tmp_1229 * _pt_data_15
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_1229, _in1=_pt_data_15)[
            "out"
        ]
    )
    del _pt_tmp_1230
    _pt_tmp_1220 = _pt_tmp_1221 + _pt_tmp_1228
    del _pt_tmp_1229
    _pt_tmp_1241 = (
        _pt_tmp_1211[_pt_tmp_44, _pt_tmp_45]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_pt_tmp_1211, in_1=_pt_tmp_44, in_2=_pt_tmp_45
        )["out"]
    )
    del _pt_tmp_1221, _pt_tmp_1228
    _pt_tmp_1241 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_1241)
    del _pt_tmp_1211, _pt_tmp_44, _pt_tmp_45
    _pt_tmp_1241 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_1241)
    _pt_tmp_1240 = 0 + _pt_tmp_1241
    _pt_tmp_1239 = (
        _pt_tmp_1240[_pt_tmp_46, _pt_tmp_47]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_1240, in_1=_pt_tmp_46, in_2=_pt_tmp_47
        )["out"]
    )
    del _pt_tmp_1241
    _pt_tmp_1239 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_1239)
    del _pt_tmp_46, _pt_tmp_47
    _pt_tmp_1239 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_1239)
    _pt_tmp_1238 = 0 + _pt_tmp_1239
    _pt_tmp_1237 = _pt_tmp_1238 + _pt_tmp_1240
    del _pt_tmp_1239
    _pt_tmp_1236 = _pt_tmp_1237 / 2
    del _pt_tmp_1238, _pt_tmp_1240
    _pt_tmp_1235 = (
        _pt_tmp_1236 * _pt_data_16
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_1236, _in1=_pt_data_16)[
            "out"
        ]
    )
    del _pt_tmp_1237
    _pt_tmp_1219 = _pt_tmp_1220 + _pt_tmp_1235
    del _pt_tmp_1236
    _pt_tmp_1218 = (
        _pt_tmp_1219[_pt_tmp_48, _pt_tmp_49]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_1219, in_1=_pt_tmp_48, in_2=_pt_tmp_49
        )["out"]
    )
    del _pt_tmp_1220, _pt_tmp_1235
    _pt_tmp_1218 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_1218)
    del _pt_tmp_1219, _pt_tmp_48, _pt_tmp_49
    _pt_tmp_1218 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_1218)
    _pt_tmp_1217 = 0 + _pt_tmp_1218
    _pt_tmp_1216 = 0 + _pt_tmp_1217
    del _pt_tmp_1218
    _pt_tmp_1215 = _pt_tmp_1216 + 0
    del _pt_tmp_1217
    _pt_tmp_1214 = actx.np.reshape(_pt_tmp_1215, (4, 48000, 15))
    del _pt_tmp_1216
    _pt_tmp_1214 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_1214)
    del _pt_tmp_1215
    _pt_tmp_1213 = actx.einsum(
        "ijk, jl, jlk -> li", _pt_data_3, _pt_tmp_30, _pt_tmp_1214
    )
    _pt_tmp_1213 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_1213)
    del _pt_tmp_1214, _pt_tmp_30
    _pt_tmp_1207 = _pt_tmp_1208 - _pt_tmp_1213
    _pt_tmp_1206 = actx.einsum(
        "i, jk, ik -> ij", _pt_tmp_5, _pt_data_0, _pt_tmp_1207
    )
    del _pt_tmp_1208, _pt_tmp_1213
    _pt_tmp_1206 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_1206)
    del _pt_tmp_1207, _pt_tmp_5
    _pt_tmp_1205 = -1 * _pt_tmp_1206
    _pt_tmp_1204 = -1 * _pt_tmp_1205
    del _pt_tmp_1206
    _pt_tmp_1132 = _pt_tmp_1133 + _pt_tmp_1204
    del _pt_tmp_1205
    _pt_tmp_1243 = _pt_tmp_798 - _actx_in_1_0_momentum_2_0
    del _pt_tmp_1133, _pt_tmp_1204
    _pt_tmp_1242 = _pt_tmp_782 * _pt_tmp_1243
    del _pt_tmp_798
    _pt_tmp_1131 = _pt_tmp_1132 + _pt_tmp_1242
    del _pt_tmp_1243, _pt_tmp_782
    _actx_in_1_1_0 = actx.tag_axis(
        0, (DiscretizationElementAxisTag(),), _actx_in_1_1_0
    )
    del _pt_tmp_1132, _pt_tmp_1242
    _actx_in_1_1_0 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _actx_in_1_1_0)
    _pt_tmp_1244 = _pt_tmp_217 - _actx_in_1_1_0
    _pt_tmp = make_obj_array(
        [
            _pt_tmp_0,
            _pt_tmp_801,
            _pt_tmp_899,
            _pt_tmp_1018,
            _pt_tmp_1131,
            _pt_tmp_1244,
        ]
    )
    del _pt_tmp_217
    return _pt_tmp
    del (
        _pt_tmp_0,
        _pt_tmp_1018,
        _pt_tmp_1131,
        _pt_tmp_1244,
        _pt_tmp_801,
        _pt_tmp_899,
    )


@dataclass(frozen=True)
class RHSInvoker:
    actx: ArrayContext

    @cached_property
    def npzfile(self):
        import os

        from immutables import Map

        kw_to_ary = np.load(
            os.path.join(
                get_actx_dgfem_suite_path(),
                "suite/cns_without_chem_3D_P4/literals.npz",
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
            get_actx_dgfem_suite_path(),
            "suite/cns_without_chem_3D_P4/ref_outputs.pkl",
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
            "_actx_in_1_0_energy_0",
            "_actx_in_1_0_mass_0",
            "_actx_in_1_0_momentum_0_0",
            "_actx_in_1_0_momentum_1_0",
            "_actx_in_1_0_momentum_2_0",
            "_actx_in_1_1_0",
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
