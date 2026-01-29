from dataclasses import dataclass
from functools import cached_property

import loopy as lp
import numpy as np
from arraycontext import ArrayContext, is_array_container_type, make_loopy_program
from arraycontext.container.traversal import (
    rec_keyed_map_array_container,
    rec_map_array_container,
)
from immutabledict import immutabledict
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
    _actx_in_1_energy_0,
    _actx_in_1_mass_0,
    _actx_in_1_momentum_0_0,
    _actx_in_1_momentum_1_0,
    _actx_in_1_momentum_2_0,
):
    _pt_t_unit = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 637199 and 0 <= _1 <= 5 }",
        "out[_0, _1] = in_0[in_1[_0] % 4, _1]",
        [
            lp.GlobalArg("out", dtype=np.int32, shape=(637200, 6)),
            lp.GlobalArg("in_0", dtype=np.int32, shape=(4, 6)),
            lp.GlobalArg("in_1", dtype=np.int8, shape=(637200,)),
        ],
    )
    _pt_t_unit_0 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 637199 and 0 <= _1 <= 5 }",
        "out[_0, _1] = in_0[in_1[_0, 0] % 162000, in_2[_0, _1] % 10]",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(637200, 6)),
            lp.GlobalArg("in_0", dtype=np.float64, shape=(162000, 10)),
            lp.GlobalArg("in_1", dtype=np.int32, shape=(637200, 1)),
            lp.GlobalArg("in_2", dtype=np.int32, shape=(637200, 6)),
        ],
    )
    _pt_t_unit_1 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 637199 and 0 <= _1 <= 5 }",
        "out[_0, _1] = in_0[in_1[_0] % 3, _1]",
        [
            lp.GlobalArg("out", dtype=np.int32, shape=(637200, 6)),
            lp.GlobalArg("in_0", dtype=np.int32, shape=(3, 6)),
            lp.GlobalArg("in_1", dtype=np.int8, shape=(637200,)),
        ],
    )
    _pt_t_unit_10 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 10799 and 0 <= _1 <= 5 }",
        "out[_0, _1] = _in0[_0, 0]*_in1[_0, _1]",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(10800, 6)),
            lp.GlobalArg("_in0", dtype=np.float64, shape=(10800, 1)),
            lp.GlobalArg("_in1", dtype=np.float64, shape=(10800, 6)),
        ],
    )
    _pt_t_unit_11 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 647999 and 0 <= _1 <= 5 }",
        "out[_0, _1] = in_0[in_1[_0, 0] % 10800, in_2[_0, _1] % 6]",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(648000, 6)),
            lp.GlobalArg("in_0", dtype=np.float64, shape=(10800, 6)),
            lp.GlobalArg("in_1", dtype=np.int32, shape=(648000, 1)),
            lp.GlobalArg("in_2", dtype=np.int32, shape=(648000, 6)),
        ],
    )
    _pt_t_unit_2 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 637199 and 0 <= _1 <= 5 }",
        "out[_0, _1] = in_0[in_1[_0, 0] % 637200, in_2[_0, _1] % 6]",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(637200, 6)),
            lp.GlobalArg("in_0", dtype=np.float64, shape=(637200, 6)),
            lp.GlobalArg("in_1", dtype=np.int32, shape=(637200, 1)),
            lp.GlobalArg("in_2", dtype=np.int32, shape=(637200, 6)),
        ],
    )
    _pt_t_unit_3 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 637199 and 0 <= _1 <= 5 }",
        "out[_0, _1] = _in0[_0, _1]*_in1[_0, 0]",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(637200, 6)),
            lp.GlobalArg("_in0", dtype=np.float64, shape=(637200, 6)),
            lp.GlobalArg("_in1", dtype=np.float64, shape=(637200, 1)),
        ],
    )
    _pt_t_unit_4 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 647999 and 0 <= _1 <= 5 }",
        "out[_0, _1] = in_0[0, _1]",
        [
            lp.GlobalArg("out", dtype=np.int32, shape=(648000, 6)),
            lp.GlobalArg("in_0", dtype=np.int32, shape=(1, 6)),
            lp.GlobalArg("in_1", dtype=np.int32, shape=(648000,)),
        ],
    )
    _pt_t_unit_5 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 647999 and 0 <= _1 <= 5 }",
        "out[_0, _1] = in_0[in_1[_0, 0] % 637200, in_2[_0, _1] % 6]",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(648000, 6)),
            lp.GlobalArg("in_0", dtype=np.float64, shape=(637200, 6)),
            lp.GlobalArg("in_1", dtype=np.int32, shape=(648000, 1)),
            lp.GlobalArg("in_2", dtype=np.int32, shape=(648000, 6)),
        ],
    )
    _pt_t_unit_6 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 647999 and 0 <= _1 <= 5 }",
        "out[_0, _1] = _in1[_0, _1] if _in0[_0, 0] else 0",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(648000, 6)),
            lp.GlobalArg("_in0", dtype=np.int8, shape=(648000, 1)),
            lp.GlobalArg("_in1", dtype=np.float64, shape=(648000, 6)),
        ],
    )
    _pt_t_unit_7 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 10799 and 0 <= _1 <= 5 }",
        "out[_0, _1] = in_0[in_1[_0] % 4, _1]",
        [
            lp.GlobalArg("out", dtype=np.int32, shape=(10800, 6)),
            lp.GlobalArg("in_0", dtype=np.int32, shape=(4, 6)),
            lp.GlobalArg("in_1", dtype=np.int8, shape=(10800,)),
        ],
    )
    _pt_t_unit_8 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 10799 and 0 <= _1 <= 5 }",
        "out[_0, _1] = in_0[in_1[_0, 0] % 162000, in_2[_0, _1] % 10]",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(10800, 6)),
            lp.GlobalArg("in_0", dtype=np.float64, shape=(162000, 10)),
            lp.GlobalArg("in_1", dtype=np.int32, shape=(10800, 1)),
            lp.GlobalArg("in_2", dtype=np.int32, shape=(10800, 6)),
        ],
    )
    _pt_t_unit_9 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 10799 and 0 <= _1 <= 5 }",
        "out[_0, _1] = _in0[_0, _1]*_in1[_0, 0]",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(10800, 6)),
            lp.GlobalArg("_in0", dtype=np.float64, shape=(10800, 6)),
            lp.GlobalArg("_in1", dtype=np.float64, shape=(10800, 1)),
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
    _pt_tmp_5 = _pt_data_1[:, :, :, 0]
    _pt_data_2 = actx.thaw(npzfile["_pt_data_2"])
    _pt_data_2 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_data_2)
    _actx_in_1_momentum_0_0 = actx.tag_axis(
        0, (DiscretizationElementAxisTag(),), _actx_in_1_momentum_0_0
    )
    _actx_in_1_momentum_0_0 = actx.tag_axis(
        1, (DiscretizationDOFAxisTag(),), _actx_in_1_momentum_0_0
    )
    _actx_in_1_mass_0 = actx.tag_axis(
        0, (DiscretizationElementAxisTag(),), _actx_in_1_mass_0
    )
    _actx_in_1_mass_0 = actx.tag_axis(
        1, (DiscretizationDOFAxisTag(),), _actx_in_1_mass_0
    )
    _pt_tmp_7 = _actx_in_1_momentum_0_0 / _actx_in_1_mass_0
    _actx_in_1_energy_0 = actx.tag_axis(
        0, (DiscretizationElementAxisTag(),), _actx_in_1_energy_0
    )
    _actx_in_1_energy_0 = actx.tag_axis(
        1, (DiscretizationDOFAxisTag(),), _actx_in_1_energy_0
    )
    _pt_tmp_15 = _actx_in_1_momentum_0_0 * _pt_tmp_7
    _pt_tmp_14 = 0 + _pt_tmp_15
    _actx_in_1_momentum_1_0 = actx.tag_axis(
        0, (DiscretizationElementAxisTag(),), _actx_in_1_momentum_1_0
    )
    del _pt_tmp_15
    _actx_in_1_momentum_1_0 = actx.tag_axis(
        1, (DiscretizationDOFAxisTag(),), _actx_in_1_momentum_1_0
    )
    _pt_tmp_17 = _actx_in_1_momentum_1_0 / _actx_in_1_mass_0
    _pt_tmp_16 = _actx_in_1_momentum_1_0 * _pt_tmp_17
    _pt_tmp_13 = _pt_tmp_14 + _pt_tmp_16
    _actx_in_1_momentum_2_0 = actx.tag_axis(
        0, (DiscretizationElementAxisTag(),), _actx_in_1_momentum_2_0
    )
    del _pt_tmp_14, _pt_tmp_16
    _actx_in_1_momentum_2_0 = actx.tag_axis(
        1, (DiscretizationDOFAxisTag(),), _actx_in_1_momentum_2_0
    )
    _pt_tmp_19 = _actx_in_1_momentum_2_0 / _actx_in_1_mass_0
    _pt_tmp_18 = _actx_in_1_momentum_2_0 * _pt_tmp_19
    _pt_tmp_12 = _pt_tmp_13 + _pt_tmp_18
    _pt_tmp_11 = 0.5 * _pt_tmp_12
    del _pt_tmp_13, _pt_tmp_18
    _pt_tmp_10 = _actx_in_1_energy_0 - _pt_tmp_11
    del _pt_tmp_12
    _pt_tmp_9 = 0.3999999999999999 * _pt_tmp_10
    del _pt_tmp_11
    _pt_tmp_8 = _actx_in_1_energy_0 + _pt_tmp_9
    del _pt_tmp_10
    _pt_tmp_6 = _pt_tmp_7 * _pt_tmp_8
    _pt_tmp_20 = _pt_tmp_17 * _pt_tmp_8
    _pt_tmp_21 = _pt_tmp_19 * _pt_tmp_8
    _pt_tmp_22 = actx.np.stack([_pt_tmp_6, _pt_tmp_20, _pt_tmp_21], axis=0)
    del _pt_tmp_8
    _pt_tmp_4 = actx.einsum("ijk, jlm, ikm -> kl", _pt_tmp_5, _pt_data_2, _pt_tmp_22)
    del _pt_tmp_20, _pt_tmp_21, _pt_tmp_6
    _pt_tmp_4 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_4)
    del _pt_tmp_22
    _pt_data_3 = actx.thaw(npzfile["_pt_data_3"])
    _pt_data_3 = actx.tag_axis(0, (DiscretizationDOFAxisTag(),), _pt_data_3)
    _pt_data_3 = actx.tag_axis(2, (DiscretizationDOFAxisTag(),), _pt_data_3)
    _pt_data_4 = actx.thaw(npzfile["_pt_data_4"])
    _pt_data_4 = actx.tag(
        (PrefixNamed(prefix="area_el_b_face_restr_all"),), _pt_data_4
    )
    _pt_data_4 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_data_4)
    _pt_data_4 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_data_4)
    _pt_tmp_25 = actx.np.reshape(_pt_data_4, (4, 162000, 1))
    _pt_tmp_25 = actx.tag_axis(1, (DiscretizationElementAxisTag(),), _pt_tmp_25)
    _pt_tmp_24 = _pt_tmp_25[:, :, 0]
    _pt_data_5 = actx.thaw(npzfile["_pt_data_5"])
    del _pt_tmp_25
    _pt_data_5 = actx.tag((PrefixNamed(prefix="from_el_present"),), _pt_data_5)
    _pt_tmp_31 = actx.np.reshape(_pt_data_5, (648000, 1))
    _pt_tmp_31 = actx.tag((PrefixNamed(prefix="from_el_present"),), _pt_tmp_31)
    _pt_data_6 = actx.thaw(npzfile["_pt_data_6"])
    _pt_data_6 = actx.tag((PrefixNamed(prefix="from_el_indices"),), _pt_data_6)
    _pt_tmp_43 = actx.np.reshape(_pt_data_6, (637200, 1))
    _pt_tmp_43 = actx.tag((PrefixNamed(prefix="from_el_indices"),), _pt_tmp_43)
    _pt_data_7 = actx.thaw(npzfile["_pt_data_7"])
    _pt_data_7 = actx.tag((PrefixNamed(prefix="dof_pick_lists"),), _pt_data_7)
    _pt_data_8 = actx.thaw(npzfile["_pt_data_8"])
    _pt_data_8 = actx.tag((PrefixNamed(prefix="dof_pick_list_indices"),), _pt_data_8)
    _pt_tmp_44 = (
        _pt_data_7[_pt_data_8]
        if actx.permits_advanced_indexing
        else actx.call_loopy(_pt_t_unit, in_0=_pt_data_7, in_1=_pt_data_8)["out"]
    )
    _pt_tmp_42 = (
        _actx_in_1_momentum_0_0[_pt_tmp_43, _pt_tmp_44]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0,
            in_0=_actx_in_1_momentum_0_0,
            in_1=_pt_tmp_43,
            in_2=_pt_tmp_44,
        )["out"]
    )
    _pt_tmp_42 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_42)
    _pt_tmp_42 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_42)
    _pt_tmp_41 = 0 + _pt_tmp_42
    _pt_tmp_46 = (
        _actx_in_1_mass_0[_pt_tmp_43, _pt_tmp_44]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0,
            in_0=_actx_in_1_mass_0,
            in_1=_pt_tmp_43,
            in_2=_pt_tmp_44,
        )["out"]
    )
    del _pt_tmp_42
    _pt_tmp_46 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_46)
    _pt_tmp_46 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_46)
    _pt_tmp_45 = 0 + _pt_tmp_46
    _pt_tmp_40 = _pt_tmp_41 / _pt_tmp_45
    del _pt_tmp_46
    _pt_tmp_49 = (
        _actx_in_1_energy_0[_pt_tmp_43, _pt_tmp_44]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0,
            in_0=_actx_in_1_energy_0,
            in_1=_pt_tmp_43,
            in_2=_pt_tmp_44,
        )["out"]
    )
    _pt_tmp_49 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_49)
    _pt_tmp_49 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_49)
    _pt_tmp_48 = 0 + _pt_tmp_49
    _pt_tmp_56 = _pt_tmp_41 * _pt_tmp_40
    del _pt_tmp_49
    _pt_tmp_55 = 0 + _pt_tmp_56
    _pt_tmp_59 = (
        _actx_in_1_momentum_1_0[_pt_tmp_43, _pt_tmp_44]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0,
            in_0=_actx_in_1_momentum_1_0,
            in_1=_pt_tmp_43,
            in_2=_pt_tmp_44,
        )["out"]
    )
    del _pt_tmp_56
    _pt_tmp_59 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_59)
    _pt_tmp_59 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_59)
    _pt_tmp_58 = 0 + _pt_tmp_59
    _pt_tmp_60 = _pt_tmp_58 / _pt_tmp_45
    del _pt_tmp_59
    _pt_tmp_57 = _pt_tmp_58 * _pt_tmp_60
    _pt_tmp_54 = _pt_tmp_55 + _pt_tmp_57
    _pt_tmp_63 = (
        _actx_in_1_momentum_2_0[_pt_tmp_43, _pt_tmp_44]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0,
            in_0=_actx_in_1_momentum_2_0,
            in_1=_pt_tmp_43,
            in_2=_pt_tmp_44,
        )["out"]
    )
    del _pt_tmp_55, _pt_tmp_57
    _pt_tmp_63 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_63)
    del _pt_tmp_43, _pt_tmp_44
    _pt_tmp_63 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_63)
    _pt_tmp_62 = 0 + _pt_tmp_63
    _pt_tmp_64 = _pt_tmp_62 / _pt_tmp_45
    del _pt_tmp_63
    _pt_tmp_61 = _pt_tmp_62 * _pt_tmp_64
    _pt_tmp_53 = _pt_tmp_54 + _pt_tmp_61
    _pt_tmp_52 = 0.5 * _pt_tmp_53
    del _pt_tmp_54, _pt_tmp_61
    _pt_tmp_51 = _pt_tmp_48 - _pt_tmp_52
    del _pt_tmp_53
    _pt_tmp_50 = 0.3999999999999999 * _pt_tmp_51
    del _pt_tmp_52
    _pt_tmp_47 = _pt_tmp_48 + _pt_tmp_50
    del _pt_tmp_51
    _pt_tmp_39 = _pt_tmp_40 * _pt_tmp_47
    _pt_data_9 = actx.thaw(npzfile["_pt_data_9"])
    _pt_data_9 = actx.tag((PrefixNamed(prefix="from_el_indices"),), _pt_data_9)
    _pt_tmp_69 = actx.np.reshape(_pt_data_9, (637200, 1))
    _pt_tmp_69 = actx.tag((PrefixNamed(prefix="from_el_indices"),), _pt_tmp_69)
    _pt_data_10 = actx.thaw(npzfile["_pt_data_10"])
    _pt_data_10 = actx.tag((PrefixNamed(prefix="dof_pick_lists"),), _pt_data_10)
    _pt_data_11 = actx.thaw(npzfile["_pt_data_11"])
    _pt_data_11 = actx.tag(
        (PrefixNamed(prefix="dof_pick_list_indices"),), _pt_data_11
    )
    _pt_tmp_70 = (
        _pt_data_10[_pt_data_11]
        if actx.permits_advanced_indexing
        else actx.call_loopy(_pt_t_unit_1, in_0=_pt_data_10, in_1=_pt_data_11)["out"]
    )
    _pt_tmp_68 = (
        _pt_tmp_41[_pt_tmp_69, _pt_tmp_70]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_41, in_1=_pt_tmp_69, in_2=_pt_tmp_70
        )["out"]
    )
    _pt_tmp_68 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_68)
    _pt_tmp_68 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_68)
    _pt_tmp_67 = 0 + _pt_tmp_68
    _pt_tmp_72 = (
        _pt_tmp_45[_pt_tmp_69, _pt_tmp_70]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_45, in_1=_pt_tmp_69, in_2=_pt_tmp_70
        )["out"]
    )
    del _pt_tmp_68
    _pt_tmp_72 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_72)
    _pt_tmp_72 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_72)
    _pt_tmp_71 = 0 + _pt_tmp_72
    _pt_tmp_66 = _pt_tmp_67 / _pt_tmp_71
    del _pt_tmp_72
    _pt_tmp_75 = (
        _pt_tmp_48[_pt_tmp_69, _pt_tmp_70]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_48, in_1=_pt_tmp_69, in_2=_pt_tmp_70
        )["out"]
    )
    _pt_tmp_75 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_75)
    _pt_tmp_75 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_75)
    _pt_tmp_74 = 0 + _pt_tmp_75
    _pt_tmp_82 = _pt_tmp_67 * _pt_tmp_66
    del _pt_tmp_75
    _pt_tmp_81 = 0 + _pt_tmp_82
    _pt_tmp_85 = (
        _pt_tmp_58[_pt_tmp_69, _pt_tmp_70]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_58, in_1=_pt_tmp_69, in_2=_pt_tmp_70
        )["out"]
    )
    del _pt_tmp_82
    _pt_tmp_85 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_85)
    _pt_tmp_85 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_85)
    _pt_tmp_84 = 0 + _pt_tmp_85
    _pt_tmp_86 = _pt_tmp_84 / _pt_tmp_71
    del _pt_tmp_85
    _pt_tmp_83 = _pt_tmp_84 * _pt_tmp_86
    _pt_tmp_80 = _pt_tmp_81 + _pt_tmp_83
    _pt_tmp_89 = (
        _pt_tmp_62[_pt_tmp_69, _pt_tmp_70]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_2, in_0=_pt_tmp_62, in_1=_pt_tmp_69, in_2=_pt_tmp_70
        )["out"]
    )
    del _pt_tmp_81, _pt_tmp_83
    _pt_tmp_89 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_89)
    del _pt_tmp_69, _pt_tmp_70
    _pt_tmp_89 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_89)
    _pt_tmp_88 = 0 + _pt_tmp_89
    _pt_tmp_90 = _pt_tmp_88 / _pt_tmp_71
    del _pt_tmp_89
    _pt_tmp_87 = _pt_tmp_88 * _pt_tmp_90
    _pt_tmp_79 = _pt_tmp_80 + _pt_tmp_87
    _pt_tmp_78 = 0.5 * _pt_tmp_79
    del _pt_tmp_80, _pt_tmp_87
    _pt_tmp_77 = _pt_tmp_74 - _pt_tmp_78
    del _pt_tmp_79
    _pt_tmp_76 = 0.3999999999999999 * _pt_tmp_77
    del _pt_tmp_78
    _pt_tmp_73 = _pt_tmp_74 + _pt_tmp_76
    del _pt_tmp_77
    _pt_tmp_65 = _pt_tmp_66 * _pt_tmp_73
    _pt_tmp_38 = _pt_tmp_39 + _pt_tmp_65
    _pt_tmp_37 = 0.5 * _pt_tmp_38
    del _pt_tmp_39, _pt_tmp_65
    _pt_tmp_100 = _pt_tmp_40 * _pt_tmp_40
    del _pt_tmp_38
    _pt_tmp_101 = _pt_tmp_60 * _pt_tmp_60
    _pt_tmp_99 = _pt_tmp_100 + _pt_tmp_101
    _pt_tmp_102 = _pt_tmp_64 * _pt_tmp_64
    _pt_tmp_98 = _pt_tmp_99 + _pt_tmp_102
    _pt_tmp_97 = actx.np.sqrt(_pt_tmp_98)
    del _pt_tmp_99
    _pt_tmp_105 = _pt_tmp_50 / _pt_tmp_45
    del _pt_tmp_98
    _pt_tmp_104 = 1.4 * _pt_tmp_105
    _pt_tmp_103 = actx.np.sqrt(_pt_tmp_104)
    del _pt_tmp_105
    _pt_tmp_96 = _pt_tmp_97 + _pt_tmp_103
    del _pt_tmp_104
    _pt_tmp_95 = actx.np.isnan(_pt_tmp_96)
    del _pt_tmp_103, _pt_tmp_97
    _pt_tmp_111 = _pt_tmp_66 * _pt_tmp_66
    _pt_tmp_112 = _pt_tmp_86 * _pt_tmp_86
    _pt_tmp_110 = _pt_tmp_111 + _pt_tmp_112
    _pt_tmp_113 = _pt_tmp_90 * _pt_tmp_90
    _pt_tmp_109 = _pt_tmp_110 + _pt_tmp_113
    _pt_tmp_108 = actx.np.sqrt(_pt_tmp_109)
    del _pt_tmp_110
    _pt_tmp_116 = _pt_tmp_76 / _pt_tmp_71
    del _pt_tmp_109
    _pt_tmp_115 = 1.4 * _pt_tmp_116
    _pt_tmp_114 = actx.np.sqrt(_pt_tmp_115)
    del _pt_tmp_116
    _pt_tmp_107 = _pt_tmp_108 + _pt_tmp_114
    del _pt_tmp_115
    _pt_tmp_106 = actx.np.isnan(_pt_tmp_107)
    del _pt_tmp_108, _pt_tmp_114
    _pt_tmp_94 = actx.np.logical_or(_pt_tmp_95, _pt_tmp_106)
    _pt_tmp_118 = actx.np.greater(_pt_tmp_96, _pt_tmp_107)
    del _pt_tmp_106, _pt_tmp_95
    _pt_tmp_117 = actx.np.where(_pt_tmp_118, _pt_tmp_96, _pt_tmp_107)
    _pt_tmp_93 = actx.np.where(_pt_tmp_94, np.float64("nan"), _pt_tmp_117)
    del _pt_tmp_107, _pt_tmp_118, _pt_tmp_96
    _pt_tmp_120 = _pt_tmp_74 - _pt_tmp_48
    del _pt_tmp_117, _pt_tmp_94
    _pt_data_12 = actx.thaw(npzfile["_pt_data_12"])
    del _pt_tmp_48, _pt_tmp_74
    _pt_data_12 = actx.tag(
        (PrefixNamed(prefix="normal_1_b_face_restr_interior"),), _pt_data_12
    )
    _pt_data_12 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_data_12)
    _pt_data_12 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_data_12)
    _pt_tmp_119 = (
        _pt_tmp_120 * _pt_data_12
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_120, _in1=_pt_data_12)["out"]
    )
    _pt_tmp_92 = _pt_tmp_93 * _pt_tmp_119
    _pt_tmp_91 = _pt_tmp_92 / 2
    del _pt_tmp_119
    _pt_tmp_36 = _pt_tmp_37 - _pt_tmp_91
    del _pt_tmp_92
    _pt_tmp_35 = (
        _pt_tmp_36 * _pt_data_12
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_36, _in1=_pt_data_12)["out"]
    )
    del _pt_tmp_37, _pt_tmp_91
    _pt_tmp_125 = _pt_tmp_60 * _pt_tmp_47
    del _pt_tmp_36
    _pt_tmp_126 = _pt_tmp_86 * _pt_tmp_73
    _pt_tmp_124 = _pt_tmp_125 + _pt_tmp_126
    _pt_tmp_123 = 0.5 * _pt_tmp_124
    del _pt_tmp_125, _pt_tmp_126
    _pt_data_13 = actx.thaw(npzfile["_pt_data_13"])
    del _pt_tmp_124
    _pt_data_13 = actx.tag(
        (PrefixNamed(prefix="normal_2_b_face_restr_interior"),), _pt_data_13
    )
    _pt_data_13 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_data_13)
    _pt_data_13 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_data_13)
    _pt_tmp_129 = (
        _pt_tmp_120 * _pt_data_13
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_120, _in1=_pt_data_13)["out"]
    )
    _pt_tmp_128 = _pt_tmp_93 * _pt_tmp_129
    _pt_tmp_127 = _pt_tmp_128 / 2
    del _pt_tmp_129
    _pt_tmp_122 = _pt_tmp_123 - _pt_tmp_127
    del _pt_tmp_128
    _pt_tmp_121 = (
        _pt_tmp_122 * _pt_data_13
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_122, _in1=_pt_data_13)["out"]
    )
    del _pt_tmp_123, _pt_tmp_127
    _pt_tmp_34 = _pt_tmp_35 + _pt_tmp_121
    del _pt_tmp_122
    _pt_tmp_134 = _pt_tmp_64 * _pt_tmp_47
    del _pt_tmp_121, _pt_tmp_35
    _pt_tmp_135 = _pt_tmp_90 * _pt_tmp_73
    del _pt_tmp_47
    _pt_tmp_133 = _pt_tmp_134 + _pt_tmp_135
    del _pt_tmp_73
    _pt_tmp_132 = 0.5 * _pt_tmp_133
    del _pt_tmp_134, _pt_tmp_135
    _pt_data_14 = actx.thaw(npzfile["_pt_data_14"])
    del _pt_tmp_133
    _pt_data_14 = actx.tag(
        (PrefixNamed(prefix="normal_4_b_face_restr_interior"),), _pt_data_14
    )
    _pt_data_14 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_data_14)
    _pt_data_14 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_data_14)
    _pt_tmp_138 = (
        _pt_tmp_120 * _pt_data_14
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_120, _in1=_pt_data_14)["out"]
    )
    _pt_tmp_137 = _pt_tmp_93 * _pt_tmp_138
    del _pt_tmp_120
    _pt_tmp_136 = _pt_tmp_137 / 2
    del _pt_tmp_138
    _pt_tmp_131 = _pt_tmp_132 - _pt_tmp_136
    del _pt_tmp_137
    _pt_tmp_130 = (
        _pt_tmp_131 * _pt_data_14
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_131, _in1=_pt_data_14)["out"]
    )
    del _pt_tmp_132, _pt_tmp_136
    _pt_tmp_33 = _pt_tmp_34 + _pt_tmp_130
    del _pt_tmp_131
    _pt_data_15 = actx.thaw(npzfile["_pt_data_15"])
    del _pt_tmp_130, _pt_tmp_34
    _pt_data_15 = actx.tag((PrefixNamed(prefix="from_el_indices"),), _pt_data_15)
    _pt_tmp_139 = actx.np.reshape(_pt_data_15, (648000, 1))
    _pt_tmp_139 = actx.tag((PrefixNamed(prefix="from_el_indices"),), _pt_tmp_139)
    _pt_data_16 = actx.thaw(npzfile["_pt_data_16"])
    _pt_data_16 = actx.tag((PrefixNamed(prefix="dof_pick_lists"),), _pt_data_16)
    _pt_tmp_141 = actx.np.zeros((648000,), dtype=np.int32)
    _pt_tmp_140 = (
        _pt_data_16[_pt_tmp_141]
        if actx.permits_advanced_indexing
        else actx.call_loopy(_pt_t_unit_4, in_0=_pt_data_16, in_1=_pt_tmp_141)["out"]
    )
    _pt_tmp_32 = (
        _pt_tmp_33[_pt_tmp_139, _pt_tmp_140]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_5, in_0=_pt_tmp_33, in_1=_pt_tmp_139, in_2=_pt_tmp_140
        )["out"]
    )
    _pt_tmp_30 = (
        actx.np.where(_pt_tmp_31, _pt_tmp_32, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_6, _in0=_pt_tmp_31, _in1=_pt_tmp_32)["out"]
    )
    del _pt_tmp_33
    _pt_tmp_30 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_30)
    del _pt_tmp_32
    _pt_tmp_30 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_30)
    _pt_tmp_29 = 0 + _pt_tmp_30
    _pt_tmp_28 = 0 + _pt_tmp_29
    del _pt_tmp_30
    _pt_data_17 = actx.thaw(npzfile["_pt_data_17"])
    del _pt_tmp_29
    _pt_data_17 = actx.tag((PrefixNamed(prefix="from_el_present"),), _pt_data_17)
    _pt_tmp_145 = actx.np.reshape(_pt_data_17, (648000, 1))
    _pt_tmp_145 = actx.tag((PrefixNamed(prefix="from_el_present"),), _pt_tmp_145)
    _pt_data_18 = actx.thaw(npzfile["_pt_data_18"])
    _pt_data_18 = actx.tag((PrefixNamed(prefix="from_el_indices"),), _pt_data_18)
    _pt_tmp_157 = actx.np.reshape(_pt_data_18, (10800, 1))
    _pt_tmp_157 = actx.tag((PrefixNamed(prefix="from_el_indices"),), _pt_tmp_157)
    _pt_data_19 = actx.thaw(npzfile["_pt_data_19"])
    _pt_data_19 = actx.tag((PrefixNamed(prefix="dof_pick_lists"),), _pt_data_19)
    _pt_data_20 = actx.thaw(npzfile["_pt_data_20"])
    _pt_data_20 = actx.tag(
        (PrefixNamed(prefix="dof_pick_list_indices"),), _pt_data_20
    )
    _pt_tmp_158 = (
        _pt_data_19[_pt_data_20]
        if actx.permits_advanced_indexing
        else actx.call_loopy(_pt_t_unit_7, in_0=_pt_data_19, in_1=_pt_data_20)["out"]
    )
    _pt_tmp_156 = (
        _actx_in_1_momentum_0_0[_pt_tmp_157, _pt_tmp_158]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_8,
            in_0=_actx_in_1_momentum_0_0,
            in_1=_pt_tmp_157,
            in_2=_pt_tmp_158,
        )["out"]
    )
    _pt_tmp_156 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_156)
    _pt_tmp_156 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_156)
    _pt_tmp_155 = 0 + _pt_tmp_156
    _pt_tmp_160 = (
        _actx_in_1_mass_0[_pt_tmp_157, _pt_tmp_158]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_8,
            in_0=_actx_in_1_mass_0,
            in_1=_pt_tmp_157,
            in_2=_pt_tmp_158,
        )["out"]
    )
    del _pt_tmp_156
    _pt_tmp_160 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_160)
    _pt_tmp_160 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_160)
    _pt_tmp_159 = 0 + _pt_tmp_160
    _pt_tmp_154 = _pt_tmp_155 / _pt_tmp_159
    del _pt_tmp_160
    _pt_tmp_163 = (
        _actx_in_1_energy_0[_pt_tmp_157, _pt_tmp_158]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_8,
            in_0=_actx_in_1_energy_0,
            in_1=_pt_tmp_157,
            in_2=_pt_tmp_158,
        )["out"]
    )
    _pt_tmp_163 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_163)
    _pt_tmp_163 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_163)
    _pt_tmp_162 = 0 + _pt_tmp_163
    _pt_tmp_170 = _pt_tmp_155 * _pt_tmp_154
    del _pt_tmp_163
    _pt_tmp_169 = 0 + _pt_tmp_170
    _pt_tmp_173 = (
        _actx_in_1_momentum_1_0[_pt_tmp_157, _pt_tmp_158]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_8,
            in_0=_actx_in_1_momentum_1_0,
            in_1=_pt_tmp_157,
            in_2=_pt_tmp_158,
        )["out"]
    )
    del _pt_tmp_170
    _pt_tmp_173 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_173)
    _pt_tmp_173 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_173)
    _pt_tmp_172 = 0 + _pt_tmp_173
    _pt_tmp_174 = _pt_tmp_172 / _pt_tmp_159
    del _pt_tmp_173
    _pt_tmp_171 = _pt_tmp_172 * _pt_tmp_174
    _pt_tmp_168 = _pt_tmp_169 + _pt_tmp_171
    _pt_tmp_177 = (
        _actx_in_1_momentum_2_0[_pt_tmp_157, _pt_tmp_158]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_8,
            in_0=_actx_in_1_momentum_2_0,
            in_1=_pt_tmp_157,
            in_2=_pt_tmp_158,
        )["out"]
    )
    del _pt_tmp_169, _pt_tmp_171
    _pt_tmp_177 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_177)
    del _pt_tmp_157, _pt_tmp_158
    _pt_tmp_177 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_177)
    _pt_tmp_176 = 0 + _pt_tmp_177
    _pt_tmp_178 = _pt_tmp_176 / _pt_tmp_159
    del _pt_tmp_177
    _pt_tmp_175 = _pt_tmp_176 * _pt_tmp_178
    _pt_tmp_167 = _pt_tmp_168 + _pt_tmp_175
    _pt_tmp_166 = 0.5 * _pt_tmp_167
    del _pt_tmp_168, _pt_tmp_175
    _pt_tmp_165 = _pt_tmp_162 - _pt_tmp_166
    del _pt_tmp_167
    _pt_tmp_164 = 0.3999999999999999 * _pt_tmp_165
    del _pt_tmp_166
    _pt_tmp_161 = _pt_tmp_162 + _pt_tmp_164
    del _pt_tmp_165
    _pt_tmp_153 = _pt_tmp_154 * _pt_tmp_161
    _pt_data_21 = actx.thaw(npzfile["_pt_data_21"])
    _pt_data_21 = actx.tag((PrefixNamed(prefix="normal_1_b_all"),), _pt_data_21)
    _pt_data_21 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_data_21)
    _pt_data_21 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_data_21)
    _pt_tmp_183 = 2.0 * _pt_data_21
    _pt_tmp_186 = (
        _pt_tmp_155 * _pt_data_21
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_155, _in1=_pt_data_21)["out"]
    )
    _pt_data_22 = actx.thaw(npzfile["_pt_data_22"])
    _pt_data_22 = actx.tag((PrefixNamed(prefix="normal_2_b_all"),), _pt_data_22)
    _pt_data_22 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_data_22)
    _pt_data_22 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_data_22)
    _pt_tmp_187 = (
        _pt_tmp_172 * _pt_data_22
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_172, _in1=_pt_data_22)["out"]
    )
    _pt_tmp_185 = _pt_tmp_186 + _pt_tmp_187
    _pt_data_23 = actx.thaw(npzfile["_pt_data_23"])
    del _pt_tmp_186, _pt_tmp_187
    _pt_data_23 = actx.tag((PrefixNamed(prefix="normal_4_b_all"),), _pt_data_23)
    _pt_data_23 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_data_23)
    _pt_data_23 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_data_23)
    _pt_tmp_188 = (
        _pt_tmp_176 * _pt_data_23
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_176, _in1=_pt_data_23)["out"]
    )
    _pt_tmp_184 = _pt_tmp_185 + _pt_tmp_188
    _pt_tmp_182 = (
        _pt_tmp_183 * _pt_tmp_184
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_10, _in0=_pt_tmp_183, _in1=_pt_tmp_184)[
            "out"
        ]
    )
    del _pt_tmp_185, _pt_tmp_188
    _pt_tmp_181 = _pt_tmp_155 - _pt_tmp_182
    del _pt_tmp_183
    _pt_tmp_180 = _pt_tmp_181 / _pt_tmp_159
    del _pt_tmp_182
    _pt_tmp_196 = _pt_tmp_181 * _pt_tmp_180
    _pt_tmp_195 = 0 + _pt_tmp_196
    _pt_tmp_200 = 2.0 * _pt_data_22
    del _pt_tmp_196
    _pt_tmp_199 = (
        _pt_tmp_200 * _pt_tmp_184
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_10, _in0=_pt_tmp_200, _in1=_pt_tmp_184)[
            "out"
        ]
    )
    _pt_tmp_198 = _pt_tmp_172 - _pt_tmp_199
    del _pt_tmp_200
    _pt_tmp_201 = _pt_tmp_198 / _pt_tmp_159
    del _pt_tmp_199
    _pt_tmp_197 = _pt_tmp_198 * _pt_tmp_201
    _pt_tmp_194 = _pt_tmp_195 + _pt_tmp_197
    _pt_tmp_205 = 2.0 * _pt_data_23
    del _pt_tmp_195, _pt_tmp_197
    _pt_tmp_204 = (
        _pt_tmp_205 * _pt_tmp_184
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_10, _in0=_pt_tmp_205, _in1=_pt_tmp_184)[
            "out"
        ]
    )
    _pt_tmp_203 = _pt_tmp_176 - _pt_tmp_204
    del _pt_tmp_184, _pt_tmp_205
    _pt_tmp_206 = _pt_tmp_203 / _pt_tmp_159
    del _pt_tmp_204
    _pt_tmp_202 = _pt_tmp_203 * _pt_tmp_206
    _pt_tmp_193 = _pt_tmp_194 + _pt_tmp_202
    _pt_tmp_192 = 0.5 * _pt_tmp_193
    del _pt_tmp_194, _pt_tmp_202
    _pt_tmp_191 = _pt_tmp_162 - _pt_tmp_192
    del _pt_tmp_193
    _pt_tmp_190 = 0.3999999999999999 * _pt_tmp_191
    del _pt_tmp_192
    _pt_tmp_189 = _pt_tmp_162 + _pt_tmp_190
    del _pt_tmp_191
    _pt_tmp_179 = _pt_tmp_180 * _pt_tmp_189
    _pt_tmp_152 = _pt_tmp_153 + _pt_tmp_179
    _pt_tmp_151 = 0.5 * _pt_tmp_152
    del _pt_tmp_153, _pt_tmp_179
    _pt_tmp_216 = _pt_tmp_154 * _pt_tmp_154
    del _pt_tmp_152
    _pt_tmp_217 = _pt_tmp_174 * _pt_tmp_174
    _pt_tmp_215 = _pt_tmp_216 + _pt_tmp_217
    _pt_tmp_218 = _pt_tmp_178 * _pt_tmp_178
    _pt_tmp_214 = _pt_tmp_215 + _pt_tmp_218
    _pt_tmp_213 = actx.np.sqrt(_pt_tmp_214)
    del _pt_tmp_215
    _pt_tmp_221 = _pt_tmp_164 / _pt_tmp_159
    del _pt_tmp_214
    _pt_tmp_220 = 1.4 * _pt_tmp_221
    _pt_tmp_219 = actx.np.sqrt(_pt_tmp_220)
    del _pt_tmp_221
    _pt_tmp_212 = _pt_tmp_213 + _pt_tmp_219
    del _pt_tmp_220
    _pt_tmp_211 = actx.np.isnan(_pt_tmp_212)
    del _pt_tmp_213, _pt_tmp_219
    _pt_tmp_227 = _pt_tmp_180 * _pt_tmp_180
    _pt_tmp_228 = _pt_tmp_201 * _pt_tmp_201
    _pt_tmp_226 = _pt_tmp_227 + _pt_tmp_228
    _pt_tmp_229 = _pt_tmp_206 * _pt_tmp_206
    _pt_tmp_225 = _pt_tmp_226 + _pt_tmp_229
    _pt_tmp_224 = actx.np.sqrt(_pt_tmp_225)
    del _pt_tmp_226
    _pt_tmp_232 = _pt_tmp_190 / _pt_tmp_159
    del _pt_tmp_225
    _pt_tmp_231 = 1.4 * _pt_tmp_232
    _pt_tmp_230 = actx.np.sqrt(_pt_tmp_231)
    del _pt_tmp_232
    _pt_tmp_223 = _pt_tmp_224 + _pt_tmp_230
    del _pt_tmp_231
    _pt_tmp_222 = actx.np.isnan(_pt_tmp_223)
    del _pt_tmp_224, _pt_tmp_230
    _pt_tmp_210 = actx.np.logical_or(_pt_tmp_211, _pt_tmp_222)
    _pt_tmp_234 = actx.np.greater(_pt_tmp_212, _pt_tmp_223)
    del _pt_tmp_211, _pt_tmp_222
    _pt_tmp_233 = actx.np.where(_pt_tmp_234, _pt_tmp_212, _pt_tmp_223)
    _pt_tmp_209 = actx.np.where(_pt_tmp_210, np.float64("nan"), _pt_tmp_233)
    del _pt_tmp_212, _pt_tmp_223, _pt_tmp_234
    _pt_tmp_236 = _pt_tmp_162 - _pt_tmp_162
    del _pt_tmp_210, _pt_tmp_233
    _pt_tmp_235 = (
        _pt_tmp_236 * _pt_data_21
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_236, _in1=_pt_data_21)["out"]
    )
    del _pt_tmp_162
    _pt_tmp_208 = _pt_tmp_209 * _pt_tmp_235
    _pt_tmp_207 = _pt_tmp_208 / 2
    del _pt_tmp_235
    _pt_tmp_150 = _pt_tmp_151 - _pt_tmp_207
    del _pt_tmp_208
    _pt_tmp_149 = (
        _pt_tmp_150 * _pt_data_21
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_150, _in1=_pt_data_21)["out"]
    )
    del _pt_tmp_151, _pt_tmp_207
    _pt_tmp_241 = _pt_tmp_174 * _pt_tmp_161
    del _pt_tmp_150
    _pt_tmp_242 = _pt_tmp_201 * _pt_tmp_189
    _pt_tmp_240 = _pt_tmp_241 + _pt_tmp_242
    _pt_tmp_239 = 0.5 * _pt_tmp_240
    del _pt_tmp_241, _pt_tmp_242
    _pt_tmp_245 = (
        _pt_tmp_236 * _pt_data_22
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_236, _in1=_pt_data_22)["out"]
    )
    del _pt_tmp_240
    _pt_tmp_244 = _pt_tmp_209 * _pt_tmp_245
    _pt_tmp_243 = _pt_tmp_244 / 2
    del _pt_tmp_245
    _pt_tmp_238 = _pt_tmp_239 - _pt_tmp_243
    del _pt_tmp_244
    _pt_tmp_237 = (
        _pt_tmp_238 * _pt_data_22
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_238, _in1=_pt_data_22)["out"]
    )
    del _pt_tmp_239, _pt_tmp_243
    _pt_tmp_148 = _pt_tmp_149 + _pt_tmp_237
    del _pt_tmp_238
    _pt_tmp_250 = _pt_tmp_178 * _pt_tmp_161
    del _pt_tmp_149, _pt_tmp_237
    _pt_tmp_251 = _pt_tmp_206 * _pt_tmp_189
    del _pt_tmp_161
    _pt_tmp_249 = _pt_tmp_250 + _pt_tmp_251
    del _pt_tmp_189
    _pt_tmp_248 = 0.5 * _pt_tmp_249
    del _pt_tmp_250, _pt_tmp_251
    _pt_tmp_254 = (
        _pt_tmp_236 * _pt_data_23
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_236, _in1=_pt_data_23)["out"]
    )
    del _pt_tmp_249
    _pt_tmp_253 = _pt_tmp_209 * _pt_tmp_254
    del _pt_tmp_236
    _pt_tmp_252 = _pt_tmp_253 / 2
    del _pt_tmp_254
    _pt_tmp_247 = _pt_tmp_248 - _pt_tmp_252
    del _pt_tmp_253
    _pt_tmp_246 = (
        _pt_tmp_247 * _pt_data_23
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_247, _in1=_pt_data_23)["out"]
    )
    del _pt_tmp_248, _pt_tmp_252
    _pt_tmp_147 = _pt_tmp_148 + _pt_tmp_246
    del _pt_tmp_247
    _pt_data_24 = actx.thaw(npzfile["_pt_data_24"])
    del _pt_tmp_148, _pt_tmp_246
    _pt_data_24 = actx.tag((PrefixNamed(prefix="from_el_indices"),), _pt_data_24)
    _pt_tmp_255 = actx.np.reshape(_pt_data_24, (648000, 1))
    _pt_tmp_255 = actx.tag((PrefixNamed(prefix="from_el_indices"),), _pt_tmp_255)
    _pt_data_25 = actx.thaw(npzfile["_pt_data_25"])
    _pt_data_25 = actx.tag((PrefixNamed(prefix="dof_pick_lists"),), _pt_data_25)
    _pt_tmp_256 = (
        _pt_data_25[_pt_tmp_141]
        if actx.permits_advanced_indexing
        else actx.call_loopy(_pt_t_unit_4, in_0=_pt_data_25, in_1=_pt_tmp_141)["out"]
    )
    _pt_tmp_146 = (
        _pt_tmp_147[_pt_tmp_255, _pt_tmp_256]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_11, in_0=_pt_tmp_147, in_1=_pt_tmp_255, in_2=_pt_tmp_256
        )["out"]
    )
    del _pt_tmp_141
    _pt_tmp_144 = (
        actx.np.where(_pt_tmp_145, _pt_tmp_146, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_6, _in0=_pt_tmp_145, _in1=_pt_tmp_146)["out"]
    )
    del _pt_tmp_147
    _pt_tmp_144 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_144)
    del _pt_tmp_146
    _pt_tmp_144 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_144)
    _pt_tmp_143 = 0 + _pt_tmp_144
    _pt_tmp_142 = 0 + _pt_tmp_143
    del _pt_tmp_144
    _pt_tmp_27 = _pt_tmp_28 + _pt_tmp_142
    del _pt_tmp_143
    _pt_tmp_26 = actx.np.reshape(_pt_tmp_27, (4, 162000, 6))
    del _pt_tmp_142, _pt_tmp_28
    _pt_tmp_26 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_26)
    del _pt_tmp_27
    _pt_tmp_23 = actx.einsum(
        "ijk, jl, jlk -> li", _pt_data_3, _pt_tmp_24, _pt_tmp_26
    )
    _pt_tmp_23 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_23)
    del _pt_tmp_26
    _pt_tmp_3 = _pt_tmp_4 - _pt_tmp_23
    _pt_tmp_0 = actx.einsum("i, jk, ik -> ij", _pt_tmp_1, _pt_data_0, _pt_tmp_3)
    del _pt_tmp_23, _pt_tmp_4
    _pt_tmp_0 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_0)
    del _pt_tmp_3
    _pt_tmp_260 = actx.np.stack(
        [
            _actx_in_1_momentum_0_0,
            _actx_in_1_momentum_1_0,
            _actx_in_1_momentum_2_0,
        ],
        axis=0,
    )
    _pt_tmp_259 = actx.einsum(
        "ijk, jlm, ikm -> kl", _pt_tmp_5, _pt_data_2, _pt_tmp_260
    )
    _pt_tmp_259 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_259)
    del _pt_tmp_260
    _pt_tmp_273 = _pt_tmp_41 + _pt_tmp_67
    _pt_tmp_272 = 0.5 * _pt_tmp_273
    _pt_tmp_277 = _pt_tmp_71 - _pt_tmp_45
    del _pt_tmp_273
    _pt_tmp_276 = (
        _pt_tmp_277 * _pt_data_12
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_277, _in1=_pt_data_12)["out"]
    )
    _pt_tmp_275 = _pt_tmp_93 * _pt_tmp_276
    _pt_tmp_274 = _pt_tmp_275 / 2
    del _pt_tmp_276
    _pt_tmp_271 = _pt_tmp_272 - _pt_tmp_274
    del _pt_tmp_275
    _pt_tmp_270 = (
        _pt_tmp_271 * _pt_data_12
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_271, _in1=_pt_data_12)["out"]
    )
    del _pt_tmp_272, _pt_tmp_274
    _pt_tmp_281 = _pt_tmp_58 + _pt_tmp_84
    del _pt_tmp_271
    _pt_tmp_280 = 0.5 * _pt_tmp_281
    _pt_tmp_284 = (
        _pt_tmp_277 * _pt_data_13
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_277, _in1=_pt_data_13)["out"]
    )
    del _pt_tmp_281
    _pt_tmp_283 = _pt_tmp_93 * _pt_tmp_284
    _pt_tmp_282 = _pt_tmp_283 / 2
    del _pt_tmp_284
    _pt_tmp_279 = _pt_tmp_280 - _pt_tmp_282
    del _pt_tmp_283
    _pt_tmp_278 = (
        _pt_tmp_279 * _pt_data_13
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_279, _in1=_pt_data_13)["out"]
    )
    del _pt_tmp_280, _pt_tmp_282
    _pt_tmp_269 = _pt_tmp_270 + _pt_tmp_278
    del _pt_tmp_279
    _pt_tmp_288 = _pt_tmp_62 + _pt_tmp_88
    del _pt_tmp_270, _pt_tmp_278
    _pt_tmp_287 = 0.5 * _pt_tmp_288
    _pt_tmp_291 = (
        _pt_tmp_277 * _pt_data_14
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_277, _in1=_pt_data_14)["out"]
    )
    del _pt_tmp_288
    _pt_tmp_290 = _pt_tmp_93 * _pt_tmp_291
    del _pt_tmp_277
    _pt_tmp_289 = _pt_tmp_290 / 2
    del _pt_tmp_291
    _pt_tmp_286 = _pt_tmp_287 - _pt_tmp_289
    del _pt_tmp_290
    _pt_tmp_285 = (
        _pt_tmp_286 * _pt_data_14
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_286, _in1=_pt_data_14)["out"]
    )
    del _pt_tmp_287, _pt_tmp_289
    _pt_tmp_268 = _pt_tmp_269 + _pt_tmp_285
    del _pt_tmp_286
    _pt_tmp_267 = (
        _pt_tmp_268[_pt_tmp_139, _pt_tmp_140]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_5, in_0=_pt_tmp_268, in_1=_pt_tmp_139, in_2=_pt_tmp_140
        )["out"]
    )
    del _pt_tmp_269, _pt_tmp_285
    _pt_tmp_266 = (
        actx.np.where(_pt_tmp_31, _pt_tmp_267, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_6, _in0=_pt_tmp_31, _in1=_pt_tmp_267)["out"]
    )
    del _pt_tmp_268
    _pt_tmp_266 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_266)
    del _pt_tmp_267
    _pt_tmp_266 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_266)
    _pt_tmp_265 = 0 + _pt_tmp_266
    _pt_tmp_264 = 0 + _pt_tmp_265
    del _pt_tmp_266
    _pt_tmp_301 = _pt_tmp_155 + _pt_tmp_181
    del _pt_tmp_265
    _pt_tmp_300 = 0.5 * _pt_tmp_301
    _pt_tmp_305 = _pt_tmp_159 - _pt_tmp_159
    del _pt_tmp_301
    _pt_tmp_304 = (
        _pt_tmp_305 * _pt_data_21
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_305, _in1=_pt_data_21)["out"]
    )
    _pt_tmp_303 = _pt_tmp_209 * _pt_tmp_304
    _pt_tmp_302 = _pt_tmp_303 / 2
    del _pt_tmp_304
    _pt_tmp_299 = _pt_tmp_300 - _pt_tmp_302
    del _pt_tmp_303
    _pt_tmp_298 = (
        _pt_tmp_299 * _pt_data_21
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_299, _in1=_pt_data_21)["out"]
    )
    del _pt_tmp_300, _pt_tmp_302
    _pt_tmp_309 = _pt_tmp_172 + _pt_tmp_198
    del _pt_tmp_299
    _pt_tmp_308 = 0.5 * _pt_tmp_309
    _pt_tmp_312 = (
        _pt_tmp_305 * _pt_data_22
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_305, _in1=_pt_data_22)["out"]
    )
    del _pt_tmp_309
    _pt_tmp_311 = _pt_tmp_209 * _pt_tmp_312
    _pt_tmp_310 = _pt_tmp_311 / 2
    del _pt_tmp_312
    _pt_tmp_307 = _pt_tmp_308 - _pt_tmp_310
    del _pt_tmp_311
    _pt_tmp_306 = (
        _pt_tmp_307 * _pt_data_22
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_307, _in1=_pt_data_22)["out"]
    )
    del _pt_tmp_308, _pt_tmp_310
    _pt_tmp_297 = _pt_tmp_298 + _pt_tmp_306
    del _pt_tmp_307
    _pt_tmp_316 = _pt_tmp_176 + _pt_tmp_203
    del _pt_tmp_298, _pt_tmp_306
    _pt_tmp_315 = 0.5 * _pt_tmp_316
    _pt_tmp_319 = (
        _pt_tmp_305 * _pt_data_23
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_305, _in1=_pt_data_23)["out"]
    )
    del _pt_tmp_316
    _pt_tmp_318 = _pt_tmp_209 * _pt_tmp_319
    del _pt_tmp_305
    _pt_tmp_317 = _pt_tmp_318 / 2
    del _pt_tmp_319
    _pt_tmp_314 = _pt_tmp_315 - _pt_tmp_317
    del _pt_tmp_318
    _pt_tmp_313 = (
        _pt_tmp_314 * _pt_data_23
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_314, _in1=_pt_data_23)["out"]
    )
    del _pt_tmp_315, _pt_tmp_317
    _pt_tmp_296 = _pt_tmp_297 + _pt_tmp_313
    del _pt_tmp_314
    _pt_tmp_295 = (
        _pt_tmp_296[_pt_tmp_255, _pt_tmp_256]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_11, in_0=_pt_tmp_296, in_1=_pt_tmp_255, in_2=_pt_tmp_256
        )["out"]
    )
    del _pt_tmp_297, _pt_tmp_313
    _pt_tmp_294 = (
        actx.np.where(_pt_tmp_145, _pt_tmp_295, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_6, _in0=_pt_tmp_145, _in1=_pt_tmp_295)["out"]
    )
    del _pt_tmp_296
    _pt_tmp_294 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_294)
    del _pt_tmp_295
    _pt_tmp_294 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_294)
    _pt_tmp_293 = 0 + _pt_tmp_294
    _pt_tmp_292 = 0 + _pt_tmp_293
    del _pt_tmp_294
    _pt_tmp_263 = _pt_tmp_264 + _pt_tmp_292
    del _pt_tmp_293
    _pt_tmp_262 = actx.np.reshape(_pt_tmp_263, (4, 162000, 6))
    del _pt_tmp_264, _pt_tmp_292
    _pt_tmp_262 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_262)
    del _pt_tmp_263
    _pt_tmp_261 = actx.einsum(
        "ijk, jl, jlk -> li", _pt_data_3, _pt_tmp_24, _pt_tmp_262
    )
    _pt_tmp_261 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_261)
    del _pt_tmp_262
    _pt_tmp_258 = _pt_tmp_259 - _pt_tmp_261
    _pt_tmp_257 = actx.einsum("i, jk, ik -> ij", _pt_tmp_1, _pt_data_0, _pt_tmp_258)
    del _pt_tmp_259, _pt_tmp_261
    _pt_tmp_257 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_257)
    del _pt_tmp_258
    _pt_tmp_325 = _pt_tmp_7 * _pt_tmp_7
    _pt_tmp_324 = _actx_in_1_mass_0 * _pt_tmp_325
    _pt_tmp_326 = 1.0 * _pt_tmp_9
    del _pt_tmp_325
    _pt_tmp_323 = _pt_tmp_324 + _pt_tmp_326
    _pt_tmp_329 = _pt_tmp_7 * _pt_tmp_17
    del _pt_tmp_324
    _pt_tmp_328 = _actx_in_1_mass_0 * _pt_tmp_329
    _pt_tmp_330 = 0.0 * _pt_tmp_9
    del _pt_tmp_329
    _pt_tmp_327 = _pt_tmp_328 + _pt_tmp_330
    del _pt_tmp_9
    _pt_tmp_333 = _pt_tmp_7 * _pt_tmp_19
    del _pt_tmp_328
    _pt_tmp_332 = _actx_in_1_mass_0 * _pt_tmp_333
    _pt_tmp_331 = _pt_tmp_332 + _pt_tmp_330
    del _pt_tmp_333
    _pt_tmp_334 = actx.np.stack([_pt_tmp_323, _pt_tmp_327, _pt_tmp_331], axis=0)
    del _pt_tmp_332
    _pt_tmp_322 = actx.einsum(
        "ijk, jlm, ikm -> kl", _pt_tmp_5, _pt_data_2, _pt_tmp_334
    )
    del _pt_tmp_323, _pt_tmp_327, _pt_tmp_331
    _pt_tmp_322 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_322)
    del _pt_tmp_334
    _pt_tmp_349 = _pt_tmp_45 * _pt_tmp_100
    _pt_tmp_350 = 1.0 * _pt_tmp_50
    del _pt_tmp_100
    _pt_tmp_348 = _pt_tmp_349 + _pt_tmp_350
    _pt_tmp_352 = _pt_tmp_71 * _pt_tmp_111
    del _pt_tmp_349
    _pt_tmp_353 = 1.0 * _pt_tmp_76
    del _pt_tmp_111
    _pt_tmp_351 = _pt_tmp_352 + _pt_tmp_353
    _pt_tmp_347 = _pt_tmp_348 + _pt_tmp_351
    del _pt_tmp_352
    _pt_tmp_346 = 0.5 * _pt_tmp_347
    del _pt_tmp_348, _pt_tmp_351
    _pt_tmp_357 = _pt_tmp_67 - _pt_tmp_41
    del _pt_tmp_347
    _pt_tmp_356 = (
        _pt_tmp_357 * _pt_data_12
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_357, _in1=_pt_data_12)["out"]
    )
    del _pt_tmp_41, _pt_tmp_67
    _pt_tmp_355 = _pt_tmp_93 * _pt_tmp_356
    _pt_tmp_354 = _pt_tmp_355 / 2
    del _pt_tmp_356
    _pt_tmp_345 = _pt_tmp_346 - _pt_tmp_354
    del _pt_tmp_355
    _pt_tmp_344 = (
        _pt_tmp_345 * _pt_data_12
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_345, _in1=_pt_data_12)["out"]
    )
    del _pt_tmp_346, _pt_tmp_354
    _pt_tmp_364 = _pt_tmp_40 * _pt_tmp_60
    del _pt_tmp_345
    _pt_tmp_363 = _pt_tmp_45 * _pt_tmp_364
    _pt_tmp_365 = 0.0 * _pt_tmp_50
    del _pt_tmp_364
    _pt_tmp_362 = _pt_tmp_363 + _pt_tmp_365
    del _pt_tmp_50
    _pt_tmp_368 = _pt_tmp_66 * _pt_tmp_86
    del _pt_tmp_363
    _pt_tmp_367 = _pt_tmp_71 * _pt_tmp_368
    _pt_tmp_369 = 0.0 * _pt_tmp_76
    del _pt_tmp_368
    _pt_tmp_366 = _pt_tmp_367 + _pt_tmp_369
    del _pt_tmp_76
    _pt_tmp_361 = _pt_tmp_362 + _pt_tmp_366
    del _pt_tmp_367
    _pt_tmp_360 = 0.5 * _pt_tmp_361
    del _pt_tmp_362, _pt_tmp_366
    _pt_tmp_372 = (
        _pt_tmp_357 * _pt_data_13
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_357, _in1=_pt_data_13)["out"]
    )
    del _pt_tmp_361
    _pt_tmp_371 = _pt_tmp_93 * _pt_tmp_372
    _pt_tmp_370 = _pt_tmp_371 / 2
    del _pt_tmp_372
    _pt_tmp_359 = _pt_tmp_360 - _pt_tmp_370
    del _pt_tmp_371
    _pt_tmp_358 = (
        _pt_tmp_359 * _pt_data_13
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_359, _in1=_pt_data_13)["out"]
    )
    del _pt_tmp_360, _pt_tmp_370
    _pt_tmp_343 = _pt_tmp_344 + _pt_tmp_358
    del _pt_tmp_359
    _pt_tmp_379 = _pt_tmp_40 * _pt_tmp_64
    del _pt_tmp_344, _pt_tmp_358
    _pt_tmp_378 = _pt_tmp_45 * _pt_tmp_379
    _pt_tmp_377 = _pt_tmp_378 + _pt_tmp_365
    del _pt_tmp_379
    _pt_tmp_382 = _pt_tmp_66 * _pt_tmp_90
    del _pt_tmp_378
    _pt_tmp_381 = _pt_tmp_71 * _pt_tmp_382
    _pt_tmp_380 = _pt_tmp_381 + _pt_tmp_369
    del _pt_tmp_382
    _pt_tmp_376 = _pt_tmp_377 + _pt_tmp_380
    del _pt_tmp_381
    _pt_tmp_375 = 0.5 * _pt_tmp_376
    del _pt_tmp_377, _pt_tmp_380
    _pt_tmp_385 = (
        _pt_tmp_357 * _pt_data_14
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_357, _in1=_pt_data_14)["out"]
    )
    del _pt_tmp_376
    _pt_tmp_384 = _pt_tmp_93 * _pt_tmp_385
    del _pt_tmp_357
    _pt_tmp_383 = _pt_tmp_384 / 2
    del _pt_tmp_385
    _pt_tmp_374 = _pt_tmp_375 - _pt_tmp_383
    del _pt_tmp_384
    _pt_tmp_373 = (
        _pt_tmp_374 * _pt_data_14
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_374, _in1=_pt_data_14)["out"]
    )
    del _pt_tmp_375, _pt_tmp_383
    _pt_tmp_342 = _pt_tmp_343 + _pt_tmp_373
    del _pt_tmp_374
    _pt_tmp_341 = (
        _pt_tmp_342[_pt_tmp_139, _pt_tmp_140]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_5, in_0=_pt_tmp_342, in_1=_pt_tmp_139, in_2=_pt_tmp_140
        )["out"]
    )
    del _pt_tmp_343, _pt_tmp_373
    _pt_tmp_340 = (
        actx.np.where(_pt_tmp_31, _pt_tmp_341, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_6, _in0=_pt_tmp_31, _in1=_pt_tmp_341)["out"]
    )
    del _pt_tmp_342
    _pt_tmp_340 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_340)
    del _pt_tmp_341
    _pt_tmp_340 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_340)
    _pt_tmp_339 = 0 + _pt_tmp_340
    _pt_tmp_338 = 0 + _pt_tmp_339
    del _pt_tmp_340
    _pt_tmp_397 = _pt_tmp_159 * _pt_tmp_216
    del _pt_tmp_339
    _pt_tmp_398 = 1.0 * _pt_tmp_164
    del _pt_tmp_216
    _pt_tmp_396 = _pt_tmp_397 + _pt_tmp_398
    _pt_tmp_400 = _pt_tmp_159 * _pt_tmp_227
    del _pt_tmp_397
    _pt_tmp_401 = 1.0 * _pt_tmp_190
    del _pt_tmp_227
    _pt_tmp_399 = _pt_tmp_400 + _pt_tmp_401
    _pt_tmp_395 = _pt_tmp_396 + _pt_tmp_399
    del _pt_tmp_400
    _pt_tmp_394 = 0.5 * _pt_tmp_395
    del _pt_tmp_396, _pt_tmp_399
    _pt_tmp_405 = _pt_tmp_181 - _pt_tmp_155
    del _pt_tmp_395
    _pt_tmp_404 = (
        _pt_tmp_405 * _pt_data_21
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_405, _in1=_pt_data_21)["out"]
    )
    del _pt_tmp_155, _pt_tmp_181
    _pt_tmp_403 = _pt_tmp_209 * _pt_tmp_404
    _pt_tmp_402 = _pt_tmp_403 / 2
    del _pt_tmp_404
    _pt_tmp_393 = _pt_tmp_394 - _pt_tmp_402
    del _pt_tmp_403
    _pt_tmp_392 = (
        _pt_tmp_393 * _pt_data_21
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_393, _in1=_pt_data_21)["out"]
    )
    del _pt_tmp_394, _pt_tmp_402
    _pt_tmp_412 = _pt_tmp_154 * _pt_tmp_174
    del _pt_tmp_393
    _pt_tmp_411 = _pt_tmp_159 * _pt_tmp_412
    _pt_tmp_413 = 0.0 * _pt_tmp_164
    del _pt_tmp_412
    _pt_tmp_410 = _pt_tmp_411 + _pt_tmp_413
    del _pt_tmp_164
    _pt_tmp_416 = _pt_tmp_180 * _pt_tmp_201
    del _pt_tmp_411
    _pt_tmp_415 = _pt_tmp_159 * _pt_tmp_416
    _pt_tmp_417 = 0.0 * _pt_tmp_190
    del _pt_tmp_416
    _pt_tmp_414 = _pt_tmp_415 + _pt_tmp_417
    del _pt_tmp_190
    _pt_tmp_409 = _pt_tmp_410 + _pt_tmp_414
    del _pt_tmp_415
    _pt_tmp_408 = 0.5 * _pt_tmp_409
    del _pt_tmp_410, _pt_tmp_414
    _pt_tmp_420 = (
        _pt_tmp_405 * _pt_data_22
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_405, _in1=_pt_data_22)["out"]
    )
    del _pt_tmp_409
    _pt_tmp_419 = _pt_tmp_209 * _pt_tmp_420
    _pt_tmp_418 = _pt_tmp_419 / 2
    del _pt_tmp_420
    _pt_tmp_407 = _pt_tmp_408 - _pt_tmp_418
    del _pt_tmp_419
    _pt_tmp_406 = (
        _pt_tmp_407 * _pt_data_22
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_407, _in1=_pt_data_22)["out"]
    )
    del _pt_tmp_408, _pt_tmp_418
    _pt_tmp_391 = _pt_tmp_392 + _pt_tmp_406
    del _pt_tmp_407
    _pt_tmp_427 = _pt_tmp_154 * _pt_tmp_178
    del _pt_tmp_392, _pt_tmp_406
    _pt_tmp_426 = _pt_tmp_159 * _pt_tmp_427
    _pt_tmp_425 = _pt_tmp_426 + _pt_tmp_413
    del _pt_tmp_427
    _pt_tmp_430 = _pt_tmp_180 * _pt_tmp_206
    del _pt_tmp_426
    _pt_tmp_429 = _pt_tmp_159 * _pt_tmp_430
    _pt_tmp_428 = _pt_tmp_429 + _pt_tmp_417
    del _pt_tmp_430
    _pt_tmp_424 = _pt_tmp_425 + _pt_tmp_428
    del _pt_tmp_429
    _pt_tmp_423 = 0.5 * _pt_tmp_424
    del _pt_tmp_425, _pt_tmp_428
    _pt_tmp_433 = (
        _pt_tmp_405 * _pt_data_23
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_405, _in1=_pt_data_23)["out"]
    )
    del _pt_tmp_424
    _pt_tmp_432 = _pt_tmp_209 * _pt_tmp_433
    del _pt_tmp_405
    _pt_tmp_431 = _pt_tmp_432 / 2
    del _pt_tmp_433
    _pt_tmp_422 = _pt_tmp_423 - _pt_tmp_431
    del _pt_tmp_432
    _pt_tmp_421 = (
        _pt_tmp_422 * _pt_data_23
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_422, _in1=_pt_data_23)["out"]
    )
    del _pt_tmp_423, _pt_tmp_431
    _pt_tmp_390 = _pt_tmp_391 + _pt_tmp_421
    del _pt_tmp_422
    _pt_tmp_389 = (
        _pt_tmp_390[_pt_tmp_255, _pt_tmp_256]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_11, in_0=_pt_tmp_390, in_1=_pt_tmp_255, in_2=_pt_tmp_256
        )["out"]
    )
    del _pt_tmp_391, _pt_tmp_421
    _pt_tmp_388 = (
        actx.np.where(_pt_tmp_145, _pt_tmp_389, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_6, _in0=_pt_tmp_145, _in1=_pt_tmp_389)["out"]
    )
    del _pt_tmp_390
    _pt_tmp_388 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_388)
    del _pt_tmp_389
    _pt_tmp_388 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_388)
    _pt_tmp_387 = 0 + _pt_tmp_388
    _pt_tmp_386 = 0 + _pt_tmp_387
    del _pt_tmp_388
    _pt_tmp_337 = _pt_tmp_338 + _pt_tmp_386
    del _pt_tmp_387
    _pt_tmp_336 = actx.np.reshape(_pt_tmp_337, (4, 162000, 6))
    del _pt_tmp_338, _pt_tmp_386
    _pt_tmp_336 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_336)
    del _pt_tmp_337
    _pt_tmp_335 = actx.einsum(
        "ijk, jl, jlk -> li", _pt_data_3, _pt_tmp_24, _pt_tmp_336
    )
    _pt_tmp_335 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_335)
    del _pt_tmp_336
    _pt_tmp_321 = _pt_tmp_322 - _pt_tmp_335
    _pt_tmp_320 = actx.einsum("i, jk, ik -> ij", _pt_tmp_1, _pt_data_0, _pt_tmp_321)
    del _pt_tmp_322, _pt_tmp_335
    _pt_tmp_320 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_320)
    del _pt_tmp_321
    _pt_tmp_439 = _pt_tmp_17 * _pt_tmp_7
    _pt_tmp_438 = _actx_in_1_mass_0 * _pt_tmp_439
    _pt_tmp_437 = _pt_tmp_438 + _pt_tmp_330
    del _pt_tmp_439
    _pt_tmp_442 = _pt_tmp_17 * _pt_tmp_17
    del _pt_tmp_438
    _pt_tmp_441 = _actx_in_1_mass_0 * _pt_tmp_442
    _pt_tmp_440 = _pt_tmp_441 + _pt_tmp_326
    del _pt_tmp_442
    _pt_tmp_445 = _pt_tmp_17 * _pt_tmp_19
    del _pt_tmp_441
    _pt_tmp_444 = _actx_in_1_mass_0 * _pt_tmp_445
    _pt_tmp_443 = _pt_tmp_444 + _pt_tmp_330
    del _pt_tmp_445
    _pt_tmp_446 = actx.np.stack([_pt_tmp_437, _pt_tmp_440, _pt_tmp_443], axis=0)
    del _pt_tmp_444
    _pt_tmp_436 = actx.einsum(
        "ijk, jlm, ikm -> kl", _pt_tmp_5, _pt_data_2, _pt_tmp_446
    )
    del _pt_tmp_437, _pt_tmp_440, _pt_tmp_443
    _pt_tmp_436 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_436)
    del _pt_tmp_446
    _pt_tmp_462 = _pt_tmp_60 * _pt_tmp_40
    _pt_tmp_461 = _pt_tmp_45 * _pt_tmp_462
    _pt_tmp_460 = _pt_tmp_461 + _pt_tmp_365
    del _pt_tmp_462
    _pt_tmp_465 = _pt_tmp_86 * _pt_tmp_66
    del _pt_tmp_461
    _pt_tmp_464 = _pt_tmp_71 * _pt_tmp_465
    _pt_tmp_463 = _pt_tmp_464 + _pt_tmp_369
    del _pt_tmp_465
    _pt_tmp_459 = _pt_tmp_460 + _pt_tmp_463
    del _pt_tmp_464
    _pt_tmp_458 = 0.5 * _pt_tmp_459
    del _pt_tmp_460, _pt_tmp_463
    _pt_tmp_469 = _pt_tmp_84 - _pt_tmp_58
    del _pt_tmp_459
    _pt_tmp_468 = (
        _pt_tmp_469 * _pt_data_12
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_469, _in1=_pt_data_12)["out"]
    )
    del _pt_tmp_58, _pt_tmp_84
    _pt_tmp_467 = _pt_tmp_93 * _pt_tmp_468
    _pt_tmp_466 = _pt_tmp_467 / 2
    del _pt_tmp_468
    _pt_tmp_457 = _pt_tmp_458 - _pt_tmp_466
    del _pt_tmp_467
    _pt_tmp_456 = (
        _pt_tmp_457 * _pt_data_12
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_457, _in1=_pt_data_12)["out"]
    )
    del _pt_tmp_458, _pt_tmp_466
    _pt_tmp_475 = _pt_tmp_45 * _pt_tmp_101
    del _pt_tmp_457
    _pt_tmp_474 = _pt_tmp_475 + _pt_tmp_350
    del _pt_tmp_101
    _pt_tmp_477 = _pt_tmp_71 * _pt_tmp_112
    del _pt_tmp_475
    _pt_tmp_476 = _pt_tmp_477 + _pt_tmp_353
    del _pt_tmp_112
    _pt_tmp_473 = _pt_tmp_474 + _pt_tmp_476
    del _pt_tmp_477
    _pt_tmp_472 = 0.5 * _pt_tmp_473
    del _pt_tmp_474, _pt_tmp_476
    _pt_tmp_480 = (
        _pt_tmp_469 * _pt_data_13
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_469, _in1=_pt_data_13)["out"]
    )
    del _pt_tmp_473
    _pt_tmp_479 = _pt_tmp_93 * _pt_tmp_480
    _pt_tmp_478 = _pt_tmp_479 / 2
    del _pt_tmp_480
    _pt_tmp_471 = _pt_tmp_472 - _pt_tmp_478
    del _pt_tmp_479
    _pt_tmp_470 = (
        _pt_tmp_471 * _pt_data_13
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_471, _in1=_pt_data_13)["out"]
    )
    del _pt_tmp_472, _pt_tmp_478
    _pt_tmp_455 = _pt_tmp_456 + _pt_tmp_470
    del _pt_tmp_471
    _pt_tmp_487 = _pt_tmp_60 * _pt_tmp_64
    del _pt_tmp_456, _pt_tmp_470
    _pt_tmp_486 = _pt_tmp_45 * _pt_tmp_487
    _pt_tmp_485 = _pt_tmp_486 + _pt_tmp_365
    del _pt_tmp_487
    _pt_tmp_490 = _pt_tmp_86 * _pt_tmp_90
    del _pt_tmp_486
    _pt_tmp_489 = _pt_tmp_71 * _pt_tmp_490
    _pt_tmp_488 = _pt_tmp_489 + _pt_tmp_369
    del _pt_tmp_490
    _pt_tmp_484 = _pt_tmp_485 + _pt_tmp_488
    del _pt_tmp_489
    _pt_tmp_483 = 0.5 * _pt_tmp_484
    del _pt_tmp_485, _pt_tmp_488
    _pt_tmp_493 = (
        _pt_tmp_469 * _pt_data_14
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_469, _in1=_pt_data_14)["out"]
    )
    del _pt_tmp_484
    _pt_tmp_492 = _pt_tmp_93 * _pt_tmp_493
    del _pt_tmp_469
    _pt_tmp_491 = _pt_tmp_492 / 2
    del _pt_tmp_493
    _pt_tmp_482 = _pt_tmp_483 - _pt_tmp_491
    del _pt_tmp_492
    _pt_tmp_481 = (
        _pt_tmp_482 * _pt_data_14
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_482, _in1=_pt_data_14)["out"]
    )
    del _pt_tmp_483, _pt_tmp_491
    _pt_tmp_454 = _pt_tmp_455 + _pt_tmp_481
    del _pt_tmp_482
    _pt_tmp_453 = (
        _pt_tmp_454[_pt_tmp_139, _pt_tmp_140]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_5, in_0=_pt_tmp_454, in_1=_pt_tmp_139, in_2=_pt_tmp_140
        )["out"]
    )
    del _pt_tmp_455, _pt_tmp_481
    _pt_tmp_452 = (
        actx.np.where(_pt_tmp_31, _pt_tmp_453, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_6, _in0=_pt_tmp_31, _in1=_pt_tmp_453)["out"]
    )
    del _pt_tmp_454
    _pt_tmp_452 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_452)
    del _pt_tmp_453
    _pt_tmp_452 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_452)
    _pt_tmp_451 = 0 + _pt_tmp_452
    _pt_tmp_450 = 0 + _pt_tmp_451
    del _pt_tmp_452
    _pt_tmp_506 = _pt_tmp_174 * _pt_tmp_154
    del _pt_tmp_451
    _pt_tmp_505 = _pt_tmp_159 * _pt_tmp_506
    _pt_tmp_504 = _pt_tmp_505 + _pt_tmp_413
    del _pt_tmp_506
    _pt_tmp_509 = _pt_tmp_201 * _pt_tmp_180
    del _pt_tmp_505
    _pt_tmp_508 = _pt_tmp_159 * _pt_tmp_509
    _pt_tmp_507 = _pt_tmp_508 + _pt_tmp_417
    del _pt_tmp_509
    _pt_tmp_503 = _pt_tmp_504 + _pt_tmp_507
    del _pt_tmp_508
    _pt_tmp_502 = 0.5 * _pt_tmp_503
    del _pt_tmp_504, _pt_tmp_507
    _pt_tmp_513 = _pt_tmp_198 - _pt_tmp_172
    del _pt_tmp_503
    _pt_tmp_512 = (
        _pt_tmp_513 * _pt_data_21
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_513, _in1=_pt_data_21)["out"]
    )
    del _pt_tmp_172, _pt_tmp_198
    _pt_tmp_511 = _pt_tmp_209 * _pt_tmp_512
    _pt_tmp_510 = _pt_tmp_511 / 2
    del _pt_tmp_512
    _pt_tmp_501 = _pt_tmp_502 - _pt_tmp_510
    del _pt_tmp_511
    _pt_tmp_500 = (
        _pt_tmp_501 * _pt_data_21
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_501, _in1=_pt_data_21)["out"]
    )
    del _pt_tmp_502, _pt_tmp_510
    _pt_tmp_519 = _pt_tmp_159 * _pt_tmp_217
    del _pt_tmp_501
    _pt_tmp_518 = _pt_tmp_519 + _pt_tmp_398
    del _pt_tmp_217
    _pt_tmp_521 = _pt_tmp_159 * _pt_tmp_228
    del _pt_tmp_519
    _pt_tmp_520 = _pt_tmp_521 + _pt_tmp_401
    del _pt_tmp_228
    _pt_tmp_517 = _pt_tmp_518 + _pt_tmp_520
    del _pt_tmp_521
    _pt_tmp_516 = 0.5 * _pt_tmp_517
    del _pt_tmp_518, _pt_tmp_520
    _pt_tmp_524 = (
        _pt_tmp_513 * _pt_data_22
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_513, _in1=_pt_data_22)["out"]
    )
    del _pt_tmp_517
    _pt_tmp_523 = _pt_tmp_209 * _pt_tmp_524
    _pt_tmp_522 = _pt_tmp_523 / 2
    del _pt_tmp_524
    _pt_tmp_515 = _pt_tmp_516 - _pt_tmp_522
    del _pt_tmp_523
    _pt_tmp_514 = (
        _pt_tmp_515 * _pt_data_22
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_515, _in1=_pt_data_22)["out"]
    )
    del _pt_tmp_516, _pt_tmp_522
    _pt_tmp_499 = _pt_tmp_500 + _pt_tmp_514
    del _pt_tmp_515
    _pt_tmp_531 = _pt_tmp_174 * _pt_tmp_178
    del _pt_tmp_500, _pt_tmp_514
    _pt_tmp_530 = _pt_tmp_159 * _pt_tmp_531
    _pt_tmp_529 = _pt_tmp_530 + _pt_tmp_413
    del _pt_tmp_531
    _pt_tmp_534 = _pt_tmp_201 * _pt_tmp_206
    del _pt_tmp_530
    _pt_tmp_533 = _pt_tmp_159 * _pt_tmp_534
    _pt_tmp_532 = _pt_tmp_533 + _pt_tmp_417
    del _pt_tmp_534
    _pt_tmp_528 = _pt_tmp_529 + _pt_tmp_532
    del _pt_tmp_533
    _pt_tmp_527 = 0.5 * _pt_tmp_528
    del _pt_tmp_529, _pt_tmp_532
    _pt_tmp_537 = (
        _pt_tmp_513 * _pt_data_23
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_513, _in1=_pt_data_23)["out"]
    )
    del _pt_tmp_528
    _pt_tmp_536 = _pt_tmp_209 * _pt_tmp_537
    del _pt_tmp_513
    _pt_tmp_535 = _pt_tmp_536 / 2
    del _pt_tmp_537
    _pt_tmp_526 = _pt_tmp_527 - _pt_tmp_535
    del _pt_tmp_536
    _pt_tmp_525 = (
        _pt_tmp_526 * _pt_data_23
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_526, _in1=_pt_data_23)["out"]
    )
    del _pt_tmp_527, _pt_tmp_535
    _pt_tmp_498 = _pt_tmp_499 + _pt_tmp_525
    del _pt_tmp_526
    _pt_tmp_497 = (
        _pt_tmp_498[_pt_tmp_255, _pt_tmp_256]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_11, in_0=_pt_tmp_498, in_1=_pt_tmp_255, in_2=_pt_tmp_256
        )["out"]
    )
    del _pt_tmp_499, _pt_tmp_525
    _pt_tmp_496 = (
        actx.np.where(_pt_tmp_145, _pt_tmp_497, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_6, _in0=_pt_tmp_145, _in1=_pt_tmp_497)["out"]
    )
    del _pt_tmp_498
    _pt_tmp_496 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_496)
    del _pt_tmp_497
    _pt_tmp_496 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_496)
    _pt_tmp_495 = 0 + _pt_tmp_496
    _pt_tmp_494 = 0 + _pt_tmp_495
    del _pt_tmp_496
    _pt_tmp_449 = _pt_tmp_450 + _pt_tmp_494
    del _pt_tmp_495
    _pt_tmp_448 = actx.np.reshape(_pt_tmp_449, (4, 162000, 6))
    del _pt_tmp_450, _pt_tmp_494
    _pt_tmp_448 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_448)
    del _pt_tmp_449
    _pt_tmp_447 = actx.einsum(
        "ijk, jl, jlk -> li", _pt_data_3, _pt_tmp_24, _pt_tmp_448
    )
    _pt_tmp_447 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_447)
    del _pt_tmp_448
    _pt_tmp_435 = _pt_tmp_436 - _pt_tmp_447
    _pt_tmp_434 = actx.einsum("i, jk, ik -> ij", _pt_tmp_1, _pt_data_0, _pt_tmp_435)
    del _pt_tmp_436, _pt_tmp_447
    _pt_tmp_434 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_434)
    del _pt_tmp_435
    _pt_tmp_543 = _pt_tmp_19 * _pt_tmp_7
    _pt_tmp_542 = _actx_in_1_mass_0 * _pt_tmp_543
    del _pt_tmp_7
    _pt_tmp_541 = _pt_tmp_542 + _pt_tmp_330
    del _pt_tmp_543
    _pt_tmp_546 = _pt_tmp_19 * _pt_tmp_17
    del _pt_tmp_542
    _pt_tmp_545 = _actx_in_1_mass_0 * _pt_tmp_546
    del _pt_tmp_17
    _pt_tmp_544 = _pt_tmp_545 + _pt_tmp_330
    del _pt_tmp_546
    _pt_tmp_549 = _pt_tmp_19 * _pt_tmp_19
    del _pt_tmp_330, _pt_tmp_545
    _pt_tmp_548 = _actx_in_1_mass_0 * _pt_tmp_549
    del _pt_tmp_19
    _pt_tmp_547 = _pt_tmp_548 + _pt_tmp_326
    del _pt_tmp_549
    _pt_tmp_550 = actx.np.stack([_pt_tmp_541, _pt_tmp_544, _pt_tmp_547], axis=0)
    del _pt_tmp_326, _pt_tmp_548
    _pt_tmp_540 = actx.einsum(
        "ijk, jlm, ikm -> kl", _pt_tmp_5, _pt_data_2, _pt_tmp_550
    )
    del _pt_tmp_541, _pt_tmp_544, _pt_tmp_547
    _pt_tmp_540 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_540)
    del _pt_tmp_5, _pt_tmp_550
    _pt_tmp_566 = _pt_tmp_64 * _pt_tmp_40
    _pt_tmp_565 = _pt_tmp_45 * _pt_tmp_566
    del _pt_tmp_40
    _pt_tmp_564 = _pt_tmp_565 + _pt_tmp_365
    del _pt_tmp_566
    _pt_tmp_569 = _pt_tmp_90 * _pt_tmp_66
    del _pt_tmp_565
    _pt_tmp_568 = _pt_tmp_71 * _pt_tmp_569
    del _pt_tmp_66
    _pt_tmp_567 = _pt_tmp_568 + _pt_tmp_369
    del _pt_tmp_569
    _pt_tmp_563 = _pt_tmp_564 + _pt_tmp_567
    del _pt_tmp_568
    _pt_tmp_562 = 0.5 * _pt_tmp_563
    del _pt_tmp_564, _pt_tmp_567
    _pt_tmp_573 = _pt_tmp_88 - _pt_tmp_62
    del _pt_tmp_563
    _pt_tmp_572 = (
        _pt_tmp_573 * _pt_data_12
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_573, _in1=_pt_data_12)["out"]
    )
    del _pt_tmp_62, _pt_tmp_88
    _pt_tmp_571 = _pt_tmp_93 * _pt_tmp_572
    _pt_tmp_570 = _pt_tmp_571 / 2
    del _pt_tmp_572
    _pt_tmp_561 = _pt_tmp_562 - _pt_tmp_570
    del _pt_tmp_571
    _pt_tmp_560 = (
        _pt_tmp_561 * _pt_data_12
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_561, _in1=_pt_data_12)["out"]
    )
    del _pt_tmp_562, _pt_tmp_570
    _pt_tmp_580 = _pt_tmp_64 * _pt_tmp_60
    del _pt_tmp_561
    _pt_tmp_579 = _pt_tmp_45 * _pt_tmp_580
    del _pt_tmp_60, _pt_tmp_64
    _pt_tmp_578 = _pt_tmp_579 + _pt_tmp_365
    del _pt_tmp_580
    _pt_tmp_583 = _pt_tmp_90 * _pt_tmp_86
    del _pt_tmp_365, _pt_tmp_579
    _pt_tmp_582 = _pt_tmp_71 * _pt_tmp_583
    del _pt_tmp_86, _pt_tmp_90
    _pt_tmp_581 = _pt_tmp_582 + _pt_tmp_369
    del _pt_tmp_583
    _pt_tmp_577 = _pt_tmp_578 + _pt_tmp_581
    del _pt_tmp_369, _pt_tmp_582
    _pt_tmp_576 = 0.5 * _pt_tmp_577
    del _pt_tmp_578, _pt_tmp_581
    _pt_tmp_586 = (
        _pt_tmp_573 * _pt_data_13
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_573, _in1=_pt_data_13)["out"]
    )
    del _pt_tmp_577
    _pt_tmp_585 = _pt_tmp_93 * _pt_tmp_586
    _pt_tmp_584 = _pt_tmp_585 / 2
    del _pt_tmp_586
    _pt_tmp_575 = _pt_tmp_576 - _pt_tmp_584
    del _pt_tmp_585
    _pt_tmp_574 = (
        _pt_tmp_575 * _pt_data_13
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_575, _in1=_pt_data_13)["out"]
    )
    del _pt_tmp_576, _pt_tmp_584
    _pt_tmp_559 = _pt_tmp_560 + _pt_tmp_574
    del _pt_tmp_575
    _pt_tmp_592 = _pt_tmp_45 * _pt_tmp_102
    del _pt_tmp_560, _pt_tmp_574
    _pt_tmp_591 = _pt_tmp_592 + _pt_tmp_350
    del _pt_tmp_102, _pt_tmp_45
    _pt_tmp_594 = _pt_tmp_71 * _pt_tmp_113
    del _pt_tmp_350, _pt_tmp_592
    _pt_tmp_593 = _pt_tmp_594 + _pt_tmp_353
    del _pt_tmp_113, _pt_tmp_71
    _pt_tmp_590 = _pt_tmp_591 + _pt_tmp_593
    del _pt_tmp_353, _pt_tmp_594
    _pt_tmp_589 = 0.5 * _pt_tmp_590
    del _pt_tmp_591, _pt_tmp_593
    _pt_tmp_597 = (
        _pt_tmp_573 * _pt_data_14
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_573, _in1=_pt_data_14)["out"]
    )
    del _pt_tmp_590
    _pt_tmp_596 = _pt_tmp_93 * _pt_tmp_597
    del _pt_tmp_573
    _pt_tmp_595 = _pt_tmp_596 / 2
    del _pt_tmp_597, _pt_tmp_93
    _pt_tmp_588 = _pt_tmp_589 - _pt_tmp_595
    del _pt_tmp_596
    _pt_tmp_587 = (
        _pt_tmp_588 * _pt_data_14
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_3, _in0=_pt_tmp_588, _in1=_pt_data_14)["out"]
    )
    del _pt_tmp_589, _pt_tmp_595
    _pt_tmp_558 = _pt_tmp_559 + _pt_tmp_587
    del _pt_tmp_588
    _pt_tmp_557 = (
        _pt_tmp_558[_pt_tmp_139, _pt_tmp_140]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_5, in_0=_pt_tmp_558, in_1=_pt_tmp_139, in_2=_pt_tmp_140
        )["out"]
    )
    del _pt_tmp_559, _pt_tmp_587
    _pt_tmp_556 = (
        actx.np.where(_pt_tmp_31, _pt_tmp_557, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_6, _in0=_pt_tmp_31, _in1=_pt_tmp_557)["out"]
    )
    del _pt_tmp_139, _pt_tmp_140, _pt_tmp_558
    _pt_tmp_556 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_556)
    del _pt_tmp_31, _pt_tmp_557
    _pt_tmp_556 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_556)
    _pt_tmp_555 = 0 + _pt_tmp_556
    _pt_tmp_554 = 0 + _pt_tmp_555
    del _pt_tmp_556
    _pt_tmp_610 = _pt_tmp_178 * _pt_tmp_154
    del _pt_tmp_555
    _pt_tmp_609 = _pt_tmp_159 * _pt_tmp_610
    del _pt_tmp_154
    _pt_tmp_608 = _pt_tmp_609 + _pt_tmp_413
    del _pt_tmp_610
    _pt_tmp_613 = _pt_tmp_206 * _pt_tmp_180
    del _pt_tmp_609
    _pt_tmp_612 = _pt_tmp_159 * _pt_tmp_613
    del _pt_tmp_180
    _pt_tmp_611 = _pt_tmp_612 + _pt_tmp_417
    del _pt_tmp_613
    _pt_tmp_607 = _pt_tmp_608 + _pt_tmp_611
    del _pt_tmp_612
    _pt_tmp_606 = 0.5 * _pt_tmp_607
    del _pt_tmp_608, _pt_tmp_611
    _pt_tmp_617 = _pt_tmp_203 - _pt_tmp_176
    del _pt_tmp_607
    _pt_tmp_616 = (
        _pt_tmp_617 * _pt_data_21
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_617, _in1=_pt_data_21)["out"]
    )
    del _pt_tmp_176, _pt_tmp_203
    _pt_tmp_615 = _pt_tmp_209 * _pt_tmp_616
    _pt_tmp_614 = _pt_tmp_615 / 2
    del _pt_tmp_616
    _pt_tmp_605 = _pt_tmp_606 - _pt_tmp_614
    del _pt_tmp_615
    _pt_tmp_604 = (
        _pt_tmp_605 * _pt_data_21
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_605, _in1=_pt_data_21)["out"]
    )
    del _pt_tmp_606, _pt_tmp_614
    _pt_tmp_624 = _pt_tmp_178 * _pt_tmp_174
    del _pt_tmp_605
    _pt_tmp_623 = _pt_tmp_159 * _pt_tmp_624
    del _pt_tmp_174, _pt_tmp_178
    _pt_tmp_622 = _pt_tmp_623 + _pt_tmp_413
    del _pt_tmp_624
    _pt_tmp_627 = _pt_tmp_206 * _pt_tmp_201
    del _pt_tmp_413, _pt_tmp_623
    _pt_tmp_626 = _pt_tmp_159 * _pt_tmp_627
    del _pt_tmp_201, _pt_tmp_206
    _pt_tmp_625 = _pt_tmp_626 + _pt_tmp_417
    del _pt_tmp_627
    _pt_tmp_621 = _pt_tmp_622 + _pt_tmp_625
    del _pt_tmp_417, _pt_tmp_626
    _pt_tmp_620 = 0.5 * _pt_tmp_621
    del _pt_tmp_622, _pt_tmp_625
    _pt_tmp_630 = (
        _pt_tmp_617 * _pt_data_22
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_617, _in1=_pt_data_22)["out"]
    )
    del _pt_tmp_621
    _pt_tmp_629 = _pt_tmp_209 * _pt_tmp_630
    _pt_tmp_628 = _pt_tmp_629 / 2
    del _pt_tmp_630
    _pt_tmp_619 = _pt_tmp_620 - _pt_tmp_628
    del _pt_tmp_629
    _pt_tmp_618 = (
        _pt_tmp_619 * _pt_data_22
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_619, _in1=_pt_data_22)["out"]
    )
    del _pt_tmp_620, _pt_tmp_628
    _pt_tmp_603 = _pt_tmp_604 + _pt_tmp_618
    del _pt_tmp_619
    _pt_tmp_636 = _pt_tmp_159 * _pt_tmp_218
    del _pt_tmp_604, _pt_tmp_618
    _pt_tmp_635 = _pt_tmp_636 + _pt_tmp_398
    del _pt_tmp_218
    _pt_tmp_638 = _pt_tmp_159 * _pt_tmp_229
    del _pt_tmp_398, _pt_tmp_636
    _pt_tmp_637 = _pt_tmp_638 + _pt_tmp_401
    del _pt_tmp_159, _pt_tmp_229
    _pt_tmp_634 = _pt_tmp_635 + _pt_tmp_637
    del _pt_tmp_401, _pt_tmp_638
    _pt_tmp_633 = 0.5 * _pt_tmp_634
    del _pt_tmp_635, _pt_tmp_637
    _pt_tmp_641 = (
        _pt_tmp_617 * _pt_data_23
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_617, _in1=_pt_data_23)["out"]
    )
    del _pt_tmp_634
    _pt_tmp_640 = _pt_tmp_209 * _pt_tmp_641
    del _pt_tmp_617
    _pt_tmp_639 = _pt_tmp_640 / 2
    del _pt_tmp_209, _pt_tmp_641
    _pt_tmp_632 = _pt_tmp_633 - _pt_tmp_639
    del _pt_tmp_640
    _pt_tmp_631 = (
        _pt_tmp_632 * _pt_data_23
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_632, _in1=_pt_data_23)["out"]
    )
    del _pt_tmp_633, _pt_tmp_639
    _pt_tmp_602 = _pt_tmp_603 + _pt_tmp_631
    del _pt_tmp_632
    _pt_tmp_601 = (
        _pt_tmp_602[_pt_tmp_255, _pt_tmp_256]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_11, in_0=_pt_tmp_602, in_1=_pt_tmp_255, in_2=_pt_tmp_256
        )["out"]
    )
    del _pt_tmp_603, _pt_tmp_631
    _pt_tmp_600 = (
        actx.np.where(_pt_tmp_145, _pt_tmp_601, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_6, _in0=_pt_tmp_145, _in1=_pt_tmp_601)["out"]
    )
    del _pt_tmp_255, _pt_tmp_256, _pt_tmp_602
    _pt_tmp_600 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_600)
    del _pt_tmp_145, _pt_tmp_601
    _pt_tmp_600 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_600)
    _pt_tmp_599 = 0 + _pt_tmp_600
    _pt_tmp_598 = 0 + _pt_tmp_599
    del _pt_tmp_600
    _pt_tmp_553 = _pt_tmp_554 + _pt_tmp_598
    del _pt_tmp_599
    _pt_tmp_552 = actx.np.reshape(_pt_tmp_553, (4, 162000, 6))
    del _pt_tmp_554, _pt_tmp_598
    _pt_tmp_552 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_552)
    del _pt_tmp_553
    _pt_tmp_551 = actx.einsum(
        "ijk, jl, jlk -> li", _pt_data_3, _pt_tmp_24, _pt_tmp_552
    )
    _pt_tmp_551 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_551)
    del _pt_tmp_24, _pt_tmp_552
    _pt_tmp_539 = _pt_tmp_540 - _pt_tmp_551
    _pt_tmp_538 = actx.einsum("i, jk, ik -> ij", _pt_tmp_1, _pt_data_0, _pt_tmp_539)
    del _pt_tmp_540, _pt_tmp_551
    _pt_tmp_538 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_538)
    del _pt_tmp_1, _pt_tmp_539
    _pt_tmp = make_obj_array(
        [_pt_tmp_0, _pt_tmp_257, _pt_tmp_320, _pt_tmp_434, _pt_tmp_538]
    )
    return _pt_tmp
    del _pt_tmp_0, _pt_tmp_257, _pt_tmp_320, _pt_tmp_434, _pt_tmp_538


@dataclass(frozen=True)
class RHSInvoker:
    actx: ArrayContext

    @cached_property
    def npzfile(self):
        import os

        from immutabledict import immutabledict

        kw_to_ary = np.load(
            os.path.join(
                get_actx_dgfem_suite_path(), "suite/euler_3D_P2/literals.npz"
            )
        )
        return immutabledict(
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
            get_actx_dgfem_suite_path(), "suite/euler_3D_P2/ref_outputs.pkl"
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

        return immutabledict(
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
            "_actx_in_1_energy_0",
            "_actx_in_1_mass_0",
            "_actx_in_1_momentum_0_0",
            "_actx_in_1_momentum_1_0",
            "_actx_in_1_momentum_2_0",
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
