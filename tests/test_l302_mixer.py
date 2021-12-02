#!/usr/bin/env python3

import numpy as np

from ost_parser import OST_ROOT
from ost_parser.core import Execution
from ost_parser.analysis.l302_mixer import (SubBandFreqs, MixerFlag,
        get_used_flag)


TEST_PATH = OST_ROOT / "2021/01/97920B-290A"


def test_mixer_8bit():
    ex = Execution(TEST_PATH)
    b_max = ex.efile.max_baseline
    conf_8bit = ex.configs_by_scan[2]
    for baseband in conf_8bit.vci:
        assert baseband.is_8bit
        for subband in baseband:
            sf = SubBandFreqs(subband, conf_8bit, b_max)
            assert sf
            assert np.isclose(sf.tau, 1.0)
            assert sf.opt_flag.name == "OKAY"
            assert sf.used_flag.name == "OKAY"


def test_mixer_3bit():
    ex = Execution(TEST_PATH)
    b_max = ex.efile.max_baseline
    conf_3bit = ex.configs_by_scan[4]
    for baseband in conf_3bit.vci:
        assert baseband.is_3bit
        for subband in baseband:
            sf = SubBandFreqs(subband, conf_3bit, b_max)
            assert np.isclose(sf.tau, 2.0)
            assert sf.opt_flag.name == "OKAY"
            assert sf.used_flag.name == "OKAY"


def test_failing():
    ex1 = Execution.from_str("2013/01/02112B-154A")
    ex2 = Execution.from_str("2021/11/27420B-310A")
    assert get_used_flag(ex1) == MixerFlag.FAIL
    assert get_used_flag(ex2) == MixerFlag.FAIL


