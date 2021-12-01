#!/usr/bin/env python3

from astropy import units as u

from ost_parser import OST_ROOT
from ost_parser.core import Execution
from ost_parser.analysis import l302_mixer


TEST_PATH = OST_ROOT / "2021/01/97920B-290A"


def test_mixer_8bit():
    ex = Execution(TEST_PATH)
    b_max = ex.efile.max_baseline
    conf_8bit = ex.configs_by_scan[2]
    for baseband in conf_8bit.vci:
        assert baseband.is_8bit
        for subband in baseband:
            sf = l302_mixer.SubBandFreqs(subband, conf_8bit, b_max)
            assert sf
            assert sf.tau.value == 1.0
            assert sf.opt_flag.name == "OKAY"
            assert sf.used_flag.name == "OKAY"


def test_mixer_3bit():
    ex = Execution(TEST_PATH)
    b_max = ex.efile.max_baseline
    conf_3bit = ex.configs_by_scan[4]
    for baseband in conf_3bit.vci:
        assert baseband.is_3bit
        for subband in baseband:
            sf = l302_mixer.SubBandFreqs(subband, conf_3bit, b_max)
            assert sf.tau.value == 2.0
            assert sf.opt_flag.name == "OKAY"
            assert sf.used_flag.name == "OKAY"


