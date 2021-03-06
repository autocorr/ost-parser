#!/usr/bin/env python3

import pytest
import numpy as np

from . import (TEST_PATH, TEST_EVLA)
from ost_parser import OST_ROOT
from ost_parser.core import (
        ParseError, Execution, EvlaFile, LoIfSetup, LoIfOffset, EvlaScan,
        get_program_paths, parse_exec_if_valid,
)


@pytest.fixture
def f_ex():
    return Execution(TEST_PATH)


@pytest.fixture
def f_efile():
    return EvlaFile(TEST_EVLA)


class TestEvlaFile:
    def test_read(self, f_efile):
        assert LoIfSetup.from_evla_text(f_efile)
        assert LoIfOffset.from_evla_text(f_efile)
        assert EvlaScan.from_evla_text(f_efile)


class TestExecution:
    def test_read(self):
        assert Execution(TEST_PATH)
        assert Execution.from_str("2021/01/97920B-310A")
        assert Execution.from_str("2021", "01", "97920B-310A")

    def test_pandas(self, f_ex):
        assert not f_ex.to_pandas().empty

    def test_sampler(self, f_ex):
        bb = f_ex.configs_by_scan[3].vci[0]
        assert bb.is_8bit
        assert not bb.is_3bit

    def test_iter_loifs(self, f_ex):
        names = ["loif01", "loif02", "loif03"]
        assert names == list(f_ex.loif_setups.keys())
        assert names == list(f_ex.loif_offsets.keys())
        assert names == list(n for n, *_ in f_ex.iter_loifs())

    def test_first(self):
        ex = Execution.from_str("2012", "01", "68312A-304A")
        assert ex
        assert len(ex.scans) == 110
        assert len(ex.loif_setups) == 1
        assert ex.efile.max_config == "C"

    def test_non_utf8_encoding(self):
        assert Execution.from_str("2012", "02", "69011B-061A")

    def test_oldstyle_evla_format(self):
        ex = Execution.from_str("2018", "04", "94018A-416A")
        assert ex
        efile = EvlaFile(ex.evla_path)
        assert efile.oldstyle

    def test_newstyle_evla_format(self):
        ex = Execution.from_str("2021", "01", "97920B-310A")
        assert ex
        assert len(ex.loif_setups) == len(ex.loif_offsets)
        efile = EvlaFile(ex.evla_path)
        assert efile.newstyle

    def test_scan_loop(self):
        ex = Execution.from_str("2018", "10", "14117A-233A")
        assert ex
        assert len(ex.scans) == 20
        assert len(ex.loif_setups) == 2
        efile = EvlaFile(ex.evla_path)
        assert efile.newstyle
        assert efile.has_scan_loop

    def test_vci_reuse1(self):
        ex = Execution.from_str("2021", "01", "97920B-310A")
        scan_ixs = range(1, 12)
        vci_ixs  = [1, 2, 3, 3, 5, 5, 7, 7, 9, 10, 11]
        for s_ix, v_ix in zip(scan_ixs, vci_ixs):
            w = ex.configs_by_scan[s_ix]
            assert w.scan.ix == s_ix
            assert w.vci.scanNum == v_ix

    def test_vci_reuse2(self):
        ex = Execution.from_str("2019", "12", "55619B-209A")
        assert ex.efile.max_config == "D"
        scan_ixs = range(1, 17)
        vci_ixs  = [1, 2, 3, 3, 3, 3, 3, 3, 9, 9, 11, 11, 11, 14, 15, 15]
        for s_ix, v_ix in zip(scan_ixs, vci_ixs):
            w = ex.configs_by_scan[s_ix]
            assert w.scan.ix == s_ix
            assert w.vci.scanNum == v_ix

    def test_array_multiconfig(self):
        ex = Execution.from_str("2012", "05", "77312A-017A")
        # allowed configurations are: CNB, C=>CNB, CNB=>B
        assert ex.efile.max_config == "B"

    def test_tuning_offset_75MHz(self):
        ex = Execution.from_str("2012", "11", "98012B-394A")
        setup = ex.configs_by_scan[1].loif_setup
        assert np.isclose(setup.bb_ac1, 236.0e6)
        assert np.isclose(setup.bb_ac2,   0.0e6)
        assert np.isclose(setup.bb_bd1, 150.0e6)
        assert np.isclose(setup.bb_bd2,   0.0e6)

    def test_all_subbands_invalid(self):
        ex = Execution.from_str("2014", "06", "53714A-242A")
        assert ex.all_invalid


def test_get_program_paths():
    n_tests = get_program_paths(exclude_tests=True)
    y_tests = get_program_paths(exclude_tests=False)
    assert len(n_tests) > 0
    assert len(y_tests) > 0
    assert len(n_tests) < len(y_tests)


def test_parse_exec_if_valid():
    assert parse_exec_if_valid(TEST_PATH)


