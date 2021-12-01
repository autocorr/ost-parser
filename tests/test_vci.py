#!/usr/bin/env python3

import pytest

from . import TEST_VCI
from ost_parser import vci


@pytest.fixture
def f_vci():
    return vci.VCI(str(TEST_VCI))


@pytest.fixture
def f_bb(f_vci):
    return f_vci[0]


@pytest.fixture
def f_sb(f_bb):
    return f_bb[0]


def test_vci(f_vci):
    assert iter(f_vci)
    assert f_vci.vciRequest is not None
    assert f_vci.summary()


def test_baseband(f_bb):
    assert iter(f_bb)
    assert f_bb.is_8bit
    assert not f_bb.is_3bit
    assert f_bb.name
    assert f_bb.bb


def test_subband(f_sb):
    assert f_sb.npol
    assert f_sb.nchan
    assert f_sb.recirc
    assert f_sb.minIntegTime is not None
    assert f_sb.integFac
    assert f_sb.inttime is not None
    assert f_sb.bw is not None
    assert f_sb.cf is not None
    assert f_sb.swIndex
    assert f_sb.sbid
    assert f_sb.baseband
    assert f_sb.baseband.is_8bit
    assert not f_sb.baseband.is_3bit
    assert f_sb.blbs
    assert f_sb.blbs
    assert f_sb.nblb


