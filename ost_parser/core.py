#!/usr/bin/env python3

import re
import warnings
from pathlib import Path
from multiprocessing import Pool
from dataclasses import dataclass

import cchardet
import pandas as pd
from pandas import IndexSlice as idx

from ost_parser import OST_ROOT
from ost_parser.vci import (VCI, BaseBand, SubBand)


KNOWN_INVALID = [
        "53714A-242A",  # 2014/06/
]


def check_all_encodings():
    paths = sorted(OST_ROOT.glob("????/??/*-*/*.evla"))
    encodings = {}
    for path in paths:
        with path.open("rb") as f:
            text_bytes = f.read()
        encoding = cchardet.detect(text_bytes)["encoding"]
        encodings[path.name] = encoding
    return encodings


class ParseError(RuntimeError):
    pass


class EvlaFile:
    config_max_baselines = {
            "A": 36400.,  # m
            "B": 11100.,
            "C":  3400.,
            "D":  1030.,
    }
    p_oldstyle = re.compile(R"loif\d+ = LoIfSetup")
    p_array_conf = re.compile(R"#   Array Configurations: (.+?)\n")
    p_project = re.compile(R"# EVLA PROJECT (.+?), DB ID (\d+?)\n")
    p_mjd = re.compile(R"#   Assumed Script Start: .+?MJD ([\.\d]+?)\)\n")

    def __init__(self, path):
        with path.open("rb") as f:
            text_bytes = f.read()
        encoding = cchardet.detect(text_bytes)["encoding"]
        text = text_bytes.decode(encoding)
        self.path = path
        self.text = text
        self.oldstyle = bool(self.p_oldstyle.search(text))
        self.has_scan_loop = "for iter1 in range" in self.text
        self.max_config = self.parse_max_config(text)
        self.max_baseline = self.config_max_baselines[self.max_config]
        # Extract project information
        try:
            self.code, self.dbid = self.p_project.search(text).groups()
        except AttributeError:
            raise ParseError("Could not parse project code information.")
        # Extract start MJD
        try:
            self.mjd = float(self.p_mjd.search(text).groups()[0])
        except AttributeError:
            raise ParseError("Could not parse assumed MJD start.")

    @property
    def newstyle(self):
        return not self.oldstyle

    def parse_max_config(self, text):
        """
        Extract the allowed array configurations from the header comments
        and determine the configuration with the longest baseline. This is
        done by selecting the lowest alphabetic value among the array
        configuration codes.

        Move configs are converted to the maximum configuration among the two.
        Selecting the digit code with `min` still works because "N" is a
        greater character value than ABCD, e.g.: the minimum alphabetic value
        of "C=>CNB" is "B" (after first removing the non-alpha characters
        "=>"). "Any" is converted "A" for similar reasons.
        """
        try:
            group = self.p_array_conf.search(text).groups()[0]
        except AttributeError:
            raise ParseError("Could not parse array configuration codes.")
        configs = [
                min(s.strip().replace("=>", ""))
                for s in group.split(",")
        ]
        return min(configs)


class LoIfSetup:
    p_old = re.compile(
            R"loif(\d+) = "
            R'LoIfSetup\((\S+?),"(\S+?)",(\S+?),(\S+?),(\S+?),(\S+?),(\S+?)\)'
            )
    p_new = re.compile(
            R"loifs\['loif(\d+)'\] = "
            R'LoIfSetup\((\S+?),"(\S+?)",(\S+?),(\S+?),(\S+?),(\S+?),(\S+?)\)'
    )
    p_offset = re.compile(R"tuningOffsetMHz\s?=\s?([-\d\.e]+)")
    offset_str = "tuningOffsetMHz"

    def __init__(self, group, tuning_offset=0.0):
        """
        Baseband center frequencies are given for `mode` True.
        Signed sum of LO frequencies (SSLO) are given for `mode` False.
        """
        assert isinstance(tuning_offset, (float, int))
        self.tuning_offset = tuning_offset
        self.ix     = int(group[0])
        self.loif   = f"loif{group[0]}"
        self.mode   = group[1] == "True"
        self.rcvr   = group[2]
        self.bb_ac1 = self.parse_frequency(group[3])
        self.bb_ac2 = self.parse_frequency(group[4])
        self.bb_bd1 = self.parse_frequency(group[5])
        self.bb_bd2 = self.parse_frequency(group[6])
        self.flag   = int(group[7])
        self.is_8bit = self.bb_ac2 + self.bb_bd2 == 0.0
        self.is_3bit = not self.is_8bit
        self.uses_offset = any([self.offset_str in g for g in group[3:7]])
        self._bb_map = {
                # 8-bit
                "A0/C0": self.bb_ac1,
                "B0/D0": self.bb_bd1,
                # 3-bit
                "A1/C1": self.bb_ac1,
                "A2/C2": self.bb_ac2,
                "B1/D1": self.bb_bd1,
                "B2/D2": self.bb_bd2,
        }

    @classmethod
    def from_evla_text(cls, efile):
        p = cls.p_old if efile.oldstyle else cls.p_new
        m_offset = cls.p_offset.search(efile.text)
        offset = float(m_offset.groups()[0]) if m_offset else 0.0
        groups = p.findall(efile.text)
        if not groups:
            raise ParseError(f"Could not find LOIF center frequencies")
        objs = [cls(g, tuning_offset=offset) for g in groups]
        return {o.loif: o for o in objs}

    def parse_frequency(self, s):
        if self.offset_str in s:
            v = eval(
                    s.replace(self.offset_str, str(self.tuning_offset))
            )
        else:
            v = float(s)
        return v * 1e6  # from MHz to Hz

    def freq_from_subband(self, subband):
        """**unit**: Hz"""
        assert self.mode  # invalid for SSLOs
        return self._bb_map[subband.baseband.name]

    def sky_freq_from_subband(self, subband):
        """**unit**: Hz"""
        bb_center_freq = self.freq_from_subband(subband)
        bb_bandwidth = subband.baseband.bw
        bb_start_freq = bb_center_freq - bb_bandwidth / 2
        sb_offset_from_bb_start = subband.cf
        return bb_start_freq + sb_offset_from_bb_start


class LoIfOffset:
    p_old = re.compile(
            R"loif(\d+)"
            R"\.setWidarOffsetFreq\((\S+?),(\S+?),(\S+?),(\S+?)\)"
    )
    p_new = re.compile(
            R"loifs\['loif(\d+)'\]"
            R"\.setWidarOffsetFreq\((\S+?),(\S+?),(\S+?),(\S+?)\)"
    )

    def __init__(self, group):
        self.ix   = int(group[0])
        self.loif = f"loif{group[0]}"
        self.df_ac1 = float(group[1]) * 1e6  # from MHz to Hz
        self.df_bd1 = float(group[2]) * 1e6
        self.df_ac2 = float(group[3]) * 1e6
        self.df_bd2 = float(group[4]) * 1e6
        self._bb_map = {
                # 8-bit
                "A0/C0": self.df_ac1,
                "B0/D0": self.df_bd1,
                # 3-bit
                "A1/C1": self.df_ac1,
                "A2/C2": self.df_ac2,
                "B1/D1": self.df_bd1,
                "B2/D2": self.df_bd2,
        }

    @classmethod
    def from_evla_text(cls, efile):
        p = cls.p_old if efile.oldstyle else cls.p_new
        groups = p.findall(efile.text)
        if not groups:
            raise ParseError(f"Could not find LOIF offset frequencies")
        objs = [cls(g) for g in groups]
        return {o.loif: o for o in objs}

    def freq_from_subband(self, subband):
        """**unit**: Hz"""
        return self._bb_map[subband.baseband.name]


class EvlaScan:
    p_loif_old = re.compile(R"\s*subarray\.setLoIfSetup\(loif(\d+)\)")
    p_loif_new = re.compile(R"\s*loifName = 'loif(\d+)'")
    p_scan = re.compile(R"\s*# Scan num\. (\d+), '(.*?)', DB ID (\d+)")
    p_time = re.compile(R"\s*# Approx sidereal time: total = (\d+)s, slew = (\d+)s, on source = (\d+)s")
    p_intent = re.compile(R"""\s*intents\.addIntent\('ScanIntent="(.+?)"'\)""")

    def __init__(self, group):
        self.ix     = int(group[0])
        self.field  = group[1]
        self.db_id  = int(group[2])
        self.loif   = f"loif{group[3]}"
        self.t_total = float(group[4][0])
        self.t_slew  = float(group[4][1])
        self.t_onsrc = float(group[4][2])
        self.intents = group[5].split()

    @classmethod
    def from_evla_text(cls, efile):
        p_loif = cls.p_loif_old if efile.oldstyle else cls.p_loif_new
        lines = efile.text.split("\n")
        loif = "NULL"
        intents = "NULL"
        scans = []
        for i, line in enumerate(lines):
            # Identify a scan block from the comment "# Scan num. 1 ..."
            # If no match is found, then continue on to the next line in the
            # file.
            try:
                ix, field, db_id = cls.p_scan.match(line).groups()
            except AttributeError:
                continue
            # The scan comment is always followed by the "# Approx sidereal..."
            # line with time information.
            try:
                times = cls.p_time.match(lines[i+1]).groups()
            except AttributeError:
                raise ParseError("Could not parse times.")
            # If the intents change, then the next line will set the "ScanIntent"
            # attribute to the intents. Note: "addIntent" is really more of a
            # setter method to the underlying Java class, not an append operation.
            try:
                intents = cls.p_intent.match(lines[i+2]).groups()[0]
            except AttributeError:
                pass
            if intents == "NULL":
                raise ParseError("Invalid addIntent convention.")
            # If a new scan block appears (with a blank line) before the
            # LOIF setting is changed, then the LOIF settings are re-used
            # from the last scan.
            for peek in lines[i:]:
                if peek.strip() == "":  # blank line after block
                    break
                m_loif = p_loif.match(peek)
                if m_loif:
                    loif = m_loif[1]
                    break
            else:
                raise ParseError("Invalid loifName/loifCounts convention.")
            if loif == "NULL":
                raise ParseError("Invalid loifName/loifCounts convention.")
            items = (ix, field, db_id, loif, times, intents)
            scans.append(cls(items))
        return scans


@dataclass
class WidarConfig:
    scan : EvlaScan
    loif_setup : LoIfSetup
    loif_offset : LoIfOffset
    vci : VCI


class Execution:
    root = Path(OST_ROOT)

    def __init__(self, path):
        if not path.exists():
            raise FileNotFoundError
        self.path = path
        self.project = path.name
        self.month = path.parent.name
        self.year = path.parent.parent.name
        self.label = f"{self.year}/{self.month}/{self.project}"
        # Read in files
        evla_paths = sorted(self.path.glob("*.evla"))
        vci_paths = sorted(self.path.glob("*.vci"))
        if not evla_paths:
            raise ParseError(f"No `.evla` files found: {self.label}")
        if not vci_paths:
            raise ParseError(f"No `.vci` files found: {self.label}")
        if len(evla_paths) > 1:
            raise ParseError(f"More than one `.evla` file found: {self.label}")
        self.evla_path = evla_paths[0]
        self.vci_paths = vci_paths
        self.vci = [VCI(str(p)) for p in self.vci_paths]
        self.n_vci = len(self.vci)
        self.vci_by_scanid = {v.scanId: v for v in self.vci}
        # Parse WIDAR configs
        efile        = EvlaFile(self.evla_path)
        loif_setups  = LoIfSetup.from_evla_text(efile)
        loif_offsets = LoIfOffset.from_evla_text(efile)
        scans        = EvlaScan.from_evla_text(efile)
        assert len(loif_setups) == len(loif_offsets)
        vci_by_scan  = self.get_vci_for_scans(scans, self.vci)
        self.configs_by_scan = {
                scan.ix: WidarConfig(
                            scan,
                            loif_setups[scan.loif],
                            loif_offsets[scan.loif],
                            vci_by_scan[scan.ix],
                    )
                for scan in scans
        }
        self.configs_by_loif = {
                loif: [
                        w for w in self.configs_by_scan.values()
                        if w.loif_setup.loif == loif
                ]
                for loif in loif_setups
        }
        self.efile = efile
        self.loif_setups = loif_setups
        self.loif_offsets = loif_offsets
        self.scans = scans
        self.vci_by_scan = vci_by_scan

    @classmethod
    def from_str(cls, *args, **kwargs):
        if len(args) == 1:
            path = cls.root / args[0]
        elif len(args) == 3:
            year, month, project = args
            path = cls.root / year / month / project
        else:
            raise ValueError(f"Invalid number of arguments: {len(args)}")
        return cls(path, **kwargs)

    @staticmethod
    def get_vci_for_scans(scans, vci):
        scan_nums = [s.ix for s in scans]
        vci_nums  = [v.scanNum for v in vci]
        vci_by_scan = {v.scanNum: v for v in vci}
        def get_vci(s_ix):
            assert s_ix >= 0
            v_ix = max([i for i in vci_nums if i <= s_ix])
            return vci_by_scan[v_ix]
        return {i: get_vci(i) for i in scan_nums}

    def iter_loifs(self):
        for loif_name in self.loif_setups.keys():
            setup  = self.loif_setups[loif_name]
            offset = self.loif_offsets[loif_name]
            yield loif_name, setup, offset

    def iter_valid_unique_subbands(self):
        configs = [ws[0] for ws in self.configs_by_loif.values()]
        for config in configs:
            for baseband in config.vci:
                for subband in baseband:
                    if subband.is_valid:
                        yield config, baseband, subband

    @property
    def all_invalid(self):
        try:
            next(self.iter_valid_unique_subbands())
            return False
        except StopIteration:
            return True

    def to_pandas(self):
        ex_items = list(self.iter_valid_unique_subbands())
        if not ex_items:
            raise ParseError("No valid subbands.")
        cf, bb, sb = list(map(list, zip(*ex_items)))  # transpose
        col_data = {
                "label":          [self.label for _ in sb],
                "year":           [self.year for _ in sb],
                "month":          [self.month for _ in sb],
                "code":           [self.efile.code for _ in sb],
                "dbid":           [self.efile.dbid for _ in sb],
                "mjd":            [self.efile.mjd for _ in sb],
                "lo_ix":          [c.loif_setup.ix for c in cf],
                "bb_ix":          [b.name for b in bb],
                "sb_ix":          [s.swIndex for s in sb],
                "max_config":     [self.efile.max_config for _ in sb],
                "max_baseline":   [self.efile.max_baseline for _ in sb],
                "lo_mode":        [c.loif_setup.mode for c in cf],
                "lo_flag":        [c.loif_setup.flag for c in cf],
                "rcvr":           [c.loif_setup.rcvr for c in cf],
                "bb_bw":          [b.bw for b in bb],
                "in_quant":       [b.inQuant for b in bb],
                "is_8bit":        [int(b.is_8bit) for b in bb],
                "npol":           [s.npol for s in sb],
                "nchan":          [s.nchan for s in sb],
                "recirc":         [s.recirc for s in sb],
                "min_integ_time": [s.minIntegTime for s in sb],
                "int_time":       [s.inttime for s in sb],
                "sb_bw":          [s.bw for s in sb],
                "sb_cf":          [s.cf for s in sb],
                "nblb":           [s.nblb for s in sb],
                "f_samp":         [s.freqSamp for s in sb],
                "f_opt":          [s.freqOpt for s in sb],
                "f_sky": [
                        c.loif_setup.sky_freq_from_subband(s)
                        for c, s in zip(cf, sb)
                ],
                "f_shift": [
                        c.loif_offset.freq_from_subband(s)
                        for c, s in zip(cf, sb)
                ],
                "bb_cf": [
                        c.loif_setup.freq_from_subband(s)
                        for c, s in zip(cf, sb)
                ],
        }
        df = pd.DataFrame(col_data)
        return df


def get_program_paths(exclude_tests=True):
    paths = sorted(OST_ROOT.glob("????/??/*"))
    if exclude_tests:
        paths = [p for p in paths if "-" in p.name]
    return paths


def parse_exec_if_valid(path, console_out=False):
    if path.name in KNOWN_INVALID:
        warnings.warn(f"Parse error: known invalid: {path}")
        return
    try:
        if console_out:
            print(f"-- {path}")
        ex = Execution(path)
        return ex
    except ParseError as e:
        warnings.warn(f"{e}")
        return


def read_all_executions(nproc=1):
    paths = get_program_paths()
    assert nproc > 0
    if nproc > 1:
        with Pool(processes=nproc) as pool:
            execs = pool.map(parse_exec_if_valid, paths)
    else:
        execs = [parse_exec_if_valid(p) for p in paths]
    return [ex for ex in execs if ex is not None]


def execs_to_df(execs, nproc=1):
    assert nproc > 0
    if nproc > 1:
        with Pool(processes=nproc) as pool:
            all_dfs = pool.map(Execution.to_pandas, execs)
    else:
        all_dfs = [ex.to_pandas() for ex in execs]
    merged = pd.concat(all_dfs)
    merged.set_index(["label", "lo_ix", "bb_ix", "sb_ix"], inplace=True)
    return merged


