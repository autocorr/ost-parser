#!/usr/bin/env python3
r"""
f_mf  = 460 Hz * b_max / 20 km * f_sky / 50 GHz
f_min = max(f_mf, 30/tau)
f_max = (f_bw / 4e4) if recirc else (f_bw / 3e3)
f_opt = 1.25 / (4 pi) sqrt(f_samp / tau)

Mixer flags:
 I.   f_min <= f_opt <= f_max
 II.  f_opt < f_min and f_min < f_max
 III. f_opt > f_max and f_min < f_max
 IV.  f_min > f_max

f_min requires:
  f_maxfringe
    array configuration max baseline <- efile
    observing frequency              <- loif_setup
  integration time                   <- subband

f_max requires:
  f_bw                <- subband
  using_recirculation <- subband
"""

from enum import Enum
from dataclasses import dataclass
from multiprocessing import Pool

from astropy import units as u

from ost_parser.vci import SubBand
from ost_parser.core import WidarConfig



class MixerFlag(Enum):
    OKAY  = 1
    BELOW = 2
    ABOVE = 3
    FAIL  = 4


BAD_MIXER_FLAGS = (MixerFlag.BELOW, MixerFlag.ABOVE, MixerFlag.FAIL)


def mixer_flag_from_freqs(f_opt, f_min, f_max):
    if f_min > f_max:
        return MixerFlag.FAIL
    elif f_opt < f_min < f_max:
        return MixerFlag.BELOW
    elif f_min < f_max < f_opt:
        return MixerFlag.ABOVE
    elif f_min <= f_opt <= f_max:
        return MixerFlag.OKAY
    else:
        raise ValueError


@dataclass
class SubBandFreqs:
    subband : SubBand
    config : WidarConfig
    b_max : int

    @property
    def tau(self):
        return self.subband.inttime

    @property
    def bw(self):
        return self.subband.bw

    @property
    def f_opt(self):
        return self.subband.freqOpt

    @property
    def f_sky(self):
        return self.config.loif_setup.sky_freq_from_subband(self.subband)

    @property
    def f_used(self):
        return self.config.loif_offset.freq_from_subband(self.subband)

    @property
    def f_maxfringe(self):
        d_scale = self.b_max.to("km").value / 20
        f_scale = self.f_sky.to("GHz").value / 50
        return 460 * u.Hz * d_scale * f_scale

    @property
    def f_min(self):
        return max(self.f_maxfringe, 30 / self.tau)

    @property
    def f_max(self):
        scale = 4e4 if self.subband.recirc > 1 else 3e3
        return self.bw / scale

    @property
    def opt_flag(self):
        return self.flag_from_freq(self.f_opt)

    @property
    def used_flag(self):
        return self.flag_from_freq(self.f_used)

    def flag_from_freq(self, freq):
        return mixer_flag_from_freqs(freq, self.f_min, self.f_max)


def get_used_flag(ex):
    b_max = ex.efile.max_baseline
    for configs in ex.configs_by_loif.values():
        config = configs[0]
        for baseband in config.vci:
            for subband in baseband:
                sf = SubBandFreqs(subband, config, b_max)
                flag = sf.used_flag
                if flag != MixerFlag.OKAY:
                    return flag
    return MixerFlag.OKAY


def fshifts_used_are_invalid(ex):
    """
    Determine if the frequency offset used is invalid for any sub-band.
    Only the first scan of an LOIF/VCI configuration requires testing.
    """
    flag = get_used_flag(ex)
    if flag != MixerFlag.OKAY:
        return ex
    else:
        return None


def validate_used_fshifts(execs, nproc=1):
    if nproc > 1:
        with Pool(processes=nproc) as pool:
            bad_execs = pool.map(fshifts_used_are_invalid, execs)
            return [ex for ex in bad_execs if ex is not None]
    else:
        return [ex for ex in execs if fshifts_used_are_invalid(ex)]


# TODO
# determine whether any sub-bands require the mixer
#   - do operations are done per-baseband since that's what f-shift is set over
#   - given all passing, use largest f_opt and determine whether
#     it can safely satisfy all sub-bands, if not, flag all subbands
#     where f_0 is greater than the f_max of the subband.


