#!/usr/bin/env python3
"""
Analyze OST observing script extracted data.
"""

import pandas as pd
from pandas import Index as idx


HF_RCVRS = ["22GHz", "33GHz", "45GHz"]
LF_RCVRS = ["75MHz", "300MHz", "1.5GHz", "3GHz", "6GHz", "10GHz", "15GHz"]


def summarize_continuum(df):
    # number of WIDAR configs that are:
    #   continuum only
    #   containing a spectral line
    #   non-standard continuum
    pass


def summarize_bands(df):
    counts = df.rcvr.groupby(level=["label","lo_ix"]).first().value_counts()
    n_hf = counts.loc[HF_RCVRS].sum()
    n_lf = counts.loc[LF_RCVRS].sum()
    print(counts)
    print(f"-- K A Q :       {n_hf: 6d}")
    print(f"-- 4 L S C X U : {n_lf: 6d}")


def summarize_samplers(df):
    labels = df.index.get_level_values("label").unique()
    bit_df = df.is_8bit
    pure_8bit = 0
    pure_3bit = 0
    hybridbit = 0
    n_setups  = 0
    for label in labels:
        samp = bit_df.xs(label, level="label").groupby("lo_ix").mean().values
        pure_8bit += (samp == 1.0).sum()
        pure_3bit += (samp == 0.0).sum()
        hybridbit += ((samp > 0.0) & (samp < 1.0)).sum()
        n_setups  += samp.size
    assert int(pure_8bit + pure_3bit + hybridbit) == n_setups
    print(f"-- Total configurations: {n_setups: 6d}")
    print(f"-- Pure 8-bit setups:    {pure_8bit: 6d}")
    print(f"-- Pure 3-bit setups:    {pure_3bit: 6d}")
    print(f"-- Hybrid setups:        {hybridbit: 6d}")
    print( "NOTE: X Band pointing inflates 8-bit count.")


def summarize(df):
    summarize_continuum(df)
    summarize_samplers(df)


