#!/usr/bin/env python3

import io
import pstats
import cProfile

from ost_parser import OST_ROOT
from ost_parser.core import Execution


TEST_PATH = OST_ROOT / "2015/04/86614A-420B"


def profile_execution():
    profiler = cProfile.Profile()
    profiler.enable()
    Execution(TEST_PATH)
    profiler.disable()
    stream = io.StringIO()
    sortby = pstats.SortKey.CUMULATIVE
    stats = pstats.Stats(profiler, stream=stream).sort_stats(sortby)
    stats.print_stats()
    print(stream.getvalue())


