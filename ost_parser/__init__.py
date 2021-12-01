#!/usr/bin/env python3

from pathlib import Path


_shm = Path("/dev/shm/ost")
if _shm.exists():
    OST_ROOT = _shm
else:
    OST_ROOT = Path("/home/mchost/evla/scripts/ost")


