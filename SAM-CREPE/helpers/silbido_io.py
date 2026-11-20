"""Read silbido contour files (.bin / .ann): the format SAM-Whistle writes its detections in and the DCLDE
annotations use. Copied from scripts/common/silbido_io.py, which was checked against SAM-Whistle's own writer
(write_binary.py) and reader (anno.py).

Writing is left to SAM-Whistle's own writeTimeFrequencyBinary (finalise.py).
"""
import struct
from pathlib import Path

import numpy as np

def read_silbido(path):
    """Whistles in a silbido .bin/.ann file: list of arrays with columns (time s, freq Hz).

    Handles files with a 'silbido!' header (version 4, time + frequency only) and the
    older header-less format.
    """
    b = Path(path).read_bytes()
    whistles, cur, version = [], 0, 0
    if b.startswith(b"silbido!"):
        version, mask = struct.unpack(">hh", b[8:12])
        if mask != 3:
            raise ValueError(f"{path}: extra per-point fields (bitmask {mask}) not supported")
        cur = struct.unpack(">i", b[14:18])[0]
    while cur < len(b):
        if version > 2:
            cur += 8  # graph id
        n = struct.unpack(">i", b[cur:cur + 4])[0]
        values = struct.unpack(">" + "dd" * n, b[cur + 4:cur + 4 + 16 * n])
        whistles.append(np.array(values).reshape(-1, 2))
        cur += 4 + 16 * n
    return whistles
