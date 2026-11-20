"""Where SAM-CREPE writes, and how fragments are saved and loaded.

Safety: everything is written to SAM-CREPE/outputs/<run name>/. A run refuses to start if its folder already
exists, unless asked to replace it, and replacing deletes only that one folder. Nothing outside
SAM-CREPE/outputs/ is ever written or deleted.

Uses only numpy, so stage B (project environment) can load what stage A (SAM-Whistle environment) saved.
"""
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"

def output_dir(name, replace=False):
    """Create SAM-CREPE/outputs/<name>/ for a new run and return it."""
    out = (OUTPUTS / name).resolve()
    if out.parent != OUTPUTS.resolve() or not name or name in (".", ".."):
        raise ValueError(f"run name {name!r} must be a single folder name inside {OUTPUTS}")
    if out.exists():
        if not replace:
            sys.exit(f"{out} already exists. Choose another --name, or pass --replace to delete and redo that run.")
        shutil.rmtree(out) # only this run's own folder, checked above to be inside OUTPUTS
    out.mkdir(parents=True)
    return out

@dataclass
class Fragment:
    """One piece of a whistle contour from the graph search. Times are seconds on the recording's clock (frame centres)."""
    points: np.ndarray        # [n x 2] time s, freq Hz
    graph_id: int             # which connected graph it came from: pieces of the same graph met at a crossing
    start_junction: bool      # True if it starts at a node where the graph branched (cut at a crossing)
    end_junction: bool        # True if it ends at one
    start_node: tuple         # (time s, freq Hz) of the graph node it starts at
    end_node: tuple           # (time s, freq Hz) of the graph node it ends at

def save_fragments(path, fragments, offset_s):
    """One .npz per selection. offset_s is what was added to SAM-Whistle's times to put them on the recording's
    clock (where the audio file starts in the recording, plus half a frame), kept so that results can be written
    back in SAM-Whistle's own time base (finalise.py)."""
    lengths = [len(f.points) for f in fragments]
    np.savez_compressed(
        path,
        points=np.vstack([f.points for f in fragments]) if fragments else np.zeros((0, 2)),
        offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(int),
        graph_id=np.array([f.graph_id for f in fragments], int),
        start_junction=np.array([f.start_junction for f in fragments], bool),
        end_junction=np.array([f.end_junction for f in fragments], bool),
        start_node=np.array([f.start_node for f in fragments]).reshape(-1, 2),
        end_node=np.array([f.end_node for f in fragments]).reshape(-1, 2),
        offset_s=offset_s)

def load_fragments(path):
    """(fragments, offset_s) from a file written by save_fragments"""
    d = np.load(path)
    o = d["offsets"]
    fragments = [Fragment(d["points"][o[i]:o[i + 1]], int(d["graph_id"][i]), bool(d["start_junction"][i]),
                          bool(d["end_junction"][i]), tuple(d["start_node"][i]), tuple(d["end_node"][i]))
                 for i in range(len(o) - 1)]
    return fragments, float(d["offset_s"])
