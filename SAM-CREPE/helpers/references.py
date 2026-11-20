"""The analysts' traces, for the figures (step 4) only.

The method (stage A and stage B: get_fragments.py, run.py and the steps) never imports this module, so it cannot
see the ground truth. Only figures.py reads it, to draw the traces next to the detections.

THE REFERENCES FOLDER: one subfolder per audio file, named like its inputs folder, holding
    analyst_<name>.csv   one analyst's trace: time_s, freq_hz (time from the start of the analyst's own clip window)
    reference.json       analysts: the names; offset_s: seconds to add to time_s to put the trace on the recording's clock

An audio file with no references folder simply has no traces drawn.
"""
import csv
import json

import numpy as np


def load_references(stem, references):
    """{analyst name: (t, f)} on the recording's clock, or {} if there are no traces for this audio file."""
    folder = references / stem
    if not (folder / "reference.json").exists():
        return {}
    info = json.loads((folder / "reference.json").read_text())
    traces = {}
    for who in info["analysts"]:
        with open(folder / f"analyst_{who}.csv", newline="") as fh:
            rows = np.array([(float(r["time_s"]), float(r["freq_hz"])) for r in csv.DictReader(fh)])
        traces[who] = (rows[:, 0] + info["offset_s"], rows[:, 1])
    return traces
