"""Step 5 of SAM-CREPE: apply the protocol's rules to the finished whistles, and write the results.

WHERE THIS FITS IN THE PIPELINE
    A  get_fragments.py   SAM-Whistle's graph search cuts each confidence map into fragments
    B1 load               fragments, confidence map and CREPE scores, all on the recording's clock
    B2 clean              remove steady tones and harmonics (harmonics kept as evidence, tones blocked)
    B3 join               join fragments of the same whistle across gaps
    B3b merge             merge the same whistle traced twice side by side (optional)
    B4 extend             extend each whistle's start and end
    B5 finalise           (this file) protocol rules; write the results

THE RULES (finalise; all values in settings.py)
    1. Steady tones again (finalise_recheck_tones): clean.py only removes fragments longer than tone_min_ms, so
       short pieces of a steady tone can survive and be joined into one long steady line. The same tone rule
       (clean.is_steady_tone) is applied to each finished whistle.
    2. The protocol's minimum: a whistle must be longer than min_whistle_ms (100 ms); anything shorter is a
       chirp, not a whistle. This is applied here, at the end, so that short fragments could first be joined
       and extended into whistles.
With finalise off, every whistle is written as it is (for the comparison runs R1 to R5).

WHAT IS WRITTEN (write_selection), for each audio file SAM-Whistle ran on, in SAM-CREPE/outputs/<run>/:
    <name>.bin              the whistles, in SAM-Whistle's own file format (silbido .bin, written with SAM-Whistle's
                            own writer) and time base (seconds from the start of that audio file, frames labelled
                            by their start), so they can be read and scored exactly like SAM-Whistle's own output
    <name>_provenance.csv   one row per point: whistle, time (seconds on the recording's clock), frequency, and
                            where the point came from ("sam", "map", "crepe", "harmonic", "interpolated", "extend ...")
run.py writes decisions.csv and settings.json for the whole run.

TWO CLOCKS
The pipeline works on the recording's clock, so that pieces cut from one long recording (like the 3 s windows)
line up with each other and with the analysts. The .bin file is written on the audio file's own clock, as
SAM-Whistle writes it. offset_s is the difference: what get_fragments.py added to SAM-Whistle's times (where the
audio file starts in the recording, plus half a frame to move from a frame's start to its centre). It is saved
with the fragments and taken off again here. For a whole recording run as one file, it is just the half frame.
"""
import csv
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from clean import is_steady_tone # noqa: E402
from helpers.paths import SAM_WHISTLE_WRITER # noqa: E402

# SAM-Whistle's own .bin writer (from the copy in external/SAM-Whistle/). It is plain Python, so it is loaded
# straight from its file: importing the whole sam_whistle package would also start Java and load torch.
_spec = importlib.util.spec_from_file_location("sam_write_binary", SAM_WHISTLE_WRITER)
_writer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_writer)

@dataclass
class Contour:
    """One whistle in the form SAM-Whistle's writer takes. Its docstring also allows a plain list of
    (time, frequency) pairs, but its code then fails (it calls dataclasses.fields on the first contour), so a
    dataclass with time and freq is used, as SAM-Whistle itself does."""
    time: np.ndarray # s, from the start of the audio file SAM-Whistle ran on (frames labelled by their start)
    freq: np.ndarray # Hz

def finalise(whistles, settings):
    """Apply the rules above. Returns (kept whistles, decisions): one decision row per whistle, kept or not."""
    decisions = []
    if not settings.finalise:
        return list(whistles), decisions
    kept = []
    for w in whistles:
        duration_ms = 1000 * np.ptp(w.points[:, 0])
        row = dict(step="finalise", fragments=" ".join(map(str, w.fragments)), duration_ms=round(duration_ms, 1),
                   start_s=round(float(w.points[0, 0]), 4), median_hz=round(float(np.median(w.points[:, 1]))))
        if settings.finalise_recheck_tones and is_steady_tone(w, settings):
            row["decision"] = (f"dropped: steady tone ({duration_ms:.0f} ms within "
                               f"{np.ptp(w.points[:, 1]):.0f} Hz)")
        elif not duration_ms > settings.min_whistle_ms:
            row["decision"] = f"dropped: not longer than {settings.min_whistle_ms:.0f} ms (a chirp)"
        else:
            row["decision"] = "kept"
            kept.append(w)
        decisions.append(row)
    return kept, decisions

def write_selection(out, name, whistles, offset_s):
    """Write <name>.bin and <name>_provenance.csv into the run folder `out` (see the module notes).
    name: the audio file's name without its extension. offset_s: see TWO CLOCKS in the module notes."""
    contours = [Contour(w.points[:, 0] - offset_s, w.points[:, 1]) for w in whistles]
    _writer.writeTimeFrequencyBinary(str(out / f"{name}.bin"), contours)
    with open(out / f"{name}_provenance.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["whistle", "time_s", "freq_hz", "source"])
        for k, w in enumerate(whistles):
            for (t, hz), source in zip(w.points, w.sources):
                writer.writerow([k, f"{t:.4f}", f"{hz:.1f}", source])
