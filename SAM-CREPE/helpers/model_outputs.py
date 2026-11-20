"""What the models produced for one audio file: SAM-Whistle's confidence map and contours, CREPE's scores.

THE INPUTS FOLDER: one subfolder per audio file (a selection window, or a whole recording), holding
    conf_map.npz        SAM-Whistle's confidence map: conf [freq x frame], freq_hz, time_s (made on Colab)
    crepe_act.npz       CREPE's scores: activation [frame x pitch], freq_hz, time (predict.py --save_activations)
    sam_input.wav       the audio SAM-Whistle ran on (stage A's graph search reads its spectrogram settings)
    sam_detections.bin  SAM-Whistle's own contours (optional; for comparison only)
    sample.json         offsets_s: for each file above, the seconds to add to its times to put them on the
                        recording's clock (where the file starts in the recording; for SAM-Whistle's files plus
                        half a frame, because SAM-Whistle labels a frame by its start).
"""
import json
from dataclasses import dataclass

import numpy as np

from helpers.silbido_io import read_silbido

REQUIRED_FILES = ("sample.json", "conf_map.npz", "crepe_act.npz")   # what load_model_outputs() cannot do without


def missing_files(folder):
    """The REQUIRED_FILES not in one input folder (all of them if the folder itself is missing)."""
    return [name for name in REQUIRED_FILES if not (folder / name).exists()]

@dataclass
class ScoreGrid:
    """One evidence source's scores on its own grid. Both sources use this layout."""
    name: str             # "conf" (SAM-Whistle) or "crepe"
    values: np.ndarray    # [time x freq] score per frame and frequency bin, 0-1, exactly as the model wrote it
    t: np.ndarray         # (n_time,) centre of each frame, seconds on the recording's clock
    f: np.ndarray         # (n_freq,) frequency of each bin, Hz

@dataclass
class ModelOutputs:
    """Everything the models produced for one audio file. Deliberately has no analyst field."""
    stem: str
    conf: ScoreGrid       # SAM-Whistle confidence: 5-50 kHz, 125 Hz apart, 2 ms frames
    crepe: ScoreGrid      # CREPE: 634 Hz-40.1 kHz, 20 cents apart (log), 2 ms frames, 1 ms later than conf
    sam: list             # [(t, f), ...] SAM-Whistle's own contours (frame centres); empty if not given

def file_offsets(folder):
    """The offsets_s of one input folder: {file name: seconds to add to put its times on the recording's clock}."""
    info = json.loads((folder / "sample.json").read_text())
    return info["offsets_s"]

def load_model_outputs(stem, inputs):
    """Load one audio file's model outputs from <inputs>/<stem>/ and put every time on the recording's clock.
    Scores are returned exactly as the models wrote them; the map is only transposed to [time x freq]."""
    folder = inputs / stem
    off = file_offsets(folder)
    c = np.load(folder / "conf_map.npz")
    a = np.load(folder / "crepe_act.npz")
    conf = ScoreGrid("conf", c["conf"].T, c["time_s"] + off["conf_map.npz"], c["freq_hz"].astype(float))
    crepe = ScoreGrid("crepe", a["activation"], a["time"] + off["crepe_act.npz"], a["freq_hz"].astype(float))
    sam = []
    if (folder / "sam_detections.bin").exists():
        sam = [(w[:, 0] + off["sam_detections.bin"], w[:, 1]) for w in read_silbido(folder / "sam_detections.bin")]
    return ModelOutputs(stem, conf, crepe, sam)
