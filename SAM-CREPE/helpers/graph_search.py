"""SAM-Whistle's graph search on a saved confidence map. Adapted from scripts/experiments/check_graph_search.py, 
using the copy of SAM-Whistle in SAM-CREPE/external/SAM-Whistle/.

Importing this module starts Java (SAM-Whistle's tonal_extraction package does so when imported, to load the
silbido jars), so JAVA_HOME must point to a JDK.

Two differences from check_graph_search.py, neither of which changes the result:
    - the audio is read from the inputs folder (<inputs>/<stem>/sam_input.wav) instead of SAM-Whistle's own data
      layout (<root_dir>/audio/<stem>.wav), so any inputs folder works;
    - SAM-Whistle's tracker would also load annotation files if it found any; here it never does, so stage A
      cannot see the analysts' traces.
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np

from helpers.paths import SAM_WHISTLE

sys.path.insert(0, str(SAM_WHISTLE))
import tyro  # noqa: E402
from sam_whistle.config import TonalConfig  # noqa: E402
from sam_whistle.evaluate.tonal_extraction.tonal_tracker import TonalTracker  # noqa: E402

# the flags of the Colab run that made the confidence maps, minus the ones that only choose the model and the
# thresholds to sweep. --use_conf makes TonalTracker expect a confidence map (0 to 1) rather than a dB
# spectrogram; the spect_cfg flags must match because they set the spectrogram shape the map has to fit.
COLAB_FLAGS = ["--use_conf", "--spect_cfg.normalize", "zscore", "--spect_cfg.no_center",
               "--spect_cfg.interp", "linear", "--spect_cfg.kernel_size", "3", "--log_dir", "logs/sam"]


class _Tracker(TonalTracker):
    """SAM-Whistle's TonalTracker, reading its audio from a given file and never loading annotations."""

    def __init__(self, cfg, stem, audio_file):
        self._audio_file = Path(audio_file)
        super().__init__(cfg, stem)

    def _load_spectrogram(self, stem):
        self.audio_dir = self._audio_file.parent # SAM-Whistle reads <audio_dir>/<name>.wav
        return super()._load_spectrogram(self._audio_file.stem)

    def _load_gt_tonals(self, stem):
        return [] # no annotations in the method


def contours_match(a, b, tol=1e-9):
    """Same number of contours with the same points, in any order. Returns (match, max abs difference).

    a, b: lists of arrays with columns (time s, freq Hz). The Java graph code does not promise an order,
    so both lists are sorted by (start time, start frequency, length) before pairing them up.
    """
    if len(a) != len(b):
        return False, float("nan") # different number of contours
    key = lambda c: (round(float(c[0, 0]), 6), round(float(c[0, 1]), 3), len(c))
    worst = 0.0
    for x, y in zip(sorted(a, key=key), sorted(b, key=key)):
        if x.shape != y.shape:
            return False, float("nan") # a pair with different numbers of points
        worst = max(worst, float(np.abs(x - y).max())) # largest time or frequency difference
    return worst <= tol, worst


def kept_edges(tracker):
    """SAM-Whistle's TonalTracker.get_tonals() (tonal_tracker.py:376-423), line for line, except that
    fragments with 3 points or fewer are skipped instead of crashing, and each piece's graph and edge are kept
    (get_fragments.py needs to know where each piece was cut at a crossing).

    get_tonals() keeps a fragment if it is longer than minlen and its points are not too sparse: the mean
    time from each point to the point 3 later must be under 18 frames (36 ms). That sparsity check raises
    ValueError on fragments of 3 points or fewer; skipping them is what the check would do anyway.

    Returns (kept, n_skipped): kept is a list of (graph id, edge, points array (time s, freq Hz)).
    """
    kept, n_skipped = [], 0
    for subgraph in tracker.active_set.getResultGraphs():
        graph = subgraph.disambiguate(tracker.cfg.disambiguate_s, tracker.freq_bin, False, False)
        for edge in graph.topological_sort():
            tone = edge.content
            if tone.get_duration() <= tracker.minlen_s:
                continue # too short (get_tonals counts these as discarded)
            samples = np.array(tone.get_time()) / 0.002 # as stat_avg_nth_wait_times(tone, 3)
            if len(samples) <= 3:
                n_skipped += 1 # where get_tonals() would crash
                continue
            if np.mean(samples[3:] - samples[:-3]) < 18:
                kept.append((int(subgraph.getGraphId()), edge, np.array([[t.time, t.freq] for t in tone])))
    return kept, n_skipped


def build_tracker(stem, conf, audio_file, minlen_ms, maxgap_ms, threshold):
    """A TonalTracker whose graph has been built from a saved confidence map, ready for kept_edges().

    conf: the map as saved (361 rows, row 0 = 5 kHz; one column per 2 ms frame). audio_file: the audio SAM-Whistle
    ran on; the tracker builds its spectrogram from it only for the shape check and its settings (hop,
    frequency offset, ...), and its values are then replaced by the map."""
    # SAM-Whistle's own config parser, as its eval_tonal.py uses, so every default we don't set (max slope
    # 1000 Hz/ms, ...) is SAM-Whistle's own. minlen and maxgap reach both the Python filter and the Java pruning.
    cfg = tyro.cli(TonalConfig, args=COLAB_FLAGS + ["--root_dir", str(Path(audio_file).parent),
                                                   "--minlen_ms", str(minlen_ms), "--maxgap_ms", str(maxgap_ms)])
    with contextlib.redirect_stdout(io.StringIO()): # TonalTracker prints per-file chatter
        tracker = _Tracker(cfg, stem, audio_file)
    assert tracker.spect_map.shape == conf.shape, f"{stem}: map {conf.shape} vs tracker {tracker.spect_map.shape}"
    tracker.spect_map = conf # what sam_inference() would have set
    # eval_tonal.extract_tonals() without its scoring and saving
    tracker.reset() # empty the graph, start at frame 0
    tracker.thre = threshold
    with contextlib.redirect_stdout(io.StringIO()):
        tracker.build_graph() # peaks per frame, linked into a graph (Java)
    return tracker
