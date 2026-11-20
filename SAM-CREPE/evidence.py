"""Evidence: is the whistle still there?

WHERE THIS IS USED
SAM-Whistle gives us fragments: pieces of a whistle with empty stretches between them and missing starts and
ends. This file is never used on the fragments themselves (SAM-Whistle already found those points). It is
used only in the empty stretches:
    join (step 3)    at every 2 ms moment inside a gap between two fragments, at the frequency on the line
                     joining them: is the whistle still there? If so, the two fragments are one whistle.
    extend (step 4)  moment by moment outward from each end of a whistle (forwards after its last point,
                     backwards before its first): does the whistle carry on? If so, the whistle is longer.

THE THREE SOURCES OF EVIDENCE
    map       SAM-Whistle's confidence map: for every 2 ms moment and every 125 Hz row from 5 to 50 kHz, a
              score from 0 to 1 for "is this part of a whistle?". It votes yes if it has a peak at or above
              map_threshold inside the search window. Peaks are found exactly the way SAM-Whistle's own graph
              search finds them, so "the map says yes" means "SAM-Whistle would have had a point here".
    crepe     CREPE's pitch scores: for every 2 ms moment, a score for each of 360 possible pitches for "is this
              the main pitch?". It votes yes if its single best pitch (its top pick) scores at least
              crepe_threshold and lies inside the search window. Only the top pick is used: experiment 2 found
              that CREPE's other pitches are at noise level, because it only reports one sound at a time.
    harmonic  The harmonics clean.py removed (use_harmonic_evidence). A harmonic is a fainter copy of the
              whistle at 2 or 3 times its frequency, so where a harmonic is, the whistle is too, even if the
              whistle itself was not traced there. It votes yes if a removed harmonic has a point at this moment
              whose frequency, divided by its multiple, lies inside the search window. Added with
              add_harmonics() after clean.py has run.

The scores mean different things, so they are never added or compared. Each source only votes yes or no at
its own threshold. Below 5 kHz the map has no rows, so only CREPE (and a harmonic above 5 kHz) can vote there.

STEADY TONES ARE NOT EVIDENCE (ignore_tone_bands)
A steady tone (like the ~20 kHz one in this recording) is full of map peaks and CREPE picks. Once clean.py has
found the tones, add_tone_bands() blocks their frequencies: nothing within tone_band_hz of a tone counts as
evidence, for any source.

Every setting used here is in settings.py. This file only reads model outputs (helpers/model_outputs.py),
never the analysts' traces, so the method cannot be tuned to the answers.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from settings import FRAME_S # noqa: E402


@dataclass(frozen=True)
class Moment:
    """What the evidence says about one 2 ms moment, near one expected frequency.

    verdict is the short answer: the sources that found the whistle, joined by "+", in the order map, crepe,
    harmonic (for example "map", "map+crepe", "crepe+harmonic"), or "none" if no source found it (the whistle
    is not there, or not visible).
    """
    t: float            # the moment, in seconds on the recording's clock
    map_hz: float       # where the map places the whistle (its peak nearest the expected frequency); NaN if no peak
    crepe_hz: float     # CREPE's top pick, if it counts (strong enough and inside the window); NaN if not
    crepe_score: float  # CREPE's top-pick score at this moment, whether or not it counts (shows how close it came)
    harmonic_hz: float  # a removed harmonic's frequency divided by its multiple, nearest the expected one; NaN if none
    verdict: str        # e.g. "map+crepe", or "none"

    @property
    def found(self):
        """True if at least one source found the whistle."""
        return self.verdict != "none"

    @property
    def sources(self):
        """The sources that found the whistle, e.g. ("map", "crepe"); empty if none."""
        return () if self.verdict == "none" else tuple(self.verdict.split("+"))

    @property
    def best_hz(self):
        """The frequency to put on the whistle at this moment. The map's wherever it has one, because its rows
        are evenly spaced (125 Hz) while CREPE's pitches are spaced by ratio (about 70 Hz apart at 6 kHz,
        350 Hz apart at 30 kHz). Next a harmonic's (it comes from the map too), then CREPE's, which is the
        only one below 5 kHz unless a harmonic is there. NaN when the verdict is "none"."""
        for hz in (self.map_hz, self.harmonic_hz, self.crepe_hz):
            if not np.isnan(hz):
                return hz
        return np.nan


def search_window_hz(slope_hz_per_s, settings):
    """How far above and below the expected frequency to look for the whistle, in Hz.

    The expected frequency is only a prediction (a straight line from the whistle's recent points), so the
    search must allow some error. The window is window_min_hz (500 Hz, SAM-Whistle's own peak tolerance), or
    wider when the whistle is sweeping fast: a whistle changing by 400 kHz per second moves 800 Hz in one
    2 ms moment, so a 500 Hz window could miss it. With window_slope_frames = 1 the window is never narrower
    than one moment's worth of the whistle's own frequency change.
    """
    return max(settings.window_min_hz, abs(slope_hz_per_s) * FRAME_S * settings.window_slope_frames)


def find_map_peaks(conf, settings):
    """Go through the confidence map once, before any questions are asked, and list for every 2 ms moment
    the frequencies where the map has a peak: the places SAM-Whistle's graph search treats as candidate
    whistle points. Only these peak frequencies are kept; the rest of the map is not needed afterwards.

    Peaks are found in the same steps, in the same order, as SAM-Whistle's graph search
    (TonalTracker._select_peaks, using find_peaks_simple and consolidate_peaks in sam_whistle/utils/tone.py):
      1. Local maxima: going up the frequency rows of one moment, a row whose score is higher than the rows
         just below and above it.
      2. Keep only those scoring at least map_threshold (0.5).
      3. A whistle is usually 3 to 4 rows thick, which can give two maxima side by side: of any two peaks
         fewer than peak_min_gap_rows rows apart, keep the higher, and repeat until none are that close.
      4. Click check (if skip_click_frames): an echolocation click lasts only a moment or two but covers
         almost every frequency, so the map suddenly has many peaks at once. If a moment has more extra peaks
         than click_extra_fraction of the rows (4 or more) compared with the last moment that was not a click,
         the whole moment is treated as a click and given no peaks. This stops a click from counting as
         whistle evidence in the middle of a gap.

    Returns (peaks, peak_scores, is_click), one entry per moment: peaks is an array of peak frequencies (Hz),
    peak_scores their map scores, and is_click marks the moments set aside as clicks (no peaks).
    """
    n_rows = conf.values.shape[1]
    previous_count = 0.25 * n_rows # SAM-Whistle's starting value, so the first moment is never a click
    peaks, peak_scores, is_click = [], [], []
    for scores in conf.values: # one moment at a time; scores has one value per frequency row
        rows = np.flatnonzero(np.diff(np.sign(np.diff(scores))) < 0) + 1 # step 1
        rows = rows[scores[rows] >= settings.map_threshold] # step 2
        gap = settings.peak_min_gap_rows
        while len(rows) > 1 and np.any(np.diff(rows) < gap): # step 3
            k = int(np.argmax(np.where(np.diff(rows) < gap, scores[rows[:-1]], -np.inf))) # strongest close pair
            rows = np.delete(rows, k + 1 if scores[rows[k]] >= scores[rows[k + 1]] else k) # drop its weaker peak
        click = False
        if len(rows) and settings.skip_click_frames: # step 4
            click = (len(rows) - previous_count) / n_rows > settings.click_extra_fraction
            if not click:
                previous_count = len(rows)
        if click:
            rows = rows[:0] # a click moment gives no peaks
        peaks.append(conf.f[rows])
        peak_scores.append(scores[rows])
        is_click.append(click)
    return peaks, peak_scores, np.array(is_click)


class Evidence:
    """All the evidence for one selection, prepared once, then asked about any moment with look_for_whistle().

    Preparing: the map's peaks are listed for every moment (find_map_peaks), and CREPE's top pick and its
    score are noted for every moment. Once clean.py has run, add_harmonics() adds the removed harmonics.
    After that, each question is a quick look-up.
    """

    def __init__(self, outputs, settings):
        """outputs: one selection's model outputs from helpers/model_outputs.py (no analyst data)."""
        self.settings = settings
        self.map_t = outputs.conf.t # time of each map moment
        self.map_peaks, self.map_peak_scores, self.map_is_click = find_map_peaks(outputs.conf, settings)
        self.crepe_t = outputs.crepe.t # time of each CREPE moment
        top = np.argmax(outputs.crepe.values, axis=1) # CREPE's top pick at each moment
        self.crepe_top_hz = outputs.crepe.f[top]
        self.crepe_top_score = outputs.crepe.values[np.arange(len(top)), top]
        # for every map moment, the fundamentals implied by removed harmonics (empty until add_harmonics)
        self.harmonic_fundamental_hz = [np.array([]) for _ in self.map_t]

    def _map_index(self, t):
        """The map moment at time t, or None if t is not on the map's clock."""
        i = int(np.rint((t - self.map_t[0]) / FRAME_S))
        return i if 0 <= i < len(self.map_t) and abs(self.map_t[i] - t) <= FRAME_S / 2 else None

    def add_harmonics(self, harmonics):
        """Record the harmonics clean.py removed, as evidence of their whistle.

        harmonics: list of (points, multiple), points an [n x 2] array of (time s, freq Hz) on the map's clock,
        multiple 2 to harmonic_max_multiple. At each of its moments, a harmonic says "the whistle is at
        freq / multiple".

        The harmonic's own frequency is kept too (harmonic_source_hz): if the harmonic itself lies on a steady
        tone, the fundamental it implies is not evidence (a 20.8 kHz tone piece would otherwise put false
        evidence at 10.4 kHz). Checked when asked, so tone bands may be added before or after this."""
        found = [[] for _ in self.map_t]
        for points, multiple in harmonics:
            for t, f in points:
                i = self._map_index(t)
                if i is not None:
                    found[i].append((f / multiple, f))
        found = [sorted(x) for x in found]
        self.harmonic_fundamental_hz = [np.array([a for a, _ in x]) for x in found]
        self.harmonic_source_hz = [np.array([b for _, b in x]) for x in found]
        self.__dict__.pop("_observation_cache", None) # the observations change, so forget any cached ones

    def _usable_harmonics(self, i):
        """The fundamentals implied by removed harmonics at map moment i, leaving out any that lie on a tone band
        or come from a harmonic that lies on one."""
        implied = self.harmonic_fundamental_hz[i]
        source = getattr(self, "harmonic_source_hz", [np.array([])] * len(self.map_t))[i]
        if not len(implied):
            return implied
        return implied[self._off_tone(implied) & self._off_tone(source)]

    def add_tone_bands(self, tone_hz):
        """Record the frequencies of the steady tones clean.py removed (one frequency per tone, e.g. its median).

        A steady tone, such as the ~20 kHz one in this recording, is full of map peaks and CREPE picks, so a
        join or extension could travel along it as if it were the whistle (the join diagnostic figure showed
        exactly that). With ignore_tone_bands on, evidence within tone_band_hz of a tone's frequency is not
        counted, anywhere in the selection's window: a tone does not move, so its frequency is blocked
        throughout. A real whistle crossing the band only loses a few moments, which a path can skip."""
        self.tone_hz = np.array(sorted(tone_hz), float)
        self.__dict__.pop("_observation_cache", None) # the observations change, so forget any cached ones

    def _off_tone(self, hz):
        """For an array of frequencies, True where it is NOT inside a tone band (always True if
        ignore_tone_bands is off or no tones were added)."""
        hz = np.asarray(hz, float)
        tones = getattr(self, "tone_hz", np.array([]))
        if not self.settings.ignore_tone_bands or not len(tones) or not hz.size:
            return np.ones(hz.shape, bool)
        return np.min(np.abs(hz[..., None] - tones), axis=-1) > self.settings.tone_band_hz

    def _crepe_index(self, t):
        """The CREPE moment paired with map moment t: the first one at or after t (CREPE's moments sit 1 ms after
        the map's and analyse almost the same sound), as in experiment 2. None if there is none."""
        j = int(np.searchsorted(self.crepe_t, t - 1e-9))
        return j if 0 <= j < len(self.crepe_t) and abs(self.crepe_t[j] - t) <= FRAME_S / 2 + 1e-9 else None

    def observations_at(self, t, use_map=True, use_crepe=True, use_harmonics=None, n=None, m=None):
        """Every piece of evidence at moment t, at any frequency: the observations join.py's route search
        can step on. Returns a list of (frequency Hz, source), source "map", "crepe" or "harmonic". Uses the
        same rules as look_for_whistle (thresholds, clicks skipped, tone bands ignored), without a window.

        An observation only counts if it is part of a run ("n of m"): take every window of m consecutive moments that
        includes this one (m = observations_run_window); if in at least one of them n or more moments (n = observations_min_run,
        this moment included) have an observation on the same track, the observation counts. "On the same track" means within
        observations_track_hz for each moment apart (from any source). A whistle leaves a run of observations; a click, a
        stray CREPE pick or noise leaves a single dot, which a route could otherwise use to jump.
            n = m = 3   "3 in a row"
            n = 3, m = 5   also keeps a faint whistle caught only every 2nd or 3rd moment
        n, m: give them to use other values than the join's (observations_min_run, observations_run_window);
        extend.py passes its own (extend_observations_min_run, extend_observations_run_window)."""
        if use_harmonics is None:
            use_harmonics = self.settings.use_harmonic_evidence
        flags = (use_map, use_crepe, use_harmonics)
        here = self._raw_observations(t, flags)
        n = self.settings.observations_min_run if n is None else n
        m = self.settings.observations_run_window if m is None else m
        if n <= 1:
            return here
        out = []
        for hz, source in here:
            on_track = {0} # moments (relative to t) with an observation on this track
            for d in range(-(m - 1), m):
                if d and any(abs(g - hz) <= self.settings.observations_track_hz * abs(d)
                             for g, _ in self._raw_observations(t + d * FRAME_S, flags)):
                    on_track.add(d)
            windows = range(-(m - 1), 1) # each window of m moments that includes this one
            if any(sum(w <= d < w + m for d in on_track) >= n for w in windows):
                out.append((hz, source))
        return out

    def _raw_observations(self, t, flags):
        """All evidence at moment t before the run check (cached, because every route asks about the same
        moments many times)."""
        cache = self.__dict__.setdefault("_observation_cache", {})
        key = (int(np.rint((t - self.map_t[0]) / FRAME_S)), flags)
        if key not in cache:
            cache[key] = self._observations_here(t, *flags)
        return cache[key]

    def _observations_here(self, t, use_map, use_crepe, use_harmonics):
        """The evidence at moment t: map peaks, CREPE's top pick and harmonic fundamentals, off the tone bands."""
        out = []
        i, j = self._map_index(t), self._crepe_index(t)
        if use_map and i is not None:
            p = self.map_peaks[i]
            out += [(float(hz), "map") for hz in p[self._off_tone(p)]]
        if use_crepe and j is not None and self.crepe_top_score[j] >= self.settings.crepe_threshold:
            if self._off_tone([self.crepe_top_hz[j]])[0]:
                out.append((float(self.crepe_top_hz[j]), "crepe"))
        if use_harmonics and i is not None:
            out += [(float(hz), "harmonic") for hz in self._usable_harmonics(i)]
        return out

    def look_for_whistle(self, t, expected_hz, window_hz, use_map=True, use_crepe=True, use_harmonics=None):
        """Is the whistle at moment t, within window_hz above or below expected_hz? Returns a Moment.

        t            the moment, seconds on the recording's clock (a multiple of 2 ms on the map's clock)
        expected_hz  where the whistle should be if it carries on (predicted by the join or extend step)
        window_hz    how far above and below to look (search_window_hz)
        use_map, use_crepe, use_harmonics
                     switch a source off, for the comparison runs (map only, or CREPE required). use_harmonics
                     defaults to the use_harmonic_evidence setting.
        """
        if use_harmonics is None:
            use_harmonics = self.settings.use_harmonic_evidence
        map_hz = crepe_hz = crepe_score = harmonic_hz = np.nan
        i = self._map_index(t)

        # The map's vote. Among the map's peaks at this moment that are inside the window (and not on a steady
        # tone), choose one.
        if use_map and i is not None:
            usable = self._off_tone(self.map_peaks[i]) & (np.abs(self.map_peaks[i] - expected_hz) <= window_hz)
            inside = self.map_peaks[i][usable]
            if len(inside):
                if self.settings.peak_choice == "nearest":
                    # the peak closest to where the whistle should be: if another whistle or a harmonic is
                    # also inside the window, this stays on the same whistle instead of jumping to the other
                    map_hz = float(inside[np.argmin(np.abs(inside - expected_hz))])
                else:
                    # "strongest": the highest-scoring peak inside the window (to test; map peaks are usually
                    # all near 1.0, so this is close to arbitrary)
                    map_hz = float(inside[np.argmax(self.map_peak_scores[i][usable])])

        # CREPE's vote, from the CREPE moment paired with t (see _crepe_index).
        j = self._crepe_index(t)
        if j is not None:
            crepe_score = float(self.crepe_top_score[j])
            if (use_crepe and crepe_score >= self.settings.crepe_threshold
                    and abs(self.crepe_top_hz[j] - expected_hz) <= window_hz
                    and self._off_tone([self.crepe_top_hz[j]])[0]):
                crepe_hz = float(self.crepe_top_hz[j])

        # The harmonics' vote. Among the fundamentals implied by removed harmonics at this moment, the one
        # nearest the expected frequency, if it is inside the window (and not on a steady tone).
        if use_harmonics and i is not None:
            implied = self._usable_harmonics(i)
            inside = implied[np.abs(implied - expected_hz) <= window_hz]
            if len(inside):
                harmonic_hz = float(inside[np.argmin(np.abs(inside - expected_hz))])

        found = [name for name, hz in (("map", map_hz), ("crepe", crepe_hz), ("harmonic", harmonic_hz))
                 if not np.isnan(hz)]
        return Moment(t, map_hz, crepe_hz, crepe_score, harmonic_hz, "+".join(found) or "none")
