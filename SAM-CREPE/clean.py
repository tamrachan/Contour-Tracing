"""Step 2 of SAM-CREPE: clean. Remove fragments that are not whistles before anything is joined or extended,
and keep the harmonics as evidence that their whistle is there.

WHERE THIS FITS IN THE PIPELINE
    A  get_fragments.py   SAM-Whistle's graph search cuts each confidence map into fragments
    B1 load               fragments, confidence map and CREPE scores, all on the recording's clock
    B2 clean              (this file)
    B3 join               join fragments of the same whistle across gaps (uses evidence.py)
    B4 extend             extend each whistle's start and end (uses evidence.py)
    B5 finalise           protocol rules (including the tone rule again); write the results

WHY CLEAN FIRST
Joining looks for fragments that line up in time and frequency. Two kinds of fragment would line up for the
wrong reason and must be removed first:
    steady tones   A sound that stays at almost one frequency, like the constant ~19.5 kHz tone in this
                   recording (probably equipment or a vessel), not a dolphin whistle. A whistle could be
                   joined onto it.
    harmonics      A whistle is a fundamental frequency plus fainter copies at 2, 3, ... times that frequency
                   (harmonics). SAM-Whistle traces many harmonics as separate fragments. They run parallel to
                   the whistle, so they could be joined to each other or mistaken for its continuation. We
                   keep the fundamental (the lowest) and remove its harmonics.
A removed harmonic still shows that its whistle is there, even where the whistle itself was not traced. So
the harmonics are handed to evidence.py (add_harmonics), where they count as evidence at their frequency
divided by their multiple (use_harmonic_evidence).

THE RULES (all values in settings.py)
    tone      a fragment lasting longer than tone_min_ms (100 ms) whose frequency changes by less than
              tone_max_range_hz (1 kHz) over its whole length. The same definition the step 4 comparison used.
              Shorter pieces of a tone are not caught here; finalise applies the rule again to the finished
              whistles (finalise_recheck_tones).
    harmonic  a fragment that overlaps a lower, non-tone fragment for at least harmonic_min_overlap_ms (20 ms),
              and whose frequency over that overlap is, in the middle (median), 2, 3, ... (up to
              harmonic_max_multiple, 5) times the lower fragment's, within harmonic_tolerance (8%).
              The lower fragment is kept.
    lone harmonic (check_lone_harmonics)
              a fragment with no lower fragment beside it may still be a harmonic, of a fundamental that
              SAM-Whistle did not trace (too faint, too short, or below 5 kHz). It is checked with evidence.py
              at 1/2, 1/3, ... up to 1/harmonic_max_multiple of its frequency: if the map or CREPE finds a fundamental there at
              lone_harmonic_min_share (half) of its moments or more, it is treated as a harmonic too.
              Otherwise it is kept, because it may be all that is left of a whistle.
"""
from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)
class Removed:
    """A fragment clean() removed, and why, for decisions.csv."""
    index: int          # its position in the list clean() was given
    reason: str         # "tone", "harmonic" or "lone harmonic"
    detail: str         # the numbers behind the decision in words
    flat: bool = False  # a flat harmonic (flat_harmonics_as_tones): no harmonic evidence; its frequency is blocked as a tone

@dataclass(frozen=True)
class HarmonicCheck:
    """The result of comparing one fragment with a possible fundamental below it."""
    is_harmonic: bool   # True if the upper fragment is a harmonic of the lower one
    multiple: int       # which harmonic: 2 (twice the frequency), 3, ...; 0 if it is not one
    ratio: float        # the measured frequency ratio over the overlap (NaN if they overlap too little to compare)
    overlap_ms: float   # how long the two fragments overlap in time

def is_steady_tone(fragment, settings):
    """True if the fragment lasts longer than tone_min_ms and its frequency stays within tone_max_range_hz."""
    duration_ms = 1000 * np.ptp(fragment.points[:, 0])
    freq_range_hz = np.ptp(fragment.points[:, 1])
    return duration_ms > settings.tone_min_ms and freq_range_hz < settings.tone_max_range_hz

def is_flat(fragment, settings):
    """True if the fragment lasts at least flat_harmonic_min_ms while its frequency stays within
    tone_max_range_hz: a flat line, too short to be called a tone but probably a piece of one."""
    duration_ms = 1000 * np.ptp(fragment.points[:, 0])
    return duration_ms >= settings.flat_harmonic_min_ms and np.ptp(fragment.points[:, 1]) < settings.tone_max_range_hz

def check_harmonic(upper, lower, settings):
    """Does `upper` behave as a harmonic of `lower` where they overlap? Returns a HarmonicCheck.

    Over the stretch of time both fragments cover, compare upper's frequency with lower's at the same moments
    (lower's frequency is read off a straight line between its points) and take the middle value (median) of
    the ratio, so a few odd points do not decide it. It is a harmonic if that ratio is within
    harmonic_tolerance of 2, 3, ... up to harmonic_max_multiple."""
    t0 = max(upper.points[0, 0], lower.points[0, 0])
    t1 = min(upper.points[-1, 0], lower.points[-1, 0])
    overlap_ms = 1000 * (t1 - t0)
    if overlap_ms < settings.harmonic_min_overlap_ms:
        return HarmonicCheck(False, 0, np.nan, overlap_ms)
    both = upper.points[(upper.points[:, 0] >= t0) & (upper.points[:, 0] <= t1)]
    ratio = float(np.median(both[:, 1] / np.interp(both[:, 0], lower.points[:, 0], lower.points[:, 1])))
    for multiple in range(2, settings.harmonic_max_multiple + 1):
        if abs(ratio - multiple) / multiple < settings.harmonic_tolerance:
            return HarmonicCheck(True, multiple, ratio, overlap_ms)
    return HarmonicCheck(False, 0, ratio, overlap_ms)

def check_lone_harmonic(fragment, evidence, settings):
    """For a fragment with no lower fragment beside it: is there a fundamental at 1/2, 1/3, ... (up to 1/harmonic_max_multiple) of its frequency
    that the map or CREPE can see? Returns (multiple, share): the multiple whose fundamental was found at the
    largest share of the fragment's moments, and that share. Only the map and CREPE are asked, not other
    harmonics, so a harmonic can never be confirmed by another harmonic."""
    best_multiple, best_share = 0, 0.0
    for multiple in range(2, settings.harmonic_max_multiple + 1):
        found = [evidence.look_for_whistle(t, f / multiple, settings.window_min_hz, use_harmonics=False).found
                 for t, f in fragment.points]
        share = float(np.mean(found))
        if share > best_share:
            best_multiple, best_share = multiple, share
    return best_multiple, best_share

def clean(fragments, settings, evidence=None):
    """Remove steady tones, then harmonics, then (if evidence is given and check_lone_harmonics is on) lone harmonics.

    Returns (kept, removed, harmonics):
        kept       the fragments that remain
        removed    a list of Removed, one per removed fragment, with the reason
        harmonics  the removed harmonics as (points, multiple), for evidence.add_harmonics()

    Tones are found first, so a tone can never count as the "lower fragment" that makes another fragment a harmonic. 
    Each remaining fragment is compared with every other remaining fragment as its possible fundamental; 
    it is removed if any of them makes it a harmonic."""
    removed, harmonics = [], []
    tone = [is_steady_tone(f, settings) for f in fragments]
    for i, f in enumerate(fragments):
        if tone[i]:
            removed.append(Removed(i, "tone", f"{1000 * np.ptp(f.points[:, 0]):.0f} ms within "
                                              f"{np.ptp(f.points[:, 1]):.0f} Hz, at {np.median(f.points[:, 1]):.0f} Hz"))
    for i, f in enumerate(fragments):
        if tone[i]:
            continue
        for k, lower in enumerate(fragments):
            if k == i or tone[k]:
                continue
            check = check_harmonic(f, lower, settings)
            if check.is_harmonic:
                flat = settings.flat_harmonics_as_tones and is_flat(f, settings)
                removed.append(Removed(i, "harmonic", f"{check.multiple}x fragment {k}: frequency ratio "
                                                      f"{check.ratio:.2f} over {check.overlap_ms:.0f} ms of overlap"
                                                      + ("; flat, so blocked as a tone" if flat else ""), flat))
                if not flat:
                    harmonics.append((f.points, check.multiple))
                break
    if evidence is not None and settings.check_lone_harmonics:
        gone = {r.index for r in removed}
        for i, f in enumerate(fragments):
            if i in gone:
                continue
            multiple, share = check_lone_harmonic(f, evidence, settings)
            if share >= settings.lone_harmonic_min_share:
                flat = settings.flat_harmonics_as_tones and is_flat(f, settings)
                removed.append(Removed(i, "lone harmonic", f"no lower fragment, but a fundamental at 1/{multiple} "
                                                           f"of its frequency at {share:.0%} of its moments"
                                                           + ("; flat, so blocked as a tone" if flat else ""), flat))
                if not flat:
                    harmonics.append((f.points, multiple))
    gone = {r.index for r in removed}
    kept = [f for i, f in enumerate(fragments) if i not in gone]
    return kept, sorted(removed, key=lambda r: r.index), harmonics

def tone_frequencies(fragments, removed):
    """The frequency (median) of each fragment removed as a steady tone or as a flat harmonic, for evidence.add_tone_bands()."""
    return [float(np.median(fragments[r.index].points[:, 1])) for r in removed if r.reason == "tone" or r.flat]

def recurring_tones(tones_per_selection, settings):
    """Tones that recur across a recording: frequencies found as a tone (tone_frequencies) in at least
    tone_min_selections selections of the same recording, within tone_band_hz of each other.

    A steady tone (such as the ~20 kHz one in the development recording) can run through a whole recording, but SAM-Whistle only
    traces it as a fragment in some selections; in the others it is still there and can carry a join route
    (the join diagnostic figure showed this in sel 026). Blocking recurring tones in every selection of the
    recording (recording_tone_bands) closes that gap, while a one-off flat sound in one selection (for example
    at 6 kHz, where whistles often turn) is not blocked everywhere.

    tones_per_selection: {selection name: list of tone frequencies}. Returns a sorted list of frequencies (the
    middle of each recurring group)."""
    found = sorted((hz, sel) for sel, tones in tones_per_selection.items() for hz in tones)
    groups, current = [], []
    for hz, sel in found: # group frequencies within tone_band_hz of the last one
        if current and hz - current[-1][0] > settings.tone_band_hz:
            groups.append(current)
            current = []
        current.append((hz, sel))
    if current:
        groups.append(current)
    return [float(np.median([hz for hz, _ in g])) for g in groups
            if len({sel for _, sel in g}) >= settings.tone_min_selections]

def remove_tone_band_fragments(fragments, kept, tone_hz, settings):
    """Remove kept fragments that sit flat inside a tone band (remove_tone_band_fragments): pieces of a steady tone
    too short (under tone_min_ms) to be removed as a tone by clean(). Called once the tone bands are known (the
    selection's own tones and, with recording_tone_bands, those recurring across the recording), so in run.py's
    second pass. Join otherwise routed whistles up to such pieces (sel 032 and 051, at ~20 to 21 kHz).

    A fragment is removed if its frequency range is under tone_max_range_hz and its median lies within tone_band_hz
    of a tone. Returns (kept, removed): removed as Removed rows numbered as in `fragments`."""
    if not (settings.remove_tone_band_fragments and settings.ignore_tone_bands and len(tone_hz)):
        return kept, []
    tones = np.asarray(tone_hz, float)
    position = {id(f): i for i, f in enumerate(fragments)}
    out, removed = [], []
    for f in kept:
        median = float(np.median(f.points[:, 1]))
        nearest = float(tones[np.argmin(np.abs(tones - median))])
        if np.ptp(f.points[:, 1]) < settings.tone_max_range_hz and abs(median - nearest) <= settings.tone_band_hz:
            removed.append(Removed(position[id(f)], "tone band", f"flat piece at {median:.0f} Hz, within "
                                                               f"{settings.tone_band_hz:.0f} Hz of the tone at {nearest:.0f} Hz", True))
        else:
            out.append(f)
    return out, removed
