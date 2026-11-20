"""Step 4 of SAM-CREPE: extend each whistle's start and end, moment by moment, while the evidence continues.

WHERE THIS FITS IN THE PIPELINE
    A  get_fragments.py   SAM-Whistle's graph search cuts each confidence map into fragments
    B1 load               fragments, confidence map and CREPE scores, all on the recording's clock
    B2 clean              remove steady tones and harmonics (harmonics kept as evidence, tones blocked)
    B3 join               join fragments of the same whistle across gaps
    B3b merge             merge the same whistle traced twice side by side (optional)
    B4 extend             (this file) extend each whistle's start and end
    B5 finalise           protocol rules; write the results

WHY
SAM-Whistle misses the start of most whistles: on the 33 selections, after join and merge, 44 of 57 analyst
traces start more than 4 ms before our whistle (median 18 ms). Ends are mostly right (median 0 ms). In the missed
starts CREPE is the main evidence (43% of the moments, the map 29%): this is where CREPE adds the most.
Extending is also risky: the map stays on just after a whistle stops (its frames are 8 ms long), and in the
first 10 ms past ends that are already right the map fires at 28% of the moments, CREPE at 7%.

HOW ONE END IS EXTENDED (all values in settings.py)
Walk outward from the tip (forwards after the last point, backwards before the first), one 2 ms moment at a time:
    1. Expected frequency: a straight line through the last fit_ms of the whistle's points, refitted after every
       accepted point, so the extension can follow a curve. Where to look (extend_prediction): "line", the search
       window around that line; or "free", as widely around the last point as the slope limit allows, still taking
       the evidence nearest the line, so the extension can follow a turn when there is no evidence on the line
       (many whistles start with a short hook that falls before the minimum, the opposite of the line's direction).
    2. Observations at this moment (evidence.observations_at: map peaks, CREPE's top pick, harmonic fundamentals;
       tone bands blocked), with extend's own n of m rule (extend_observations_min_run, _run_window).
       Kept if inside the search window around the expected frequency, and if the slope limit and the octave
       rule allow the step from the last accepted point (as in join), and if the extension would not climb or fall
       faster than max_sustained_slope_hz_per_s over sustained_slope_window_ms (as in join; a click is a vertical
       line, so an extension could otherwise climb one).
    3. Which sources may extend (extend_rule):
           crepe_required  (default) only moments where CREPE agrees. On the 33 selections it gave the same
                           coverage as map_carry (0.90, from 0.88) with 3 points past an analyst's end instead of 29
           map_carry       CREPE always; the map or a harmonic alone only within map_carry_ms of the last moment
                           CREPE agreed (CREPE agreeing at the whistle's own tip counts; with extend_map_fallback the
                           tip itself counts, so the map alone may add at most map_carry_ms)
           map_only        only the map or a harmonic
       The point takes the map's frequency where it has one (evenly spaced rows), then a harmonic's, then CREPE's.
    4. Stop when: no accepted moment for more than extend_stop_unsupported_ms (the protocol: a gap of more
       than 20 ms makes two whistles), the end has grown by max_extension_ms, the whistle would become longer
       than max_whistle_ms (if set), another whistle is at this moment within window_min_hz of the expected
       frequency (extend_stop_at_other; whistles may overlap in time, this only stops one running onto another's
       trace: on the 33 selections all such stops were alongside parallel duplicates, none at two crossing
       whistles), or the edge of the map is reached.
Only accepted moments are added (never the unsupported tail); moments skipped between them are filled in
(fill_method, as in join) and marked "interpolated". Each added point is marked "extend" plus its source.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from settings import FRAME_S # noqa: E402
from evidence import search_window_hz # noqa: E402
from join import Whistle, fill_route, step_allowed, sustained_slope # noqa: E402

def direction_slope(points, settings):
    """The whistle's slope (Hz per second) at its tip: a straight line through its last fit_ms of points.
    points are in walking order, so the tip is the last row. 0 if there are fewer than two points."""
    near = points[np.abs(points[:, 0] - points[-1, 0]) <= settings.fit_ms / 1000 + FRAME_S / 4]
    if len(near) < 2:
        return 0.0
    return float(np.polyfit(near[:, 0], near[:, 1], 1)[0])

def crepe_agrees(evidence, t, hz, settings):
    """Does CREPE's top pick agree with a whistle at (t, hz)? (score at least crepe_threshold, within the window)"""
    m = evidence.look_for_whistle(t, hz, settings.window_min_hz, use_map=False, use_crepe=True, use_harmonics=False)
    return "crepe" in m.sources

def someone_else_here(t, hz, window, others):
    """Is another whistle at moment t within window of hz?"""
    for o in others:
        if o.points[0, 0] - FRAME_S / 2 <= t <= o.points[-1, 0] + FRAME_S / 2:
            if abs(float(np.interp(t, o.points[:, 0], o.points[:, 1])) - hz) <= window:
                return True
    return False

def extend_one_end(w, end, evidence, others, settings):
    """Extend whistle w at its "start" or "end". others: the other whistles (it must not run into them).
    Returns (added points [n x 2] in time order, their sources, decision row)."""
    step = 1 if end == "end" else -1
    walk = w.points if end == "end" else w.points[::-1] # the whistle in walking order: tip last
    tip_t, tip_hz = walk[-1]
    length_ms = 1000 * np.ptp(w.points[:, 0])
    rule = settings.extend_rule
    n, m = settings.extend_observations_min_run, settings.extend_observations_run_window
    t_lo, t_hi = evidence.map_t[0], evidence.map_t[-1]

    # CREPE's last agreement: on the whistle's own points within map_carry_ms of the tip, or the tip itself
    # with extend_map_fallback
    last_crepe = tip_t if settings.extend_map_fallback else None
    if last_crepe is None:
        for t, hz in walk[::-1]: # from the tip inward
            if abs(t - tip_t) > settings.map_carry_ms / 1000 + FRAME_S / 4:
                break
            if crepe_agrees(evidence, t, hz, settings):
                last_crepe = t # the agreement nearest the tip
                break

    recent = walk.copy() # the points the direction is fitted on
    added, added_src = [], []
    last_t, last_hz = tip_t, tip_hz
    reason = ""
    k = 0
    while True:
        k += 1
        t = tip_t + step * k * FRAME_S
        grown_ms = 1000 * abs(t - tip_t)
        if grown_ms > settings.max_extension_ms + 1e-6:
            reason = "max_extension_ms"
            break
        if settings.max_whistle_ms and length_ms + grown_ms > settings.max_whistle_ms + 1e-6:
            reason = "max_whistle_ms"
            break
        if not t_lo - FRAME_S / 2 <= t <= t_hi + FRAME_S / 2:
            reason = "edge of the map"
            break
        if 1000 * abs(t - last_t) > settings.extend_stop_unsupported_ms + 1e-6:
            reason = f"no evidence for more than {settings.extend_stop_unsupported_ms:.0f} ms"
            break
        slope = direction_slope(recent, settings) # continue the straight line through the last fit_ms
        expected = last_hz + slope * (t - last_t)
        window = search_window_hz(slope, settings)
        if settings.extend_prediction == "free": # look as widely as the slope limit allows from the last point,
            centre = last_hz # but still prefer the evidence nearest the line
            window = settings.max_slope_hz_per_s * abs(t - last_t) + settings.freq_tolerance_hz
        else:
            centre = expected
        if settings.extend_stop_at_other and someone_else_here(t, expected, settings.window_min_hz, others):
            reason = "reached another whistle"
            break
        obs = [(hz, src) for hz, src in evidence.observations_at(t, n=n, m=m)
               if abs(hz - centre) <= window and step_allowed(last_hz, hz, abs(t - last_t), settings)]
        if settings.extend_prediction == "free" and obs:
            # evidence near the line wins, as with "line"; only if there is none does the extension turn, and then
            # to the evidence nearest the line and whatever else is there (so CREPE must agree at that same place)
            near = [(hz, src) for hz, src in obs if abs(hz - expected) <= search_window_hz(slope, settings)]
            if near:
                obs = near
            else:
                nearest = min(obs, key=lambda o: abs(o[0] - expected))[0]
                obs = [(hz, src) for hz, src in obs if abs(hz - nearest) <= settings.window_min_hz]
        crepe_here = [hz for hz, src in obs if src == "crepe"]
        if crepe_here:
            last_crepe = t
        if rule == "crepe_required":
            allowed = ("map", "harmonic", "crepe") if crepe_here else ()
        elif rule == "map_only":
            allowed = ("map", "harmonic")
        else: # map_carry
            carry = last_crepe is not None and 1000 * abs(t - last_crepe) <= settings.map_carry_ms + 1e-6
            allowed = ("map", "harmonic", "crepe") if (crepe_here or carry) else ()
        chosen = None
        for source in ("map", "harmonic", "crepe"): # map's frequency first, as Moment.best_hz
            here = [hz for hz, src in obs if src == source and source in allowed]
            if here:
                chosen = (min(here, key=lambda hz: abs(hz - expected)), source)
                break
        if chosen is None:
            continue
        hz, source = chosen
        if settings.max_sustained_slope_hz_per_s: # the same limit as join: no faster than real whistles, sustained
            window_s = settings.sustained_slope_window_ms / 1000
            tail = np.vstack([recent[np.abs(recent[:, 0] - t) <= window_s + FRAME_S / 4], [t, hz]])
            tail = tail[np.argsort(tail[:, 0])]
            if len(tail) > 1 and sustained_slope(tail, settings.sustained_slope_window_ms) > settings.max_sustained_slope_hz_per_s:
                reason = "faster than max_sustained_slope_hz_per_s"
                break
        added.append((t, hz))
        added_src.append(source + ("+crepe" if crepe_here and source != "crepe" else ""))
        last_t, last_hz = t, hz
        recent = np.vstack([recent, [t, hz]])

    row = dict(step="extend", fragments=" ".join(map(str, w.fragments)), end=end, added_ms=0.0,
               points_added=0, sources="", stopped=reason)
    if not added:
        return np.zeros((0, 2)), [], row
    # the route from the tip to the furthest accepted point, in time order, filled in between (fill_route returns
    # the moments strictly between its first and last point)
    route = np.vstack([[tip_t, tip_hz], np.array(added)])
    route_src = ["tip"] + added_src
    if end == "end":
        fill, fill_src = fill_route(route, route_src, w.points[-3:], route[-1:], settings)
        pts = np.vstack([fill, route[-1:]])
        src = fill_src + [route_src[-1]]
    else:
        route, route_src = route[::-1], route_src[::-1]
        fill, fill_src = fill_route(route, route_src, route[:1], w.points[:3], settings)
        pts = np.vstack([route[:1], fill])
        src = [route_src[0]] + fill_src
    src = [f"extend {s}" for s in src]
    row.update(added_ms=round(1000 * abs(added[-1][0] - tip_t), 1), points_added=len(added),
               sources=" ".join(sorted(set(added_src))))
    return pts, src, row


def extend_whistles(whistles, evidence, settings):
    """Extend every whistle at its start and its end (see the module notes). Returns (whistles, decisions).
    Whistles are extended one at a time, each seeing the others as they are by then, so two extensions
    cannot run into each other."""
    decisions = []
    if not settings.extend:
        return whistles, decisions
    whistles = list(whistles)
    for i in range(len(whistles)):
        for end in ("start", "end"):
            w = whistles[i]
            if settings.max_whistle_ms and 1000 * np.ptp(w.points[:, 0]) >= settings.max_whistle_ms:
                decisions.append(dict(step="extend", fragments=" ".join(map(str, w.fragments)), end=end,
                                      added_ms=0.0, points_added=0, sources="", stopped="already max_whistle_ms"))
                continue
            others = whistles[:i] + whistles[i + 1:]
            pts, src, row = extend_one_end(w, end, evidence, others, settings)
            decisions.append(row)
            if len(pts):
                if end == "end":
                    w = Whistle(np.vstack([w.points, pts]), list(w.sources) + src, w.fragments)
                else:
                    w = Whistle(np.vstack([pts, w.points]), src + list(w.sources), w.fragments)
                whistles[i] = w
    return whistles, decisions