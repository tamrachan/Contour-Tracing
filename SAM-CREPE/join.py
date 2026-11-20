"""Step 3 of SAM-CREPE: join. Decide which fragments are pieces of the same whistle, and join them across the gaps
between them.
    Part 1  find_candidates: find and score every candidate pair (the route search below)
    Part 2  accept_joins, build_whistles: accept candidates by evidence, then cost, and build the joined whistles

WHERE THIS FITS IN THE PIPELINE
    A  get_fragments.py   SAM-Whistle's graph search cuts each confidence map into fragments
    B1 load               fragments, confidence map and CREPE scores, all on the recording's clock
    B2 clean              remove steady tones and harmonics (harmonics kept as evidence, tones blocked)
    B3 join               (this file)
    B4 extend             extend each whistle's start and end (uses evidence.py)
    B5 finalise           protocol rules; write the results

THE IDEA: A ROUTE THROUGH THE OBSERVATIONS
Experiment 3 showed that SAM-Whistle's graph search breaks whistles into pieces and that changing its
settings cannot put them back together; experiment 2 showed that the evidence (mostly the confidence map)
still sees the whistle in most gaps. So for a fragment A and a later fragment B, we look for a route through
the gap from A's last point to B's first point.

The observations are every piece of evidence inside the gap (evidence.observations_at): each map peak, CREPE's
top pick and each fundamental implied by a removed harmonic, at every 2 ms moment, minus anything on a steady
tone. A route goes from A's end to B's start stepping from observation to observation, always forward in time. A first
version that checked evidence along a straight line between the two ends found 1 of the 9 real joins in the
33 selections, because whistles often turn around inside a gap; a route through the observations follows the
whistle wherever it goes.

RULES FOR A ROUTE (all values in settings.py)
    skip     Between two observations the route may skip at most max_unsupported_ms (20 ms) with no observation: the
             protocol counts a silence shorter than that as part of the same whistle.
    slope    From one point to the next the frequency may change by at most max_slope_hz_per_s times the time
             between them (550 kHz/s; the analysts' traces never change faster), plus freq_tolerance_hz.
    octave   Two adjacent points may not differ by a factor of max_jump_ratio (2, an octave) or more: a whistle
             never suddenly jumps onto a harmonic. A and B themselves may be far apart in frequency, if the
             whistle sweeps there smoothly.
    cost     Of all allowed routes, the cheapest is chosen: skip_cost for every moment skipped, plus
             travel_cost_per_khz for every kHz the route moves up or down. So a smooth route that follows the
             whistle wins over a detour, and a route with observations wins over one with long skips.
If no allowed route exists, A and B are not joined (that is acceptable: we only join where there is evidence).

TWO SPECIAL CASES
    overlap  B starts up to max_join_overlap_ms (10 ms, protocol) before A ends: they are joined if B starts
             within window_min_hz of where A is at that moment (two traces of the same whistle).
    step     (allow_steps, off by default) A direct jump from A's end to B's start within step_max_gap_ms
             (20 ms, protocol), allowed to break the slope rule but never the octave rule. Off because every step
             candidate in the 33 selections joined two different sounds.

WHAT COUNTS AS AN OBSERVATION (evidence.py, all switchable in settings.py)
    Steady tones are not observations: the tones clean.py found in this selection, flat harmonic pieces, and tones
    that recur in at least tone_min_selections selections of the same recording are blocked
    (recording_tone_bands). An observation must also be part of a run of observations_min_run moments: a single dot cannot
    carry a route.

Every pair whose gap is within max_join_gap_ms (200 ms) is recorded, joined or not, with the reason, for
decisions.csv. Part 2 chooses between competing candidates in the order of rank_key: by evidence (the share of the
gap with observations on the route), then by cost.
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from settings import FRAME_S # noqa: E402

@dataclass
class Candidate:
    """A possible join of fragment a (earlier) to fragment b (later), and how it was judged."""
    a: int # index of the earlier fragment
    b: int # index of the later fragment
    gap_ms: float # time from a's last point to b's first point (negative = they overlap)
    kind: str = "" # "path", "overlap" or "step" if it is a candidate; "" if rejected
    path: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)), repr=False) # route points (time s, Hz), A's end to B's start
    sources: list = field(default_factory=list, repr=False) # the source of each route point ("A", "map", "crepe", "harmonic", "B")
    skipped: int = 0 # moments in the gap with no observation on the route
    longest_skip_ms: float = 0.0
    support: float = np.nan # share of gap moments the route has an observation at
    cost: float = np.nan # the route's cost (see the module notes)
    cost_per_moment: float = np.nan # cost divided by the gap's length in moments: used to rank candidates in part 2
    same_crossing: bool = False # a ended and b started at the same junction node of SAM-Whistle's graph
    rejected: str = "" # why it is not a candidate ("" if it is one)

def step_allowed(f_from, f_to, dt, settings, check_slope=True):
    """Can a whistle go from f_from to f_to in dt seconds? The octave rule always applies; the slope rule unless
    check_slope is False (steps)."""
    if max(f_to / f_from, f_from / f_to) >= settings.max_jump_ratio:
        return False
    return not check_slope or abs(f_to - f_from) <= settings.max_slope_hz_per_s * dt + settings.freq_tolerance_hz

def best_route(a_t, a_hz, b_t, b_hz, evidence, settings):
    """The cheapest allowed route from (a_t, a_hz) to (b_t, b_hz) through the observations in between.

    Works forward one moment at a time (dynamic programming): for every observation, remember the cheapest way to
    reach it from A's end and which observation it came from; at the end, follow those links back from B's start.
    Returns (points [n x 2], sources, skipped moments, longest skip in moments, cost), or None if no route."""
    n_steps = int(round((b_t - a_t) / FRAME_S)) # B's start is n_steps moments after A's end
    max_skip = int(round(settings.max_unsupported_ms / 1000 / FRAME_S))
    # nodes by moment: moment 0 is A's end, moment n_steps is B's start, the observations are in between
    layers = [[(a_hz, "A")]]
    for k in range(1, n_steps):
        layers.append(evidence.observations_at(a_t + k * FRAME_S, use_crepe=settings.join_use_crepe))
    layers.append([(b_hz, "B")])
    best = [[(np.inf, 0, None)] * len(layer) for layer in layers] # (cost, skipped, link to (k, index))
    best[0][0] = (0.0, 0, None)
    for k in range(1, n_steps + 1):
        for v, (f_v, _) in enumerate(layers[k]):
            for back in range(1, max_skip + 2): # the previous observation is 1 to max_skip + 1 moments back
                k_u = k - back
                if k_u < 0:
                    break
                for u, (f_u, _) in enumerate(layers[k_u]):
                    cost_u, skipped_u, _ = best[k_u][u]
                    if cost_u == np.inf or not step_allowed(f_u, f_v, back * FRAME_S, settings):
                        continue
                    cost = (cost_u + settings.skip_cost * (back - 1)
                            + settings.travel_cost_per_khz * abs(f_v - f_u) / 1000)
                    if cost < best[k][v][0]:
                        best[k][v] = (cost, skipped_u + back - 1, (k_u, u))
    cost, skipped, link = best[n_steps][0]
    if cost == np.inf:
        return None
    points, sources, longest, here = [], [], 0, (n_steps, 0)
    while here is not None:
        k, v = here
        points.append((a_t + k * FRAME_S, layers[k][v][0]))
        sources.append(layers[k][v][1])
        prev = best[k][v][2]
        if prev is not None:
            longest = max(longest, k - prev[0] - 1)
        here = prev
    return np.array(points[::-1]), sources[::-1], skipped, longest, cost

def judge_pair(i, j, A, B, evidence, settings):
    """Decide whether fragment B (index j) could continue fragment A (index i)."""
    a_t, a_hz = A.points[-1]
    b_t, b_hz = B.points[0]
    c = Candidate(i, j, round(1000 * (b_t - a_t), 1))
    c.same_crossing = bool(A.end_junction and B.start_junction and np.allclose(A.end_node, B.start_node))
    bonus = settings.same_graph_bonus if c.same_crossing else 0.0

    if b_t <= a_t: # overlap: is B where A is at that moment?
        a_there = float(np.interp(b_t, A.points[:, 0], A.points[:, 1]))
        if abs(a_there - b_hz) <= settings.window_min_hz:
            c.kind, c.cost, c.cost_per_moment, c.support = "overlap", -bonus, -bonus, 1.0
            c.path, c.sources = np.array([[b_t, b_hz]]), ["B"]
        else:
            c.rejected = f"overlaps A but starts {abs(a_there - b_hz):.0f} Hz away from it"
        return c

    n_steps = int(round((b_t - a_t) / FRAME_S))
    route = best_route(a_t, a_hz, b_t, b_hz, evidence, settings)
    too_steep = None
    if route is not None and settings.max_sustained_slope_hz_per_s:
        slope = sustained_slope(route[0], settings.sustained_slope_window_ms)
        if slope > settings.max_sustained_slope_hz_per_s:
            too_steep, route = slope, None # faster than real whistles go, for that long
    if route is not None:
        c.path, c.sources, c.skipped, longest, cost = route
        c.kind, c.longest_skip_ms = "path", longest * 1000 * FRAME_S
        c.support = 1 - c.skipped / max(1, n_steps - 1)
        c.cost = cost - bonus
        c.cost_per_moment = c.cost / n_steps
        return c
    if settings.allow_steps and c.gap_ms <= settings.step_max_gap_ms and step_allowed(a_hz, b_hz, 0, settings, check_slope=False):
        c.kind, c.skipped, c.longest_skip_ms, c.support = "step", n_steps - 1, c.gap_ms, 0.0
        c.path, c.sources = np.array([[a_t, a_hz], [b_t, b_hz]]), ["A", "B"]
        c.cost = settings.skip_cost * (n_steps - 1) - bonus
        c.cost_per_moment = c.cost / n_steps
        return c
    c.rejected = (f"the route climbs or falls {too_steep / 1000:.0f} kHz/s over {settings.sustained_slope_window_ms:.0f} ms "
                  f"(more than max_sustained_slope_hz_per_s)" if too_steep else "no allowed route through the evidence")
    return c

def sustained_slope(points, window_ms):
    """How steeply a route climbs or falls for a sustained stretch: the largest frequency change over any window of
    window_ms, divided by window_ms (Hz per second). The route is read every 2 ms along straight lines between its
    points. A route shorter than the window is judged by its whole change spread over the window, so a short steep
    step is allowed as long as it does not go further than a real whistle would in window_ms."""
    t, f = points[:, 0], points[:, 1]
    window_s = window_ms / 1000
    grid = np.arange(t[0], t[-1] + FRAME_S / 2, FRAME_S)
    fg = np.interp(grid, t, f)
    k = int(round(window_s / FRAME_S))
    if len(fg) <= k:
        return float(abs(fg[-1] - fg[0]) / window_s)
    return float(np.max(np.abs(fg[k:] - fg[:-k])) / window_s)

def rank_key(candidate, settings):
    """How candidates are ordered when they compete for the same fragment end (part 2 accepts them in this
    order). rank_by "support": the one whose route has observations at the largest share of the gap first, then the
    cheapest per moment. rank_by "cost": cheapest per moment first. "support" is the default because ranking by
    cost let short joins to other sounds win over longer, well-supported joins of the same whistle."""
    if settings.rank_by == "support":
        return (-candidate.support, candidate.cost_per_moment)
    return (candidate.cost_per_moment, -candidate.support)

def find_candidates(fragments, evidence, settings):
    """Every pair (A earlier, B later) whose gap is at most max_join_gap_ms and overlap at most
    max_join_overlap_ms, judged by judge_pair. Returns a list of Candidate, joined and rejected alike."""
    out = []
    for i, A in enumerate(fragments):
        for j, B in enumerate(fragments):
            if i == j or B.points[0, 0] <= A.points[0, 0]: # B must start after A starts
                continue
            gap_ms = 1000 * (B.points[0, 0] - A.points[-1, 0])
            if -settings.max_join_overlap_ms <= gap_ms <= settings.max_join_gap_ms:
                out.append(judge_pair(i, j, A, B, evidence, settings))
    return out

# ---------------------------------------------------------------------------------------------------------------
# Part 2: accept candidates and build the joined whistles
#
#   accept_joins     choose which candidates become joins: by evidence, then by cost (rank_key), each fragment end
#                    used once
#   build_whistles   follow the accepted joins from fragment to fragment, and build one whistle per chain:
#                    the fragments' own points, the route's observations in each gap, and the skipped moments filled
#                    in (fill_method). Every point records where it came from (its provenance).
#   join_decisions   one row per candidate for decisions.csv: what was judged, and what was decided
# ---------------------------------------------------------------------------------------------------------------
from scipy.interpolate import PchipInterpolator # noqa: E402

@dataclass
class Whistle:
    """One whistle after joining: one or more fragments, and the routes that join them."""
    points: np.ndarray # [n x 2] time s (the recording's clock), frequency Hz, one point per 2 ms moment, in time order
    sources: list # where each point came from:
                  #   "sam"           a point of one of SAM-Whistle's fragments
                  #   "map", "crepe", "harmonic"   an observation on a join route (evidence.py)
                  #   "interpolated"  a moment the route skipped, filled in by fill_method
    fragments: list # the indices of the fragments it is made of, in time order


def accept_joins(candidates, settings):
    """Choose which candidates become joins. Returns the accepted candidates, and for every candidate the
    decision in words ({(a, b): decision}).

    Candidates are taken in order of evidence, then cost (rank_key): first the one whose route has observations
    at the largest share of the gap, and between equals the one with the lower cost per moment. A candidate is
    accepted unless fragment a's end or fragment b's start has already been used by a candidate taken earlier:
    each end joins at most once, so where two routes compete for the same fragment (for example a whistle and
    another sound both able to continue it), the one with more evidence wins. Chains form by themselves: A to B and
    B to C make one whistle A-B-C. Because b always starts after a, a chain can never loop back on itself."""
    accepted, decision = [], {}
    end_used, start_used = {}, {}
    for c in sorted([c for c in candidates if c.kind], key=lambda c: rank_key(c, settings)):
        if c.a in end_used:
            decision[(c.a, c.b)] = f"not joined: fragment {c.a}'s end already joined to fragment {end_used[c.a]}"
        elif c.b in start_used:
            decision[(c.a, c.b)] = f"not joined: fragment {c.b}'s start already joined from fragment {start_used[c.b]}"
        else:
            accepted.append(c)
            end_used[c.a], start_used[c.b] = c.b, c.a
            decision[(c.a, c.b)] = "joined"
    for c in candidates:
        if not c.kind:
            decision[(c.a, c.b)] = f"not a candidate: {c.rejected}"
    return accepted, decision

def fill_route(route_points, route_sources, before, after, settings):
    """The route through one gap as one point per 2 ms moment: its observations where it has them, and the moments
    it skipped filled in. route_points runs from A's last point to B's first point (both excluded from the
    result, since they belong to the fragments). before / after: the last few points of A and the first few
    of B, so a "pchip" curve leaves A and arrives at B in their own directions.

    fill_method "linear": a straight line between the two observations either side of the skipped moments.
    fill_method "pchip": a smooth curve through all the route's observations and the nearby fragment points, that
    never overshoots them (PCHIP, piecewise cubic Hermite)."""
    t_route, f_route = route_points[:, 0], route_points[:, 1]
    n = int(round((t_route[-1] - t_route[0]) / FRAME_S))
    t_all = t_route[0] + FRAME_S * np.arange(1, n) # every moment strictly inside the gap
    have = {int(round((t - t_route[0]) / FRAME_S)): (f, s) for t, f, s in zip(t_route, f_route, route_sources)}
    if settings.fill_method == "pchip":
        tt = np.concatenate([before[:, 0], t_route, after[:, 0]])
        ff = np.concatenate([before[:, 1], f_route, after[:, 1]])
        tt, keep = np.unique(tt, return_index=True)
        curve = PchipInterpolator(tt, ff[keep])
    else:
        curve = lambda t: np.interp(t, t_route, f_route)
    points, sources = [], []
    for k, t in enumerate(t_all, start=1):
        if k in have:
            points.append((t, have[k][0]))
            sources.append(have[k][1])
        else:
            points.append((t, float(curve(t))))
            sources.append("interpolated")
    return np.array(points).reshape(-1, 2), sources

def build_whistles(fragments, accepted, settings):
    """Follow the accepted joins and build one Whistle per chain of fragments. Fragments that were not joined
    become whistles on their own.

    Gap ("path"): the fragment's points, then the route through the gap (fill_route), then the next fragment.
    Overlap: the earlier fragment's points up to the moment the later one starts, then the later one's points
    (the two are traces of the same sound, so nothing is averaged).
    Step: the two fragments' points with nothing in between: the whistle jumps."""
    link = {c.a: c for c in accepted} # fragment: the join that continues it
    has_before = {c.b for c in accepted}
    whistles = []
    for first in range(len(fragments)):
        if first in has_before:
            continue # not the start of a chain
        pts, src, members = [fragments[first].points], [["sam"] * len(fragments[first].points)], [first]
        k = first
        while k in link:
            c = link[k]
            A, B = fragments[c.a], fragments[c.b]
            if c.kind == "overlap":
                t_b = B.points[0, 0]
                keep = pts[-1][:, 0] < t_b - FRAME_S / 2 # the earlier trace stops where the later begins
                pts[-1], src[-1] = pts[-1][keep], [s for s, kp in zip(src[-1], keep) if kp]
            elif c.kind == "path":
                fill, fill_src = fill_route(c.path, c.sources, A.points[-3:], B.points[:3], settings)
                pts.append(fill)
                src.append(fill_src)
            pts.append(B.points)
            src.append(["sam"] * len(B.points))
            members.append(c.b)
            k = c.b
        points = np.vstack(pts)
        sources = [s for part in src for s in part]
        order = np.argsort(points[:, 0], kind="stable")
        # one point per moment: SAM-Whistle's graph search occasionally repeats a point (in the 33 selections,
        # once, in sel 033), so a repeated moment keeps its first point
        t_sorted = points[order, 0]
        new_moment = np.concatenate([[True], np.diff(t_sorted) > FRAME_S / 4])
        order = order[new_moment]
        whistles.append(Whistle(points[order], [sources[i] for i in order], members))
    return whistles

def join_decisions(candidates, decision):
    """One row per candidate, for decisions.csv."""
    rows = []
    for c in candidates:
        rows.append(dict(step="join", fragment_a=c.a, fragment_b=c.b, gap_ms=c.gap_ms, kind=c.kind or "-",
                         support=round(c.support, 3) if not np.isnan(c.support) else "",
                         skipped_moments=c.skipped, longest_skip_ms=c.longest_skip_ms,
                         cost_per_moment=round(c.cost_per_moment, 3) if not np.isnan(c.cost_per_moment) else "",
                         same_crossing=c.same_crossing, decision=decision[(c.a, c.b)]))
    return rows