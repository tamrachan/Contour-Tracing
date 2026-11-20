"""Step 3b of SAM-CREPE (optional): merge parallel duplicates. Combine two whistles that are the same whistle traced
twice, side by side.

WHERE THIS FITS IN THE PIPELINE
    A  get_fragments.py   SAM-Whistle's graph search cuts each confidence map into fragments
    B1 load               fragments, confidence map and CREPE scores, all on the recording's clock
    B2 clean              remove steady tones and harmonics (harmonics kept as evidence, tones blocked)
    B3 join               join fragments of the same whistle across gaps
    B3b merge             (this file) switched on and off with merge_duplicates
    B4 extend             extend each whistle's start and end
    B5 finalise           protocol rules; write the results

WHY
A whistle on SAM-Whistle's confidence map is a ridge 3 to 4 rows (375 to 500 Hz) thick. Its graph search can
trace such a ridge twice, about one row (125 Hz) apart, and the two traces then count as two pieces of the
same whistle. In the 33 selections, 57 to 81% of the breaks between pieces of a whistle were such parallel
traces, in published SAM-Whistle as well as in our runs, and every one lay on a single ridge of the map
(docs/join_research_check.md, in the project repository). Joining cannot fix this: it connects one fragment's end to another's start,
while these two run alongside each other.

THE RULE (all values in settings.py)
Two whistles are merged if
    - they overlap in time for more than merge_min_overlap_ms (10 ms; shorter overlaps are handled by join),
    - at merge_min_share (80%) or more of the moments they share, they are within merge_max_hz (350 Hz,
      SAM-Whistle's matching tolerance) of each other, and
    - (merge_same_graph_only) at least one of their fragments came from the same SAM-Whistle graph, as all
      17 parallel pairs in the 33 selections did, so a second dolphin's whistle passing close by is not merged.
The two are compared every 2 ms across the stretch they both cover (SAM-Whistle's traces do not have a point at
every moment). Across that stretch the merged whistle keeps the points of ONE of the two (taking points from
both would zigzag between two traces about 125 Hz apart), chosen by merge_keep:
    - "map": the trace that sits higher on the map's ridge, i.e. whose average confidence score along the
      stretch is higher (the trace closer to the centre of the whistle);
    - "longer": the longer whistle (the more complete trace).
Before and after the stretch, each whistle keeps its own points. Merging repeats until no pair qualifies, so
three traces of one whistle become one.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from settings import FRAME_S # noqa: E402
from join import Whistle # noqa: E402

def overlap_span(p, q):
    """The stretch of time both whistles cover: (start, end) in seconds; end <= start if they do not overlap."""
    return max(p.points[0, 0], q.points[0, 0]), min(p.points[-1, 0], q.points[-1, 0])

def compare(p, q, fragments, settings):
    """How alike whistles p and q are where they overlap. Returns (overlap_ms, share within merge_max_hz,
    same_graph).

    SAM-Whistle's traces do not have a point at every 2 ms moment, so two parallel traces may share only a few
    exact moments. Both are therefore read every 2 ms across the stretch they both cover, along straight lines
    between their own points, and compared there."""
    start, end = overlap_span(p, q)
    if end <= start:
        return 0.0, 0.0, False
    grid = np.arange(start, end + FRAME_S / 2, FRAME_S)
    fp = np.interp(grid, p.points[:, 0], p.points[:, 1])
    fq = np.interp(grid, q.points[:, 0], q.points[:, 1])
    close = float(np.mean(np.abs(fp - fq) <= settings.merge_max_hz))
    same_graph = bool({fragments[i].graph_id for i in p.fragments} & {fragments[i].graph_id for i in q.fragments})
    return 1000 * (end - start), close, same_graph

def ridge_score(w, start, end, conf):
    """How high whistle w sits on the map's ridge between start and end: its average confidence score there.
    w is read every 2 ms along straight lines between its own points, and at each moment the score is taken
    from the map's nearest moment and nearest frequency row (125 Hz apart)."""
    grid = np.arange(start, end + FRAME_S / 2, FRAME_S)
    hz = np.interp(grid, w.points[:, 0], w.points[:, 1])
    moment = np.clip(np.rint((grid - conf.t[0]) / FRAME_S).astype(int), 0, len(conf.t) - 1)
    row = np.clip(np.searchsorted(conf.f, hz), 1, len(conf.f) - 1)
    row = np.where(np.abs(conf.f[row - 1] - hz) <= np.abs(conf.f[row] - hz), row - 1, row) # nearest row
    return float(np.mean(conf.values[moment, row]))

def combine(p, q, conf, settings):
    """One whistle from p and q. Across the stretch both cover, the points of one of them are kept, chosen by
    merge_keep (see the module notes); before and after it, the points of whichever whistle is there.
    Returns (whistle, which trace was kept, the two ridge scores)."""
    start, end = overlap_span(p, q)
    scores = (ridge_score(p, start, end, conf), ridge_score(q, start, end, conf))
    if settings.merge_keep == "map":
        keep_p = scores[0] >= scores[1]
    else: # "longer"
        keep_p = np.ptp(p.points[:, 0]) >= np.ptp(q.points[:, 0])
    kept, other = (p, q) if keep_p else (q, p)
    outside = (other.points[:, 0] < start - FRAME_S / 2) | (other.points[:, 0] > end + FRAME_S / 2)
    points = np.vstack([kept.points, other.points[outside]])
    sources = list(kept.sources) + [s for s, o in zip(other.sources, outside) if o]
    order = np.argsort(points[:, 0], kind="stable")
    whistle = Whistle(points[order], [sources[i] for i in order], sorted(set(p.fragments) | set(q.fragments)))
    return whistle, ("a" if keep_p else "b"), scores

def merge_duplicates(whistles, fragments, settings, conf):
    """Merge parallel duplicates (see the module notes). conf is the selection's confidence map
    (ModelOutputs.conf: scores, times and frequencies), used to choose which trace to keep. Returns
    (whistles, decisions): decisions is one row per pair of overlapping whistles that was compared, for
    decisions.csv."""
    decisions = []
    if not settings.merge_duplicates or len(whistles) < 2:
        return whistles, decisions
    whistles = list(whistles)
    while True:
        best = None
        for i in range(len(whistles)):
            for j in range(i + 1, len(whistles)):
                p, q = whistles[i], whistles[j]
                start, end = overlap_span(p, q)
                if 1000 * (end - start) <= settings.merge_min_overlap_ms:
                    continue # they do not overlap long enough to compare
                overlap_ms, share, same_graph = compare(p, q, fragments, settings)
                ok = (overlap_ms > settings.merge_min_overlap_ms and share >= settings.merge_min_share
                      and (same_graph or not settings.merge_same_graph_only))
                if ok and (best is None or overlap_ms > best[0]):
                    best = (overlap_ms, i, j, share, same_graph)
                if not ok:
                    reason = ("overlap too short" if overlap_ms <= settings.merge_min_overlap_ms
                              else f"within {settings.merge_max_hz:.0f} Hz at only {share:.0%} of the overlap"
                              if share < settings.merge_min_share else "different SAM-Whistle graphs")
                    decisions.append(dict(step="merge", fragments_a=" ".join(map(str, p.fragments)),
                                          fragments_b=" ".join(map(str, q.fragments)), overlap_ms=round(overlap_ms),
                                          share_within=round(share, 3), same_graph=same_graph,
                                          decision=f"not merged: {reason}"))
        if best is None:
            break
        overlap_ms, i, j, share, same_graph = best
        p, q = whistles[i], whistles[j]
        merged, kept, (score_a, score_b) = combine(p, q, conf, settings)
        decisions.append(dict(step="merge", fragments_a=" ".join(map(str, p.fragments)),
                              fragments_b=" ".join(map(str, q.fragments)), overlap_ms=round(overlap_ms),
                              share_within=round(share, 3), same_graph=same_graph,
                              ridge_score_a=round(score_a, 3), ridge_score_b=round(score_b, 3),
                              decision=f"merged, kept {kept} across the overlap"))
        whistles = [w for k, w in enumerate(whistles) if k not in (i, j)] + [merged]
        decisions = [d for d in decisions if d["decision"].startswith("merged")] # pairs are compared again
    return sorted(whistles, key=lambda w: w.points[0, 0]), decisions
