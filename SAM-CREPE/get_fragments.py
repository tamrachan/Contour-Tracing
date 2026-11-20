"""Stage A of SAM-CREPE: cut each selection's saved confidence map into fragments with SAM-Whistle's own graph
graph search, and record where each fragment was cut.

For each audio file in the inputs folder (--inputs; default DEFAULT_INPUTS in helpers/paths.py, now inputs_sars_T08):
  1. build SAM-Whistle's graph on conf_map.npz with fragment_minlen_ms and fragment_maxgap_ms (settings.py),
     using the same code as experiments 1 and 3 (helpers/graph_search.py, with the copy of SAM-Whistle in
     external/SAM-Whistle/);
  2. keep the same pieces get_tonals_safe() keeps, and for each also its graph id and whether its first and
     last nodes are junctions (where the graph branched at a crossing);
  3. convert times to the recording's clock (frame centres) and save <stem>.npz (store.save_fragments), with
     the offset that was added (offset_s).

Check: if experiment 3 has a run at the same settings (helpers/paths.py DEFAULT_REFERENCE, minlen<M>_maxgap<G>/),
every file's fragments must equal its contours exactly; the result is in fragments.csv. Skipped if not there.

Writes SAM-CREPE/outputs/<name>/ (default name: fragments_minlen<M>_maxgap<G>):
    <stem>.npz       the fragments
    fragments.csv    one row per file: fragments, graphs, junction ends, check against experiment 3
    settings.json    the settings used
    inputs_used.txt  the inputs folder

Run with SAM-CREPE's environment and Java, from the repo root:
    $env:JAVA_HOME = 'C:\\Program Files\\Java\\jdk-23'
    SAM-CREPE\\.venv\\Scripts\\python SAM-CREPE\\get_fragments.py [--set fragment_minlen_ms=50]
        [--inputs <folder>] [--name my_fragments] [--stems <name> ...] [--replace]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# This file's path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)) # so the SAM-CREPE modules below can be imported

from settings import Settings # noqa: E402
from store import Fragment, output_dir, save_fragments # noqa: E402
from helpers.paths import DEFAULT_INPUTS, DEFAULT_REFERENCE # noqa: E402
from helpers.model_outputs import file_offsets # noqa: E402
from helpers.silbido_io import read_silbido # noqa: E402
from helpers.graph_search import build_tracker, contours_match, kept_edges # noqa: E402

def node(edge, which):
    """An edge's first ("from") or last ("to") graph node. 
    JPype renames the Java field `from`, a Python keyword, to `from_`."""
    return getattr(edge, "from_" if which == "from" else "to")

def fragments_of(stem, inputs, settings):
    """The file's fragments, times in seconds from the start of the audio file SAM-Whistle ran on (frames
    labelled by their start, as SAM-Whistle writes them)."""
    conf = np.load(inputs / stem / "conf_map.npz")["conf"] # SAM-Whistle's confidence map (frequency x frame)
    # build SAM-Whistle's graph on the map, with the fragment settings
    tracker = build_tracker(stem, conf, inputs / stem / "sam_input.wav", settings.fragment_minlen_ms,
                            settings.fragment_maxgap_ms, threshold=settings.map_threshold)
    kept, _ = kept_edges(tracker) # the pieces get_tonals_safe() keeps: (graph id, edge, points)
    out = []
    for graph_id, edge, points in kept:
        a, b = node(edge, "from"), node(edge, "to") # the fragment's first and last graph nodes
        # whether each end is a junction, and each end node's (time, freq): where the graph was cut
        out.append(Fragment(points, graph_id, bool(a.is_junction()), bool(b.is_junction()),
                            (float(a.time), float(a.freq)), (float(b.time), float(b.freq))))
    return out

def shifted(fragments, offset):
    """The same fragments with every time moved by `offset` seconds."""
    return [Fragment(np.column_stack([f.points[:, 0] + offset, f.points[:, 1]]), f.graph_id, f.start_junction,
                     f.end_junction, (f.start_node[0] + offset, f.start_node[1]), (f.end_node[0] + offset, f.end_node[1]))
            for f in fragments]

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", action="append", metavar="NAME=VALUE", help="override a setting in settings.py (repeatable)")
    ap.add_argument("--inputs", default=str(DEFAULT_INPUTS), help="folder with one subfolder of model outputs per "
                                                                  "audio file (default: DEFAULT_INPUTS in helpers/paths.py)")
    ap.add_argument("--name", help="output folder name (default fragments_minlen<M>_maxgap<G>)")
    ap.add_argument("--stems", nargs="*", help="only these files (default: every subfolder of --inputs with a conf_map.npz)")
    ap.add_argument("--replace", action="store_true", help="delete and redo an existing run of the same name")
    args = ap.parse_args()

    settings = Settings().with_overrides(args.set) # the defaults, with any --set overrides
    inputs = Path(args.inputs).resolve()
    # every window with a confidence map, unless --stems
    stems = args.stems or sorted(p.parent.name for p in inputs.glob("*/conf_map.npz"))
    if not stems:
        sys.exit(f"no <name>/conf_map.npz found in {inputs}")
    tag = f"minlen{settings.fragment_minlen_ms}_maxgap{settings.fragment_maxgap_ms}" # names the run by its fragment settings
    out = output_dir(args.name or f"fragments_{tag}", replace=args.replace) # refuses an existing folder unless --replace
    # record the settings and inputs used, for the later steps
    settings.save(out / "settings.json")
    (out / "inputs_used.txt").write_text(str(inputs) + "\n")
    reference = DEFAULT_REFERENCE / tag # experiment 3's run at the same settings, if there is one

    rows = []
    for stem in stems:
        sam_time = fragments_of(stem, inputs, settings) # fragments on SAM-Whistle's clock
        offset_s = file_offsets(inputs / stem)["sam_detections.bin"] # SAM-Whistle's clock to the recording's
        save_fragments(out / f"{stem}.npz", shifted(sam_time, offset_s), offset_s) # saved on the recording's clock
        # check against experiment 3's contours for this file, if it has them ("" if not)
        match = ""
        if (reference / f"{stem}.bin").exists():
            match, _ = contours_match([f.points for f in sam_time], read_silbido(reference / f"{stem}.bin"))
        # this file's row of fragments.csv
        rows.append(dict(stem=stem, fragments=len(sam_time), graphs=len({f.graph_id for f in sam_time}),
                         start_at_junction=sum(f.start_junction for f in sam_time),
                         end_at_junction=sum(f.end_junction for f in sam_time), matches_experiment3=match))
        print(f"{stem}: {len(sam_time):3d} fragments in {rows[-1]['graphs']:3d} graphs; "
              f"{rows[-1]['start_at_junction']:3d} start / {rows[-1]['end_at_junction']:3d} end at a crossing"
              + (f"; matches experiment 3: {match}" if match != "" else ""))

    with open(out / "fragments.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # summary of the check, over the files experiment 3 had
    checked =[r["matches_experiment3"] for r in rows if r["matches_experiment3"] != ""]
    if checked:
        print(f"\n{sum(checked)} of {len(checked)} selections match experiment 3's {tag} contours exactly")
    print(f"written: {out}")

if __name__ == "__main__":
    main()
