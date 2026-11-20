"""Stage B of SAM-CREPE: repair SAM-Whistle's fragments, selection by selection, and write the results.

WHAT IT DOES
Reads the fragments get_fragments.py saved (stage A) and each selection's confidence map and CREPE scores, then
runs the steps below. Each step can be switched off in settings.py or with --set, which is how the comparison
runs R1 to R6 in the README are made.
    pass 1, every selection:
        clean (clean)                    remove steady tones and harmonics; note the tones' frequencies
    then, per recording:
        recurring tones (recording_tone_bands)   tones found in at least tone_min_selections selections of the
                                         same recording are blocked in all of them
    pass 2, every selection:
        evidence                         the map, CREPE, and the removed harmonics (use_harmonic_evidence),
                                         with the tone bands blocked (ignore_tone_bands)
        join (join)                      join fragments of the same whistle across gaps
        merge (merge_duplicates)         merge the same whistle traced twice side by side
        extend (extend)                  extend each whistle's start and end
        finalise (finalise)              protocol rules (tones again, the 100 ms minimum)
        write                            <stem>.bin and <stem>_provenance.csv
    at the end: decisions.csv (every decision of every step, with the reason) and settings.json

A selection whose fragments, conf_map.npz, crepe_act.npz or sample.json is missing is skipped, not an error:
it is printed at the start and listed in skipped.txt.

Selections are grouped into recordings by the part of their name before "_sel_" (<recording>_sel_026 is from
<recording>). A whole recording given as one file (no "_sel_" in its name) works too; it is then its own group,
so recurring tones can only come from itself.

INPUTS
The fragments from get_fragments.py (stage A), and for each selection the confidence map and CREPE scores in
the inputs folder (helpers/model_outputs.py describes its layout). By default the same inputs folder stage A
used (its inputs_used.txt); --inputs overrides it.

SAFETY
Reads SAM-CREPE/outputs/<fragments>/ and the inputs folder; writes only SAM-CREPE/outputs/<name>/, and refuses
if that folder exists unless --replace is given (store.output_dir). Never imports the analysts' traces.

Run with SAM-CREPE's environment, from the repo root, for example:
    SAM-CREPE\\.venv\\Scripts\\python SAM-CREPE\\run.py --name R6
    SAM-CREPE\\.venv\\Scripts\\python SAM-CREPE\\run.py --name R4 --set extend=false --set finalise=false
    SAM-CREPE\\.venv\\Scripts\\python SAM-CREPE\\run.py --name R6_no_merge --set merge_duplicates=false
Scoring against the analysts is separate (scripts/experiments/evaluate_contours.py; see the README).
"""
import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from settings import Settings  # noqa: E402
from store import OUTPUTS, load_fragments, output_dir  # noqa: E402
from evidence import Evidence  # noqa: E402
from clean import clean, recurring_tones, remove_tone_band_fragments, tone_frequencies  # noqa: E402
from join import Whistle, accept_joins, build_whistles, find_candidates, join_decisions  # noqa: E402
from merge import merge_duplicates  # noqa: E402
from extend import extend_whistles  # noqa: E402
from finalise import finalise, write_selection  # noqa: E402
from helpers.model_outputs import load_model_outputs, missing_files  # noqa: E402
from helpers.paths import DEFAULT_INPUTS  # noqa: E402

def recording_of(stem):
    """The recording a selection comes from: the part of its name before "_sel_" (the whole name if there is none)."""
    return stem.split("_sel_")[0]

def in_original_numbers(row, ids):
    """Decision rows from join, merge, extend and finalise number fragments by their position among the
    fragments clean kept; rewrite them as positions among all of the selection's fragments (as in the clean
    rows and in the fragments file), so every row in decisions.csv uses the same numbers."""
    row = dict(row)
    for key in ("fragment_a", "fragment_b"):
        if key in row:
            row[key] = ids[int(row[key])]
    for key in ("fragments", "fragments_a", "fragments_b"):
        if key in row and row[key] != "":
            row[key] = " ".join(str(ids[int(i)]) for i in str(row[key]).split())
    return row

def run_selection(stem, inputs, fragments, kept, removed, harmonics, tones, settings):
    """Pass 2 for one selection (see the module notes). Returns (whistles, decision rows)."""
    rows = []
    if settings.clean:                              # short flat pieces inside a tone band (known only now)
        kept, more = remove_tone_band_fragments(fragments, kept, tones, settings)
        removed = list(removed) + more
        rows += [dict(step="clean", fragment=r.index, decision=f"removed: {r.reason}", detail=r.detail, flat=r.flat)
                 for r in more]
    ids = [i for i in range(len(fragments)) if i not in {r.index for r in removed}]
    outputs = load_model_outputs(stem, inputs)
    evidence = Evidence(outputs, settings)
    evidence.add_harmonics(harmonics)
    evidence.add_tone_bands(tones)

    if settings.join:
        candidates = find_candidates(kept, evidence, settings)
        accepted, decision = accept_joins(candidates, settings)
        whistles = build_whistles(kept, accepted, settings)
        rows += join_decisions(candidates, decision)
    else:
        whistles = [Whistle(f.points, ["sam"] * len(f.points), [i]) for i, f in enumerate(kept)]
    whistles, merge_rows = merge_duplicates(whistles, kept, settings, outputs.conf)
    whistles, extend_rows = extend_whistles(whistles, evidence, settings)
    whistles, finalise_rows = finalise(whistles, settings)
    rows += merge_rows + extend_rows + finalise_rows
    rows = [in_original_numbers(r, ids) for r in rows]
    for w in whistles:
        w.fragments = [ids[i] for i in w.fragments]
    return whistles, rows

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", required=True, help="run name: the output folder SAM-CREPE/outputs/<name>")
    ap.add_argument("--fragments", help="stage A folder in SAM-CREPE/outputs (default: "
                                        "fragments_minlen<fragment_minlen_ms>_maxgap<fragment_maxgap_ms>)")
    ap.add_argument("--set", action="append", metavar="NAME=VALUE", help="override a setting in settings.py (repeatable)")
    ap.add_argument("--inputs", help="folder of model outputs, one subfolder per audio file (default: the one "
                                     "stage A used, from its inputs_used.txt, or else the development set)")
    ap.add_argument("--stems", nargs="*", help="only these selections (default: all in the fragments folder)")
    ap.add_argument("--replace", action="store_true", help="delete and redo an existing run of the same name")
    args = ap.parse_args()

    settings = Settings().with_overrides(args.set)
    source = OUTPUTS / (args.fragments or f"fragments_minlen{settings.fragment_minlen_ms}_maxgap{settings.fragment_maxgap_ms}")
    if not source.is_dir():
        sys.exit(f"{source} not found: run get_fragments.py with the same fragment settings first")
    if args.inputs:
        inputs = Path(args.inputs).resolve()
    elif (source / "inputs_used.txt").exists():
        inputs = Path((source / "inputs_used.txt").read_text().strip())
    else:
        inputs = DEFAULT_INPUTS # stage A folders made before inputs_used.txt existed
    print(f"fragments: {source}\ninputs:    {inputs}", flush=True)
    stems = args.stems or sorted(p.stem for p in source.glob("*.npz"))

    # skip (and report) any selection whose fragments or model outputs are missing, rather than failing part way
    skipped = {}
    for stem in stems:
        missing = [] if (source / f"{stem}.npz").exists() else [f"{source.name}/{stem}.npz"]
        missing += [f"{stem}/{name}" for name in missing_files(inputs / stem)]
        if missing:
            skipped[stem] = missing
            print(f"skipping {stem}: not found: {', '.join(missing)}", flush=True)
    stems = [s for s in stems if s not in skipped]
    if not stems:
        sys.exit("no selection has all its files: nothing to run")

    out = output_dir(args.name, replace=args.replace)
    settings.save(out / "settings.json")
    (out / "fragments_used.txt").write_text(str(source) + "\n")
    (out / "inputs_used.txt").write_text(str(inputs) + "\n")
    if skipped:
        (out / "skipped.txt").write_text("".join(f"{s}: {', '.join(m)}\n" for s, m in skipped.items()))
    print(f"output:    {out}\n{len(stems)} selections to run" + (f", {len(skipped)} skipped" if skipped else ""),
          flush=True)

    # pass 1: clean every selection, and note its tones
    print(f"\npass 1 of 2: clean" + ("" if settings.clean else " (switched off)"), flush=True)
    prepared, tones = {}, {}
    for i, stem in enumerate(stems, 1):
        fragments, offset_s = load_fragments(source / f"{stem}.npz")
        if settings.clean:
            evidence = Evidence(load_model_outputs(stem, inputs), settings) # for the lone-harmonic check
            kept, removed, harmonics = clean(fragments, settings, evidence)
        else:
            kept, removed, harmonics = list(fragments), [], []
        tones[stem] = tone_frequencies(fragments, removed)
        prepared[stem] = (fragments, offset_s, kept, removed, harmonics)
        print(f"  [{i}/{len(stems)}] {stem}: {len(fragments):3d} fragments, {len(removed):3d} removed", flush=True)

    # tones that recur across each recording
    shared = {}
    if settings.recording_tone_bands:
        print("\nrecurring tones, per recording", flush=True)
        for rec in sorted({recording_of(s) for s in stems}):
            shared[rec] = recurring_tones({s: tones[s] for s in stems if recording_of(s) == rec}, settings)
            print(f"  {rec}: recurring tones blocked in every selection: "
                  + (", ".join(f"{hz / 1000:.1f} kHz" for hz in shared[rec]) or "none"), flush=True)

    # pass 2: join, merge, extend, finalise, write
    print("\npass 2 of 2: join, merge, extend, finalise, write", flush=True)
    all_rows = []
    for i, stem in enumerate(stems, 1):
        fragments, offset_s, kept, removed, harmonics = prepared[stem]
        whistles, rows = run_selection(stem, inputs, fragments, kept, removed, harmonics,
                                       tones[stem] + shared.get(recording_of(stem), []), settings)
        write_selection(out, stem, whistles, offset_s)
        clean_rows = [dict(step="clean", fragment=r.index, decision=f"removed: {r.reason}", detail=r.detail,
                           flat=r.flat) for r in removed]
        all_rows += [dict(stem=stem, **r) for r in clean_rows + rows]
        n_joined = sum(1 for r in rows if r.get("step") == "join" and r.get("decision") == "joined")
        n_merged = sum(1 for r in rows if r.get("step") == "merge" and str(r.get("decision", "")).startswith("merged"))
        extended_ms = sum(r.get("added_ms", 0) for r in rows if r.get("step") == "extend")
        print(f"  [{i}/{len(stems)}] {stem}: {len(fragments):3d} fragments, {len(removed):3d} removed by clean, "
              f"{n_joined:2d} joins, {n_merged:2d} merges, {extended_ms:5.0f} ms added by extend; "
              f"{len(whistles):3d} whistles written", flush=True)

    columns = []
    for r in all_rows:
        columns += [k for k in r if k not in columns]
    with open(out / "decisions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nwritten: {out}" + (f"\n{len(skipped)} selections skipped (listed in skipped.txt)" if skipped else ""))

if __name__ == "__main__":
    main()
