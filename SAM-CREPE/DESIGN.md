# SAM-CREPE: design

This document describes how SAM-CREPE works and why it was built this way. It covers what each file does, how the steps connect, the rules each step applies, the checks made on the development data, and what is still open. For how to set up and run it, see `README.md`.

SAM-CREPE repairs SAM-Whistle's fragmented whistle contours. SAM-Whistle's own confidence map is the main evidence and CREPE's pitch scores are a second source.

## Files

Everything runs from this folder: its own environment (`.venv`, from `requirements.txt`), its own copies of SAM-Whistle's code (`external/SAM-Whistle/`) and of CREPE with its bottlenose model (`external/CREPE/`), and the Colab notebook that makes SAM-Whistle's maps (`colab/`). From outside it needs only Java (stage A) and Google Colab (step 1, new audio only). Scoring against the analysts is deliberately kept separate, in `scripts/experiments/`. That folder, like the other `scripts/`, `results/` and `docs/` paths named below, is in the project repository where the experiments were run, not in this one.

| File | What it does |
|---|---|
| `prepare_inputs.py` | Step 1: cuts the audio into windows (`cut`), runs CREPE on them (`crepe`), and adds the maps made on Colab (`collect`) |
| `colab/create_sam_whistle_maps.ipynb` | Step 1: the Colab notebook that makes SAM-Whistle's confidence maps on a GPU |
| `get_fragments.py` | Stage A: runs SAM-Whistle's own graph search on the saved confidence maps and keeps every fragment, its graph, and whether each end was cut at a crossing (needs Java) |
| `run.py` | Stage B: runs the repair steps below (clean, join, merge, extend, finalise) on every selection and writes the run folder. Works on a whole recording or on windows cut from one; recurring tone bands are collected per recording in both cases |
| `evidence.py` | The question join and extend ask at every moment: is the whistle near the expected frequency? The map, CREPE and removed harmonics each vote |
| `clean.py` | B2: removes steady tones and harmonics, and passes the harmonics to `evidence.py` as evidence of their fundamental |
| `join.py` | B3: finds routes through the evidence between fragment ends (part 1), accepts them by evidence, then cost, and builds whistles (part 2) |
| `merge.py` | B3b (optional, `merge_duplicates`): merges parallel duplicates, the same whistle traced twice side by side |
| `extend.py` | B4: extends each whistle's start and end while the evidence (by default CREPE) supports it |
| `finalise.py` | B5: the protocol's rules (steady tones again, longer than 100 ms), and writes each selection's `.bin` (SAM-Whistle's own format and writer) and `_provenance.csv` |
| `figures.py` | One PNG per selection with four panels: analysts, SAM-Whistle, CREPE, SAM-CREPE |
| `settings.py` | Every setting, with where its default comes from (protocol, SAM-Whistle, measured, decided, or a guess) |
| `store.py` | Where SAM-CREPE writes, and how fragments are saved and loaded |
| `helpers/` | `paths.py` (every location), `model_outputs.py` (reads an inputs folder; describes its layout), `silbido_io.py` (reads `.bin` files), `graph_search.py` (SAM-Whistle's graph search on a saved map, from experiments 1 and 3), `references.py` (the analysts' traces, read only by `figures.py`) |
| `external/SAM-Whistle/` | An unmodified copy of SAM-Whistle's code; see its `PROVENANCE.md`. Its model weights are downloaded by the Colab notebook |
| `external/CREPE/` | CREPE's `predict.py` (the team's fork, with `--save_activations`) and the bottlenose model; see its `PROVENANCE.md` |
| `requirements.txt` | The environment for every step: the exact versions that produced all results so far |
| `recordings/` | Where `prepare_inputs.py cut` looks for recordings: each `.wav` with its Raven selection table (now `T08_1.WAV` and `MasterT08_1.selections.txt`). Only the folder is in git |
| `inputs_<name>/` | Model outputs made by `prepare_inputs.py`, one folder per window. `inputs_sars_T08/` (all 335 selections of T08_1) is the default inputs folder (`helpers/paths.py`). Not in git |
| `references/` | The analysts' traces (one `analyst_<name>.csv` per analyst and a `reference.json`) for the 33 development selections, for the figures only. Not in git |
| `baselines/` | Experiment 3's graph search runs (`minlen<M>_maxgap<G>/`, `.bin` per selection), which `get_fragments.py` compares its fragments against; the check is skipped if the folder is missing. Not in git |
| `outputs/` | One folder per run (and per set of fragments and figures); never overwritten without `--replace`. Not in git |

torch is installed (CPU build) for CREPE and because SAM-Whistle's graph search module imports its model code. SAM-Whistle's model itself runs on Colab, because it needs a GPU. Python 3.12 or later is needed (tested with 3.13).

## How the steps connect

Each step reads what the step before it wrote, and records where it read from (`inputs_used.txt`, `fragments_used.txt`), so every result can be traced back to its inputs.

```
Step 1  prepare_inputs.py cut, crepe; colab/create_sam_whistle_maps.ipynb; prepare_inputs.py collect
inputs_<name>/<selection>/                  conf_map.npz, crepe_act.npz, sam_input.wav, sample.json
    |
Step 2  get_fragments.py (stage A)
outputs/fragments_minlen<M>_maxgap<G>/      <selection>.npz, fragments.csv, settings.json, inputs_used.txt
    |
Step 3  run.py --name <run> (stage B: clean, join, merge, extend, finalise)
outputs/<run>/                              <selection>.bin, <selection>_provenance.csv, decisions.csv, settings.json,
    |                                       fragments_used.txt, inputs_used.txt
Step 4  figures.py (also reads references/<selection>/, the analysts' traces)
outputs/figures_<sam-run>_<run>/            <selection>.png

Scoring (optional, outside SAM-CREPE): scripts/experiments/evaluate_contours.py, compare_runs.py, plot_runs.py
```

```
         SAM-Whistle (saved maps)                         CREPE (saved scores)
                  |                                               |
   A. FRAGMENTS (get_fragments.py)                                |
      graph search, minlen 50, maxgap 50                          |
      keep every piece, its graph, and whether it was cut         |
      at a crossing                                               |
                  |  outputs/fragments_.../<stem>.npz             |
                  v                                               v
   B. REPAIR (run.py)
      B1 load       fragments, map and CREPE on one clock
      B2 clean      drop steady tones and harmonics; keep harmonics as evidence
      B3 join       pair fragment ends; is there evidence between them?
      B3b merge     (optional) merge the same whistle traced twice side by side
      B4 extend     walk outward from each end while evidence continues
      B5 finalise   protocol rules, write .bin, provenance and decisions
                  |  outputs/<run>/<stem>.bin
                  v
   C. SCORE (scripts/experiments/evaluate_contours.py)   (the only place analysts are used)
```

## Step 1: what an inputs folder holds

Each selection's subfolder holds the files SAM-CREPE reads (layout in `helpers/model_outputs.py`):

- `conf_map.npz`: SAM-Whistle's confidence map, made on Colab (GPU);
- `crepe_act.npz`: CREPE's scores for every pitch bin every 2 ms, from `predict.py --save_activations` (and `crepe_f0.csv`, its pitch track);
- `sam_input.wav`: the audio SAM-Whistle ran on (stage A takes its spectrogram settings from it);
- `crepe_input.wav`: the same window at the recording's own sample rate, for CREPE;
- `sample.json`: where the window sits in the recording, and the time offsets that put every file on the recording's clock;
- `sam_detections.bin` (optional, only in the old development set): SAM-Whistle's own detections, for comparison.

The inputs folder itself also holds `index.csv` (one row per window) and `settings.json` (the step 1 settings used).

How `cut` makes the windows: each window is centred on its selection and at least `window_min_s` (3 s) long, with `window_margin_s` either side of longer selections, and is moved inside the recording if it would run past its start or end. `sam_input.wav` is resampled to 192 kHz (what SAM-Whistle's model was trained on) and is exactly a whole number of SAM-Whistle's frames long (8 ms frames every 2 ms). SAM-Whistle labels a frame by its start, so its map and detections are offset by half a frame (4 ms) more than the audio; CREPE labels its frames by their centre.

The design checks below were made on an earlier development set, `inputs/`: the 33 selections the analysts traced, copied byte for byte (checked 2026-11-13) from `results/test_inputs/sars/` in the project repository, which `scripts/data_prep/make_step3_inputs.py` builds there. That folder is not in this checkout; `inputs_sars_T08/`, cut by `prepare_inputs.py` from the whole of T08_1, replaces it as the default and contains windows for the same 33 selections. The analysts' traces are deliberately never copied into an inputs folder, so the method's inputs never contain the answers.

## Stage B, step by step

### The one question every step asks (`evidence.py`)

```
look_for_whistle(t, expected_hz, window_hz):
    map       = the map peak (found as SAM-Whistle finds them, clicks skipped) within expected_hz +- window_hz,
                the one nearest expected_hz if there are several (peak_choice)
    crepe     = CREPE's top pick, if its score is at least crepe_threshold and it is within the window
    harmonic  = a removed harmonic's frequency divided by its multiple, if within the window
    verdict   = the sources that found it, e.g. "map+crepe", or "none"
```

Each source votes yes or no at its own threshold; raw scores are never mixed. Below 5 kHz the map has no rows, so only CREPE (and a harmonic above 5 kHz) can vote. The window is the larger of `window_min_hz` and the expected slope times `window_slope_frames` frames.

### B2: clean (`clean.py`)

Joining looks for fragments that line up in time and frequency. Steady tones (equipment or vessel noise) and harmonics would line up for the wrong reason, so they are removed first. A removed harmonic still shows that its whistle is there, so it is kept as evidence.

```
for each fragment F:
    if F lasts more than tone_min_ms and its frequency changes less than tone_max_range_hz:
        remove F (steady tone)
for each remaining fragment F:
    if another remaining fragment G overlaps F for at least harmonic_min_overlap_ms
       and F's frequency is within harmonic_tolerance of a whole multiple (up to harmonic_max_multiple) of G's there:
        remove F (harmonic of G); keep it as harmonic evidence
for each remaining fragment F (check_lone_harmonics):
    if the map or CREPE finds a fundamental at 1/2, 1/3, ... (up to 1/harmonic_max_multiple) of F's frequency at lone_harmonic_min_share of its moments:
        remove F (harmonic of a fundamental SAM-Whistle missed); keep it as harmonic evidence
hand all removed harmonics to evidence.py (use_harmonic_evidence)
once the tone bands are known (run.py, second pass; remove_tone_band_fragments):
    remove kept fragments that sit flat (within tone_max_range_hz) inside a tone band: short pieces of a tone
```

### B3: join (`join.py`)

Revised on 2026-11-12. A first version checked evidence along a straight line between the two fragments and found only 1 of the 9 real joins, because whistles often turn around inside a gap. It now searches for a route through the evidence. The observations are every piece of evidence in the gap (`evidence.observations_at`): map peaks, CREPE's top pick and fundamentals implied by removed harmonics, minus anything on a steady tone. Blocked as tones: the selection's own tone fragments, flat harmonic pieces (`flat_harmonics_as_tones`), and tones that recur in at least `tone_min_selections` selections of the recording (`recording_tone_bands`; in T08_1: 19.9, 20.9, 27.6 and 41.0 kHz). A harmonic that itself lies on a tone gives no evidence (a 20.8 kHz tone piece put false evidence at 10.4 kHz). An observation must also be part of a run: n (`observations_min_run`) of m (`observations_run_window`) moments on the same track. Competing joins are ranked by evidence first (`rank_by`). Diagnostic figures in `outputs/diagnostics/`: `join_stepping_stones.png` (first route search), `_v2.png` (smooth routes, tones blocked) and `_v3.png` (current).

```
for every pair (A, B) where B starts after A ends, within max_join_gap_ms (or overlapping by at most max_join_overlap_ms):
    if they overlap: join if B starts within window_min_hz of where A is at that moment
    else: find the cheapest route from A's last point to B's first point, stepping from observation to observation:
        between two observations the route may skip at most max_unsupported_ms (20 ms) with no observation
        between two adjacent route points the frequency may change by at most max_slope_hz_per_s
            times the time between them (plus freq_tolerance_hz)
        two adjacent route points may never differ by max_jump_ratio (an octave) or more
            (A and B themselves may differ by more, if the whistle sweeps there smoothly)
        the route may not climb or fall faster than max_sustained_slope_hz_per_s over sustained_slope_window_ms
            (320 kHz/s over 20 ms; real whistles reach 318 at their 99th percentile; a route up a click goes faster)
        cost = skip_cost per moment skipped + travel_cost_per_khz per kHz moved up or down
    no allowed route: not joined (acceptable; we only join where there is evidence)
    (allow_steps, off: a direct jump within step_max_gap_ms that may break the slope limit but not the octave rule)

accept candidates by evidence, then cost (rank_by = support: the largest share of the gap covered by observations
    first, then the cheapest per moment); each fragment end is used at most once
build whistles: chain the accepted joins; fill the moments skipped between observations (fill_method), marked "interpolated"
```

### B3b: merge parallel duplicates (`merge.py`, optional)

A whistle on the map is a ridge 3 to 4 rows thick, and the graph search can trace it twice about 125 Hz apart. In the 33 selections 57 to 81% of the breaks between pieces of a whistle were such parallel traces (`docs/join_research_check.md`). Join cannot fix this: it connects one fragment's end to another's start, while these two run alongside each other.

```
if merge_duplicates:
    repeat:
        for every pair of whistles that overlap in time for more than merge_min_overlap_ms:
            read both every 2 ms across the overlap
            qualifies if within merge_max_hz at merge_min_share of those moments or more,
                and (merge_same_graph_only) they share a SAM-Whistle graph
        no pair qualifies: stop
        merge the qualifying pair with the longest overlap:
            across the overlap keep one trace (merge_keep: "map" = higher average map score, or "longer")
            before and after it keep each whistle's own points
```

### B4: extend (`extend.py`)

SAM-Whistle misses the start of most whistles, and CREPE is the main evidence in those missed starts. Extending is also risky: the map stays on just after a whistle stops (its frames are 8 ms long), which is why CREPE is required by default.

```
from each whistle end, one 2 ms moment at a time:
    expected_hz = straight line through the last fit_ms of points, refitted after every accepted point
    observations within the window (extend's own n of m), passing the slope and octave rules
        extend_prediction = free (default): if there is no evidence near the line, look as widely as the slope
            limit allows and take the evidence nearest the line (follows the hook many whistles start with)
        no faster than max_sustained_slope_hz_per_s over sustained_slope_window_ms (as in join)
    extend_rule = crepe_required (default):  accept only moments where CREPE agrees
    extend_rule = map_carry:  accept CREPE; the map (or a harmonic) alone for up to map_carry_ms after CREPE last agreed
                              (extend_map_fallback, off: the tip counts as CREPE agreeing)
    extend_rule = map_only:   accept only the map (or a harmonic)
    stop after more than extend_stop_unsupported_ms without support, after max_extension_ms, at max_whistle_ms
        (0 = no limit; the protocol gives none), at the edge of the map, or where another whistle is within
        window_min_hz of the expected frequency (extend_stop_at_other)
add only accepted moments; fill the skipped ones (fill_method)
```

### B5: finalise (`finalise.py`)

```
apply the tone rule again to the finished whistles (finalise_recheck_tones)
drop whistles not longer than min_whistle_ms (the protocol: a whistle must be longer than 100 ms)
write outputs/<run>/<selection>.bin            SAM-Whistle's own format and time base (its write_binary.py)
write outputs/<run>/<selection>_provenance.csv  where each point came from: sam, map, crepe, harmonic, interpolated, extend
(run.py) write outputs/<run>/decisions.csv, settings.json, fragments_used.txt
```

The 100 ms minimum is applied here, at the end, so that short fragments can first be joined and extended into whistles.

**Two clocks:** the pipeline works on the recording's clock, so pieces of one long recording line up; the `.bin` is written on the clock of the audio file SAM-Whistle ran on. The difference (`offset_s`: where that file starts in the recording, plus half a frame) is saved with the fragments in stage A and taken off again here.

## Scoring against the analysts (outside SAM-CREPE)

Scoring is the only step that reads the analysts' traces for scores. The scorer lives in `scripts/experiments/` on purpose, so the method can never import the analysts' traces; it reads them, and the selections' time offsets, from `results/test_inputs/sars/`. SAM-CREPE's environment can run it. It uses the same scorer as experiment 3: coverage, pieces, missed start and end, overrun, other detections, contours per window, and a paired comparison of every whistle with the reference run.

```
SAM-CREPE\.venv\Scripts\python scripts\experiments\evaluate_contours.py --root SAM-CREPE\outputs --sets R0 R1 R2 R3 R4 R4m R5 R6 --reference R1 --out evaluation.csv
```

To compare every run with published SAM-Whistle instead, use `--reference R0` (and another `--out` name, so the first file is kept).

Check before reading the results: R0 must score like published SAM-Whistle (coverage 0.85, 76% of whistles in one piece, 1.26 pieces, 9.4 contours per window) and R1 like experiment 3's 100/50 (0.88, 58%, 1.53). If not, look at that run's `settings.json`: every step should be `false`.

**Results per selection.** `evaluate_contours.py` prints only the medians over the 33 selections. `scripts/experiments/compare_runs.py` shows one measure for every selection and run, marks each value better (+) or worse (-) than the reference run (default R0), and writes `per_selection_<measure>.csv` next to the evaluation file:

```
SAM-CREPE\.venv\Scripts\python scripts\experiments\compare_runs.py --csv SAM-CREPE\outputs\evaluation.csv
SAM-CREPE\.venv\Scripts\python scripts\experiments\compare_runs.py --csv SAM-CREPE\outputs\evaluation.csv --measure sam_pieces
```

Measures: `sam_point_coverage` (default), `sam_pieces`, `sam_missed_start_ms`, `sam_missed_end_ms`, `overrun_ms`, `off_frac`, `sam_dev_hz`, `sam_other_detections`, `window_contours`. Options: `--reference <run>`, `--runs R0 R1 R6` (which runs, in order).

**Figures comparing any runs, with their scores.** `scripts/experiments/plot_runs.py` makes one PNG per selection with a panel per run, drawn like `figures.py`, with each panel's title giving the selection's coverage, pieces and other detections from the evaluation file. A whistle that runs on past the analysts' end shows its overrun beyond the dashed line.

```
SAM-CREPE\.venv\Scripts\python scripts\experiments\plot_runs.py --runs R0 R1 R4m R6 --eval SAM-CREPE\outputs\evaluation.csv
```

This writes `outputs/figures_R0_R1_R4m_R6/<selection>.png`. Options: `--runs`, `--stems`, `--name`, `--replace`.

## Definitions used in the design checks

- **Mostly on an analyst's whistle:** at least half of a fragment's points lie inside the time span an analyst traced and within 350 Hz of that analyst's trace. 350 Hz is SAM-Whistle's own matching tolerance, also used by `scripts/comparison/make_comparison.py`.
- **Coverage of an analyst's whistle:** the share of the analyst's trace (on a 5 ms grid) that has a fragment point within 6 ms and 350 Hz, as in `make_comparison.py`.

## Design checks (development data: the 33 selections)

- **Stage A** (2026-11-11 to 2026-11-13): the fragments of all 33 selections equal experiment 3's contours exactly, at minlen 150 (the published setting), 100 and 50 (maxgap 50).
- **Running from this folder** (2026-11-13): with SAM-CREPE's own environment, copy of SAM-Whistle and `inputs/`, stage A still matches experiment 3 exactly at minlen 150, 100 and 50; `run.py` with every step off reproduces published SAM-Whistle (R0: 0.85, 76%, 1.26, 9.4 contours per window) and experiment 3's 100/50 (R1: 0.88, 58%, 1.53); the full pipeline (R6) scores exactly as before (0.90, 79%, 1.14).
- **Step 1** (2026-11-13): `cut` gives sample-for-sample the same SAM-Whistle audio as the development windows where they overlap, and CREPE in SAM-CREPE's environment reproduces the saved CREPE scores exactly.
- **evidence.py** (2026-11-12): every point of SAM-Whistle's own fragments (49,936 at minlen 50) gets a map vote at exactly its frequency, with the click check on or off. Every removed harmonic point gets a harmonic vote at its fundamental.
- **clean.py** (2026-11-12, minlen 50, harmonics up to 5x): of 878 fragments it removes 53 tones, 589 harmonics (15 of them flat, blocked as tone bands) and 5 lone harmonics, and keeps 231. None of the removed fragments is mostly on an analyst's whistle. Coverage of the analysts' whistles is 0.873 before and after; three traces lose under half a percentage point where a removed harmonic briefly crossed the whistle. Of the moments where a removed harmonic has no kept fragment at its fundamental, 5% are on an analyst's whistle, where the harmonic is the only sign of the whistle.
- **join.py** (2026-11-13, R4: clean and join with map and CREPE): coverage 0.88, 58% of whistles in one piece, 1.44 pieces per whistle, 0.27 other detections during a whistle, 6.1 contours per window. Of the 31 remaining breaks between pieces of a whistle, 25 are parallel duplicates and 6 are gaps with no evidence for more than 20 ms; none was a right join lost to a competing one.
- **merge.py** (2026-11-13, R4 plus merge, `merge_keep = map`): 37 merges; every fragment used exactly once; `merge_duplicates = false` changes nothing. Coverage 0.88, 79% in one piece, 1.14 pieces, 0.18 other detections, 5.0 contours per window. With `merge_keep = longer` coverage fell to 0.86. Judged against the analysts: 9 merges combined two traces of the same whistle, 23 traces of other sounds, 3 a whistle with a trace beside it, and 2 (sel 025, 026) two different analyst whistles. In those 5 the two ridge scores are almost equal (0.99 and 1.00), so `merge_keep = map` is close to a coin toss there.
- **Before extend** (2026-11-13, R4 plus merge): 44 of 57 analyst traces start more than 4 ms before our whistle (median 18 ms); ends are mostly right (median 0 ms). In the missed starts CREPE has evidence at 43% of the moments, the map at 29%; only 3 of 44 missed starts have map evidence and no CREPE. Past ends that are already right, the map fires at 28% of the first 10 ms, CREPE at 7%. Overruns of more than 20 ms past an analyst's end: 15 traces, 12 from SAM-Whistle's own fragments, 3 from join, none from merge.
- **extend.py** (2026-11-13, R4 plus merge plus extend): with `extend_rule = crepe_required` (default) coverage rises from 0.88 to 0.90 and the median missed start falls from 18 to 9 ms; 105 added points lie on an analyst's whistle, 3 just past its start or end, 46 elsewhere. `map_carry` gives the same coverage with 29 points past an end; `map_only` and `extend_map_fallback` add more points off the whistle for no gain. Figure `outputs/diagnostics/extend_off_points.png`: 44 of the 46 points "elsewhere" follow the map's ridge on steep starts and ends, and are counted off because of the few ms timing offset (see below) on a steep sweep. With `extend_stop_at_other` off, coverage is unchanged but 4 times as many points are added, mostly along parallel duplicates.
- **run.py and finalise.py** (2026-11-13): with every step switched off, `run.py` on the minlen 100 fragments writes `.bin` files identical to experiment 3's 100/50 contours in all 33 selections, so the format, the writer and the time base match SAM-Whistle's. Test runs `outputs/draft_R1` and `outputs/draft_R6` (all steps, default settings), scored with `evaluate_contours.py`: coverage 0.88 to 0.90 (9 whistles better, 21 the same, 3 worse; Wilcoxon p = 0.005), whistles in one piece 58% to 79%, pieces 1.53 to 1.14 (11 fewer, none more), median missed start 19 to 11 ms, other detections during a whistle 3.53 to 0.09, contours per window 15.3 to 4.2 (published SAM-Whistle: 0.85, 76%, 1.26, 9.4). Worse: sel 040 and 023 (a join carries the whistle 420 to 500 ms past the analysts' end), sel 002 (a join overruns by 42 ms), sel 005 (a merge kept the trace beside the whistle).
- **Why fragments start at minlen 50:** the protocol's 100 ms minimum applies to the finished whistle, so it is applied in finalise. Shorter pieces are kept so they can be joined. Compared with minlen 100, minlen 50 keeps 4 more short pieces on analysts' whistles after cleaning, and lets 15 short pieces of the steady tone through, which finalise catches by applying the tone rule again.

## Safety rules for everything in this folder

- **Reads** its own folders, plus the recording and selection table given to step 1. Never writes, moves or deletes anything outside `SAM-CREPE/`.
- **Writes** only to `SAM-CREPE/outputs/<name>/` (steps 2 to 4) and `SAM-CREPE/inputs_<name>/` (step 1). Each refuses to start if its folder already exists, unless given `--replace`, which deletes only that one folder. Nothing writes to `references/` or `baselines/`.
- **No analyst data in the method.** Steps 1 to 3 read only model outputs (`helpers/model_outputs.py`), never the analysts' traces. Only `figures.py` (step 4) reads `references/`, through `helpers/references.py`, to draw the traces. Scoring is done separately by `scripts/experiments/evaluate_contours.py`. The analysts are used only for design checks, which are reported above and in each file's header.
- **Nothing is hard-coded.** Every setting is in `settings.py`. Edit the default there, or override it for one run with `--set name=value`. Every run saves the settings it used in `settings.json`.

## To watch and to experiment with later

- **Tone rule on flat whistles.** A real whistle that holds one frequency (within 1 kHz) for more than 100 ms would be removed as a tone. None was in the 33 selections, but check this on the unseen recordings. Options if it happens: raise `tone_min_ms`, or also require a tone to sit at the same frequency across many selections.
- **`fragment_minlen_ms`:** 50 (default), 100, or SAM-Whistle's published 150.
- **`peak_choice`:** `nearest` (default) or `strongest`.
- **`skip_click_frames`:** on (default, as SAM-Whistle) or off.
- **`use_harmonic_evidence` and `check_lone_harmonics`:** on (default) or off, to measure what the harmonics add.
- **How much a harmonic's vote counts.** Now (decided 2026-11-12): the fundamental implied by a removed harmonic votes yes or no exactly like a map peak. Alternatives to test later: a weaker vote (for example half a supported moment in a join), or never allowing a harmonic alone to extend a whistle.
- **`harmonic_max_multiple`:** 5 (decided 2026-11-12). Checking up to 5 removed 45 more harmonics than up to 3, none on an analyst's whistle; above 5 almost nothing changes.
- **`allow_steps`:** off by default, because every step candidate in the 33 selections joined two different sounds. To test on other recordings.
- **Maximum whistle length.** In sel 040 a join carries the whistle past the point where the analysts stopped tracing. Whether that is one whistle or two depends on how long a whistle may be; the protocol only gives a minimum (100 ms). Decide with the team whether long whistles should be split, and if so at what length.
- **`ignore_tone_bands`:** on (default) or off. Blocking tones stops routes riding along them, but also stops joins of whistles that sit at a tone's frequency (sel 063, 065).
- **`fill_method`:** `linear` (default) or `pchip` (a smooth curve), for the moments skipped between two observations of a route.
- **`observations_min_run` (n) and `observations_run_window` (m):** an observation counts if at least n of the m moments in some window around it have an observation on the same track (within `observations_track_hz`, 500 Hz, per moment apart). Default n = m = 3 ("3 in a row"). Figure: `outputs/diagnostics/run_rule_v2.png`. With m = 5 the made-up sparse trail is kept and stray dots are still ignored, but sel 051's CREPE picks are about 5 moments (10 ms) apart, too sparse even for 3 of 5; m around 10 would be needed. On the 33 selections, m = 3 and m = 5 give the same joins.
- **A second round (later):** in gaps still unfilled, filter the audio to a band around the expected whistle (removing the tone) and rerun CREPE there ("guided CREPE"). First check that those gaps contain whistle energy at all.
- **`merge_duplicates`:** on (default) or off. Report results both ways, because the merge improves the count of pieces more than what is found.
- **`merge_max_hz`:** 350 (default) or 500 (the search window). At 350, 4 pairs of duplicates on analysts' whistles are not merged (sel 012, 023, 051, 056), the two traces being further apart for over 20% of their overlap.
- **`merge_keep`:** `map` (default) or `longer`.
- **Two whistles that touch (sel 025, 026).** Merging combined two different analyst whistles that run close together and share a SAM-Whistle graph. Possible check, not built: refuse a merge if the two traces are more than `merge_max_hz` apart for a long continuous stretch (two whistles that touch and separate), as opposed to a doubled trace that stays close.
- **A timing offset of a few ms (2026-11-13).** Shifting SAM-Whistle's points to best match the analysts' traces gives a best shift of 6 ms earlier (quartiles 5 to 8 ms, all 59 sweeping traces in the same direction). Against CREPE the shift is only 2 ms (one frame), so most of it lies between the analysts and both models, not in SAM-Whistle's time base (`+0.004 s` frame centre, `make_step3_inputs.py`). It accounts for about 6 ms of the 14 to 18 ms median missed start and for the 6 ms median end difference. To check how the analysts' traces are timed (Raven spectrogram window) before reporting start and end errors in ms.
- **Overruns past the analyst's end.** After join and merge, 15 analyst traces (11 whistles) have a whistle running more than 20 ms past their end: 12 come from SAM-Whistle's own fragments (21 to 42 ms, one 209 ms), 3 from join (sel 002, 023), none from merge. A trimming rule (for example, CREPE as the end cue) could go in finalise; to check first whether CREPE is absent in those tails but present at ends that are right.
- **`extend_stop_at_other` and crossing whistles.** Extend stops where another whistle is at the same moment within 500 Hz of where this one is expected (whistles may still overlap in time). On the 33 selections it stopped 40 times, all alongside parallel duplicates, including the 4 pairs merge did not merge (sel 012, 023, 051, 056); none at two different whistles crossing. On busier recordings it could cut a whistle's start at a crossing. If so: stop only if the other whistle stays within 500 Hz for several moments ahead (a duplicate stays alongside, a crossing passes through), or switch it off.
- **Join to a 20 to 21 kHz sound in sel 051.** After the 2026-11-13 changes (`docs/R_experiments.md`) one join remains from the whistle to a piece at 19.9 to 21.0 kHz. That piece varies by more than 1 kHz and has its own harmonic, so it is probably a real sound rather than the tone; the fall is within what real whistles do. The research idea "the direction at each fragment's tip as a cost" could separate it.
- **Whistles SAM-Whistle's graph search never forms (sel 008, about 2.3 to 2.7 s).** The map and CREPE both have evidence (63% and 62% of moments), SAM-Whistle's own peak picking finds the peaks, but its graph keeps no contour there, even with no minimum length (inside its Java code). SAM-CREPE starts from fragments, so it cannot recover such a whistle. A possible later step: start new whistles from long runs of map and CREPE evidence where there is no fragment.
- **Two whistles joined into one (sel 016).** One contour from 1.8 to 2.9 s, joined by a 6.6 kHz drop in 25 ms (263 kHz/s, within what real whistles do, with evidence every 20 ms or less). By the protocol it is one whistle; nobody traced it. To check with the team, with the maximum whistle length question.
- **`max_sustained_slope_hz_per_s`, `remove_tone_band_fragments`, `extend_prediction`:** added 2026-11-13 after reviewing the figures (`docs/R_experiments.md`). Compare on and off on the unseen data.
- **Ideas for join from the research check** (`docs/join_research_check.md`): the direction at each fragment's tip as a cost; an allowed frequency change that grows with the time since the last evidence; a shape check of each joined whistle; matching all fragment ends together (Hungarian algorithm).
- **`max_join_gap_ms`, `max_slope_hz_per_s`, `skip_cost`, `travel_cost_per_khz`, `extend_rule`, `map_carry_ms`:** tune now that join and extend are built.
