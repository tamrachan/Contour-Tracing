# SAM-CREPE

This folder contains post-processing that **cleans**, **re-joins**, **merges** and **extends** SAM-Whistle's fragmented whistle contours. SAM-Whistle's own confidence map is the main evidence and CREPE's frequency bins are a second source.

The work has two stages: **stage A** (`get_fragments.py`, step 2) cuts SAM-Whistle's confidence maps into fragments, and **stage B** (`run.py`, step 3) repairs them. For how the method works and why (what each file does, the rules of every step, and the design checks), see [DESIGN.md](DESIGN.md).

## How to run

> **Note:** All commands should be run from the repo root in PowerShell.

### Set up (once)

Needs Python 3.12 or later (tested with 3.13) and, for step 2, Java (a JDK).

```
py -3.13 -m venv SAM-CREPE\.venv
SAM-CREPE\.venv\Scripts\python -m pip install -r SAM-CREPE\requirements.txt
```

### Step 1: inputs

Step 1 turns your recordings into the files the later steps read: for every selection, a short clip of audio around it, and what each of the two models (SAM-Whistle and CREPE) found in that clip.

Put each recording's `.wav` and its Raven selection table (e.g. `T08_1.WAV` + `MasterT08_1.selections.txt`) in `SAM-CREPE/recordings/`. A table belongs to the `.wav` whose name its own name ends with (`MasterT08_1` ends with `T08_1`).

Each command below is `prepare_inputs.py`, then the command (what to do), then its options (the words starting with `--`, each followed by its value). Replace `<name>` with a name you choose, the same one in every command.

| Command | What it does |
|---|---|
| `cut` | Cuts a clip around every selection in every recording (at least 3 s long, centred on the selection), and makes `SAM-CREPE\inputs_<name>\`, the *inputs folder*, with one subfolder per clip. Also makes `sam_whistle_upload.zip` inside it, for Colab |
| `crepe` | Runs CREPE (the pitch tracker) on every clip in the inputs folder, on this computer. About 45 s per clip |
| `collect` | Puts the SAM-Whistle results downloaded from Colab into the inputs folder, next to the clips they belong to |

| Option | Used with | What it means |
|---|---|---|
| `--name <name>` | `cut` | Required, no default. A short name for this batch of recordings, e.g. `sars_T08`: letters, numbers, `_` or `-`, no spaces or slashes. It names the inputs folder (`inputs_<name>`) and is the `LABEL` in Colab. `cut` stops if that folder already exists, so choose a new name for each batch (or add `--replace` to delete and redo it) |
| `--inputs <folder>` | `crepe`, `collect` | The inputs folder `cut` made: `SAM-CREPE\inputs_<name>` |
| `--maps <file>` | `collect` | The zip downloaded from Colab, `conf_maps_<name>.zip`, wherever your browser saved it |

```
SAM-CREPE\.venv\Scripts\python SAM-CREPE\prepare_inputs.py cut --name <name>
SAM-CREPE\.venv\Scripts\python SAM-CREPE\prepare_inputs.py crepe --inputs SAM-CREPE\inputs_<name>
```

The CREPE settings (model, `--compress 20`, `--step 0.002`, `--high_pass 500`, `weighted_argmax`) and the window settings are in `settings.py` (step 1). Checked on 2026-11-13: `cut` gives sample-for-sample the same SAM-Whistle audio as the development windows where they overlap, and CREPE in SAM-CREPE's environment reproduces the saved CREPE scores exactly.

Colab (SAM-Whistle needs a GPU, so it runs on Google Colab; before or after `crepe`, either order): open `colab/create_sam_whistle_maps.ipynb` in Google Colab, choose a T4 GPU, upload `SAM-CREPE\inputs_<name>\sam_whistle_upload.zip`, set `LABEL = '<name>'`, run all cells, download `conf_maps_<name>.zip`.

> Note: If you don't change the LABEL, the default name is "unseen".

```
SAM-CREPE\.venv\Scripts\python SAM-CREPE\prepare_inputs.py collect --inputs SAM-CREPE\inputs_<name> --maps <downloads>\conf_maps_<name>.zip
```

Example, for a batch called `sars_T08`:

```
SAM-CREPE\.venv\Scripts\python SAM-CREPE\prepare_inputs.py cut --name sars_T08
SAM-CREPE\.venv\Scripts\python SAM-CREPE\prepare_inputs.py crepe --inputs SAM-CREPE\inputs_sars_T08
SAM-CREPE\.venv\Scripts\python SAM-CREPE\prepare_inputs.py collect --inputs SAM-CREPE\inputs_sars_T08 --maps $HOME\Downloads\conf_maps_sars_T08.zip
```

### Step 2: get fragments from SAM-Whistle (stage A)


```powershell
# Set the Java environment
$env:JAVA_HOME = 'C:\Program Files\Java\jdk-23'

# Command to run with defaults
SAM-CREPE\.venv\Scripts\python SAM-CREPE\get_fragments.py

# Run instead if you want to get fragments of min 100ms
SAM-CREPE\.venv\Scripts\python SAM-CREPE\get_fragments.py --set fragment_minlen_ms=100
# Run instead if you want to get fragments of min 150ms
SAM-CREPE\.venv\Scripts\python SAM-CREPE\get_fragments.py --set fragment_minlen_ms=150
```

Each writes `outputs/fragments_minlen<M>_maxgap<G>/`: one `.npz` per selection, plus `fragments.csv`, `settings.json` and `inputs_used.txt`. Minlen 50 is the default used by stage B; 100 and 150 are needed only for the comparison runs R1 and R0. Use `--name` to choose a folder name and `--inputs` for another inputs folder, for example `--inputs SAM-CREPE\inputs_unseen --name fragments_unseen_minlen50`.

### Step 3. Post-processing (stage B)

```
SAM-CREPE\.venv\Scripts\python SAM-CREPE\run.py --name R6
```

`--name` is required. Options: `--set name=value` (any setting, repeatable), `--fragments <folder>` (default: the stage A folder matching the fragment settings), `--inputs <folder>` (default: the one stage A used), `--stems` (only some selections), `--replace`. A run folder holds:

- `<selection>.bin`: the whistles, in SAM-Whistle's own `.bin` format and time base;
- `<selection>_provenance.csv`: every point of every whistle, and where it came from;
- `decisions.csv`: every decision of every step (fragments removed, joins judged, merges, extensions, whistles kept or dropped), with the reason; fragments are numbered as in the stage A file;
- `settings.json`, `fragments_used.txt` and `inputs_used.txt`: exactly what produced the run.

### Runs, one component switched on at a time

| Run | Fragments | Clean | Join, map | Join, map and CREPE | Merge | Extend | Finalise |
|---|---|---|---|---|---|---|---|
| R0 published | 150 / 50 | | | | | | |
| R1 bar to beat | 100 / 50 | | | | | | |
| R2 | 50 / 50 | yes | | | | | |
| R3 | 50 / 50 | yes | yes | | | | |
| R4 | 50 / 50 | yes | yes | yes | | | |
| R4m | 50 / 50 | yes | yes | yes | yes | | |
| R5 | 50 / 50 | yes | yes | yes | yes | yes | |
| R6 full | 50 / 50 | yes | yes | yes | yes | yes | yes |

R0 and R1 are SAM-Whistle without any repair (experiment 3). Because the merge mostly changes how pieces are counted (see the merge step in [DESIGN.md](DESIGN.md)), report R4 and R4m side by side, and the final results with and without it. Success means coverage at least R1's 0.88 with clearly fewer pieces (towards the published 76% of whistles in one piece, or better), fewer stray contours, and no increase in overrun or false merges. Settings are then compared by running with different values; the chosen values must be fixed before testing on unseen Sarasota recordings.

To run comparisons with one step switched on at a time, run the following commands for R0, R1, R2, R3, R4, R5, R6 (where R is a "Run"):

```powershell
# R0 is SAM-Whistle's run without my post-processing steps
# Used for comparison
# If get_fragments for min 150ms was not run previously, run here
SAM-CREPE\.venv\Scripts\python SAM-CREPE\get_fragments.py --set fragment_minlen_ms=150
# Then run R0 run
SAM-CREPE\.venv\Scripts\python SAM-CREPE\run.py --name R0 --fragments fragments_minlen150_maxgap50 --set clean=false --set join=false --set merge_duplicates=false --set extend=false --set finalise=false --set recording_tone_bands=false

# Run R1
# If get_fragments for min 100ms was not run previously, run here
SAM-CREPE\.venv\Scripts\python SAM-CREPE\get_fragments.py --set fragment_minlen_ms=100
SAM-CREPE\.venv\Scripts\python SAM-CREPE\run.py --name R1 --fragments fragments_minlen100_maxgap50 --set clean=false --set join=false --set merge_duplicates=false --set extend=false --set finalise=false --set recording_tone_bands=false

# Run R2
SAM-CREPE\.venv\Scripts\python SAM-CREPE\run.py --name R2 --set join=false --set merge_duplicates=false --set extend=false --set finalise=false

# Run R3
SAM-CREPE\.venv\Scripts\python SAM-CREPE\run.py --name R3 --set join_use_crepe=false --set merge_duplicates=false --set extend=false --set finalise=false

# Run R4
SAM-CREPE\.venv\Scripts\python SAM-CREPE\run.py --name R4 --set merge_duplicates=false --set extend=false --set finalise=false

# Run R4m
SAM-CREPE\.venv\Scripts\python SAM-CREPE\run.py --name R4m --set extend=false --set finalise=false

# Run R5
SAM-CREPE\.venv\Scripts\python SAM-CREPE\run.py --name R5 --set finalise=false

# Run R6
SAM-CREPE\.venv\Scripts\python SAM-CREPE\run.py --name R6
```

### Step 4. Figures: analysts, SAM-Whistle, CREPE and SAM-CREPE side by side

```
SAM-CREPE\.venv\Scripts\python SAM-CREPE\figures.py
SAM-CREPE\.venv\Scripts\python SAM-CREPE\figures.py --sam-run R0 --run R6 --stems T08_1_sel_026 T08_1_sel_040
```

One PNG per selection in `outputs/figures_<sam-run>_<run>/`, four panels over the same spectrogram: the analysts' traces alone; SAM-Whistle unrepaired (the `--sam-run`, default R0, the published settings); CREPE's top pick wherever its score is at least `crepe_threshold`; and SAM-CREPE (the `--run`, default R6). In every panel the analysts' traces are dashed, each contour on a whistle has its own colour (so the pieces can be counted), other contours are thin red, a short bar marks each contour's start and end, and in the SAM-CREPE panel the points join added are orange circles and the points extend added yellow triangles. The traces come from `references/`; for audio without references the analysts' panel is empty and every contour is blue.
