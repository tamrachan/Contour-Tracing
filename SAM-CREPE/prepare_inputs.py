"""Step 1 of SAM-CREPE: make the inputs (SAM-Whistle's confidence maps and CREPE's scores) for new audio.

Three commands, run in this order:

    cut      Cut the audio into the windows SAM-Whistle and CREPE analyse, and write an inputs folder,
             SAM-CREPE/inputs_<name>/, with one subfolder per window. Three ways in:
                 (nothing)                                  every recording in RECORDINGS (SAM-CREPE/recordings/, or
                                                            --folder): each .wav with its Raven selection table, one
                                                            window per selection; all recordings in one inputs folder
                 --recording R.wav --selections table.txt   one window per selection of a Raven selection table
                                                            (like MasterT08_1.selections.txt), centred on it
                 --recording R.wav                          the whole file as one window
             In the folder, a table belongs to the .wav whose name its own name ends with, before .selections.txt
             (or .txt): T08_1.WAV goes with MasterT08_1.selections.txt. A .wav without exactly one table is skipped
             with a warning (a whole long recording as one window is slow; use --recording for that).
             Each subfolder gets:
                 sam_input.wav     the window for SAM-Whistle: resampled to sam_sample_rate (192 kHz, what its model
                                   was trained on), and exactly a whole number of its 2 ms frames long, at least
                                   one 3 s block (window_min_s)
                 crepe_input.wav   the same window at the recording's own sample rate, for CREPE
                 sample.json       where the window sits in the recording (offsets_s: seconds to add to each file's
                                   times to put them on the recording's clock)
             It also writes sam_whistle_upload.zip (audio/<window>.wav and meta.json), to drag into the Colab
             notebook (SAM-CREPE/colab/create_sam_whistle_maps.ipynb), which makes the confidence maps on a GPU.
             crepe and the Colab notebook both start from what cut wrote, so they can run in either order.
    crepe    Run CREPE (external/CREPE/predict.py, the bottlenose model) on every crepe_input.wav, with the CREPE
             settings in settings.py, and save crepe_act.npz (every pitch bin's score, every 2 ms) and
             crepe_f0.csv. Runs on the CPU: about 45 s per 3 s window.
    collect  Unpack the conf_maps_<label>.zip downloaded from Colab into the inputs folder (conf_map.npz in each
             subfolder), checking that every window has its map and that the map fits the window.

After these, the inputs folder is ready for step 2 (get_fragments.py --inputs SAM-CREPE/inputs_<name>).

Times: selection windows are centred on the selection and at least window_min_s long (window_margin_s either side
of longer selections), moved inside the recording if they would run past its start or end. SAM-Whistle labels a
frame by its start and its frames are 8 ms long, so its map and detections are offset by half a frame (4 ms) more
than the audio; CREPE labels its frames by their centre.

SAFETY: writes only SAM-CREPE/inputs_<name>/ (refuses if it exists, unless --replace, which deletes only that
folder). Never touches any other inputs folder or the recordings, and never reads the analysts'
traces.

Run with SAM-CREPE's environment, from the repo root, for example:
    SAM-CREPE\\.venv\\Scripts\\python SAM-CREPE\\prepare_inputs.py cut --name unseen
    SAM-CREPE\\.venv\\Scripts\\python SAM-CREPE\\prepare_inputs.py cut --recording data\\raw\\sars_T08_1\\T08_1.WAV
        --selections data\\raw\\sars_T08_1\\MasterT08_1.selections.txt --name unseen --exclude SAM-CREPE\\inputs_<earlier>
    SAM-CREPE\\.venv\\Scripts\\python SAM-CREPE\\prepare_inputs.py crepe --inputs SAM-CREPE\\inputs_unseen
    (Colab: run colab/create_sam_whistle_maps.ipynb with inputs_unseen\\sam_whistle_upload.zip; download
     conf_maps_unseen.zip)
    SAM-CREPE\\.venv\\Scripts\\python SAM-CREPE\\prepare_inputs.py collect --inputs SAM-CREPE\\inputs_unseen
        --maps conf_maps_unseen.zip
"""
import argparse
import csv
import io
import json
import math
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)) # so the SAM-CREPE modules below can be imported
from settings import Settings # noqa: E402
from helpers.paths import CREPE, SAM_CREPE # noqa: E402
from helpers.model_outputs import file_offsets # noqa: E402

# Fixed by SAM-Whistle's spectrogram (its TonalConfig, as run on Colab: n_fft 1536 and hop 384 samples at 192 kHz,
# i.e. 8 ms frames every 2 ms, labelled by their start), not settings to change.
SAM_N_FFT, SAM_HOP = 1536, 384
PAD_S = 0.025 # audio read beyond each end before resampling, then trimmed off (no resampling edge effects)

# Where `cut` looks for recordings when no --recording is given: each recording's .wav with its Raven selection
# table (.txt). Change it here, or for one run with --folder.
RECORDINGS = SAM_CREPE / "recordings"

def inputs_folder(name, replace):
    """SAM-CREPE/inputs_<name>/, new. Refuses an existing one unless replace (then deletes only that folder)."""
    # reject names that could point outside SAM-CREPE/
    if not name or any(c in name for c in r"\/:") or name in (".", ".."):
        raise ValueError(f"--name {name!r} must be a plain folder name")
    out = SAM_CREPE / f"inputs_{name}"
    if out.exists():
        if not replace:
            sys.exit(f"{out} already exists. Choose another --name, or pass --replace to delete and redo it.")
        shutil.rmtree(out)
    out.mkdir()
    return out

def read_selections(path):
    """{selection number: (begin s, end s)} from a Raven selection table (tab separated)."""
    sel = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            sel[int(row["Selection"])] = (float(row["Begin Time (s)"]), float(row["End Time (s)"]))
    return sel

def recording_of(stem):
    """The recording a window comes from: the part of its name before "_sel_" (as run.py reads it)."""
    return stem.split("_sel_")[0]

def find_recordings(folder):
    """[(wav, selection table)] for the recordings in folder. A table belongs to the .wav whose name (any case) its
    own name ends with, before .selections.txt or .txt: T08_1.WAV goes with MasterT08_1.selections.txt (and not
    MasterT08_10.selections.txt). Skips, with a warning, a .wav with no table or several, a table that fits several
    .wav files, and a .wav whose name contains "_sel_" (window names must split back into recording and selection)."""
    folder = Path(folder)
    if not folder.is_dir():
        sys.exit(f"{folder} is not a folder. Put each recording's .wav and its Raven selection table (.txt) in it, "
                 f"or cut one recording with --recording.")
    files = sorted(p for p in folder.iterdir() if p.is_file())
    wavs = [p for p in files if p.suffix.lower() == ".wav"]
    tables = [p for p in files if p.suffix.lower() == ".txt"]

    # the part of a table's name that a .wav name must end with
    def key(table):
        name = table.name.lower()
        return name[:-len(".selections.txt")] if name.endswith(".selections.txt") else name[:-len(".txt")]

    fits = {t: [w for w in wavs if key(t).endswith(w.stem.lower())] for t in tables} # each table: the .wav files it could belong to
    pairs = []
    # pair each .wav with its one table, or warn and skip it
    for w in wavs:
        mine = [t for t in tables if w in fits[t]]
        if "_sel_" in w.stem:
            print(f"WARNING skipped {w.name}: its name contains '_sel_'; rename it")
        elif not mine:
            print(f"WARNING skipped {w.name}: no selection table (a .txt whose name ends with {w.stem!r})")
        elif len(mine) > 1:
            print(f"WARNING skipped {w.name}: more than one selection table fits it: {', '.join(t.name for t in mine)}")
        elif len(fits[mine[0]]) > 1:
            print(f"WARNING skipped {w.name}: its table {mine[0].name} also fits "
                  f"{', '.join(o.name for o in fits[mine[0]] if o != w)}; rename the files so each table fits one")
        else:
            pairs.append((w, mine[0]))
    # warn about tables that no .wav uses
    for t in tables:
        if not fits[t]:
            print(f"WARNING not used: {t.name} (no .wav whose name it ends with)")
    return pairs

def taken_spans(folders):
    """{recording: [(start, end), ...]}: the spans of each recording already covered by the windows of these inputs
    folders (--exclude), the recording read from each window's name as in run.py."""
    spans = {}
    for folder in folders:
        if not Path(folder).is_dir():
            sys.exit(f"--exclude {folder}: no such folder")
        for item in Path(folder).iterdir():
            # a window folder: its span runs from its start for the length of its sam_input.wav
            if (item / "sample.json").exists() and (item / "sam_input.wav").exists():
                start = file_offsets(item)["sam_input.wav"]
                spans.setdefault(recording_of(item.name), []).append(
                    (start, start + sf.info(str(item / "sam_input.wav")).duration))
    return spans

def windows(recording, selections, settings, duration, only):
    """[(name, selection number or None, begin, end, window start, window length)] in the recording's seconds."""
    stem = Path(recording).stem
    # no selection table: the whole file as one window, at least window_min_s long
    if not selections:
        return [(stem, None, 0.0, duration, 0.0, max(duration, settings.window_min_s))]
    out = []
    for n, (b, e) in sorted(read_selections(selections).items()):
        if only and n not in only:
            continue
        # at least window_min_s, or the selection with window_margin_s either side
        length = max(settings.window_min_s, (e - b) + 2 * settings.window_margin_s)
        # centred on the selection, moved inside the recording if it would run past its start or end
        start = min(max(0.0, (b + e) / 2 - length / 2), max(0.0, duration - length))
        out.append((f"{stem}_sel_{n:03d}", n, b, e, start, length))
    return out

def write_window(recording, info, item, settings, channel, out):
    """Cut one window of the recording into out/<name>/ (sam_input.wav, crepe_input.wav, sample.json).
    Returns its index.csv row."""
    name, n, b, e, start, length = item
    sr, sr_out = info.samplerate, settings.sam_sample_rate
    # resampling ratio (up / down) from the recording's rate to SAM-Whistle's
    g = math.gcd(sr_out, sr)
    up, down = sr_out // g, sr // g
    pad = int(round(PAD_S * sr)) # PAD_S in samples
    i0 = int(round(start * sr)) # the window's first sample in the recording
    n_in = int(round(length * sr)) # the window's length in samples
    lo, hi = max(0, i0 - pad), min(info.frames, i0 + n_in + pad) # the samples to read, with padding, kept inside the file
    x, _ = sf.read(str(recording), start=lo, stop=hi, dtype="float64", always_2d=True)
    x = x[:, channel] # the chosen channel only
    # CREPE's window: the recording's own samples (zero-padded at the end if the recording is shorter)
    crepe = x[i0 - lo:i0 - lo + n_in]
    crepe = np.pad(crepe, (0, n_in - len(crepe)))
    # SAM-Whistle's window: resampled, then exactly n_frames frames long
    # (enough frames for at least window_min_s, and enough to cover the whole window)
    n_frames = max(int(round(settings.window_min_s * sr_out - SAM_N_FFT)) // SAM_HOP + 1,
                   math.ceil((length * sr_out - SAM_N_FFT) / SAM_HOP) + 1)
    n_out = SAM_N_FFT + (n_frames - 1) * SAM_HOP # samples spanned by n_frames frames
    y = resample_poly(x, up, down) # resample the padded audio
    k0 = int(round((i0 - lo) * up / down)) # the window's start in the resampled audio (past the padding)
    sam = y[k0:k0 + n_out]
    sam = np.pad(sam, (0, n_out - len(sam))) # zero-pad at the end if short
    t0 = i0 / sr # exact window start on the recording's clock
    folder = out / name
    folder.mkdir()
    sf.write(folder / "sam_input.wav", sam.astype(np.float32), sr_out, subtype="FLOAT")
    sf.write(folder / "crepe_input.wav", crepe.astype(np.float32), sr, subtype="FLOAT")
    half_frame = SAM_N_FFT / sr_out / 2 # 4 ms: SAM-Whistle labels its frames by their start, not their centre
    # where the window sits in the recording, and each file's offset to the recording's clock
    (folder / "sample.json").write_text(json.dumps(dict(
        stem=name, recording=str(Path(recording).resolve()), selection=n,
        begin_s=round(b, 6), end_s=round(e, 6), window_start_s=round(t0, 6),
        offsets_s={"sam_input.wav": round(t0, 6), "conf_map.npz": round(t0 + half_frame, 6),
                   "sam_detections.bin": round(t0 + half_frame, 6),
                   "crepe_input.wav": round(t0, 6), "crepe_act.npz": round(t0, 6)}), indent=1))
    print(f"{name}: {t0:.3f} s, {n_out / sr_out:.3f} s, {n_frames} frames")
    return dict(stem=name, recording=Path(recording).name, selection=n if n is not None else "", begin_s=b,
                end_s=e, window_start_s=round(t0, 6), window_s=round(n_out / sr_out, 6), frames=n_frames)

# The cut command: plan every window of the recordings, then write the inputs folder and the Colab upload zip.
def cut(args):
    settings = Settings().with_overrides(args.set) # the defaults, with any --set overrides
    # which recordings to cut: the one given with --recording, or every pair found in the folder
    if args.recording:
        jobs = [(Path(args.recording), args.selections)]
    elif args.selections:
        sys.exit("--selections needs --recording (or put both in the recordings folder and give neither)")
    else:
        folder = Path(args.folder) if args.folder else RECORDINGS
        jobs = find_recordings(folder)
        if not jobs:
            sys.exit(f"no recordings to cut in {folder}")
        print(f"{len(jobs)} recording(s) in {folder}: " + ", ".join(f"{w.name} + {t.name}" for w, t in jobs))
    spans = taken_spans(args.exclude) if args.exclude else {} # spans already used by the --exclude folders
    # plan every window first, so nothing is written if a recording can't be used
    plan, skipped = [], 0
    for recording, selections in jobs:
        info = sf.info(str(recording))
        if not 0 <= args.channel < info.channels:
            sys.exit(f"{recording.name} has {info.channels} channel(s); --channel {args.channel} is not one of them")
        items = windows(recording, selections, settings, info.duration, args.only)
        taken = spans.get(recording.stem, []) # leave out selections inside a window already used
        keep = [it for it in items if not any(a < it[3] and it[2] < b for a, b in taken)]
        skipped += len(items) - len(keep)
        if keep:
            plan.append((recording, info, keep))
    if not plan:
        sys.exit("no windows to cut")
    out = inputs_folder(args.name, args.replace)
    rows = []
    # write each window, and add its sam_input.wav to the zip for the Colab notebook
    with zipfile.ZipFile(out / "sam_whistle_upload.zip", "w", zipfile.ZIP_DEFLATED) as upload:
        for recording, info, items in plan:
            for item in items:
                rows.append(write_window(recording, info, item, settings, args.channel, out))
                upload.write(out / item[0] / "sam_input.wav", f"{args.name}/audio/{item[0]}.wav")
        # meta.json for the notebook: every window is a test file (none for training)
        upload.writestr(f"{args.name}/meta.json", json.dumps({"train": [], "test": [r["stem"] for r in rows]},
                                                             indent=1))
    # index.csv: one row per window
    with open(out / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    settings.save(out / "settings.json") # the settings used, read again by crepe
    print(f"\n{len(rows)} windows from {len(plan)} recording(s) in {out}"
          + (f"; {skipped} selections left out (inside an --exclude window)" if skipped else ""))
    print(f"next, in either order: upload {out / 'sam_whistle_upload.zip'} to colab/create_sam_whistle_maps.ipynb "
          f"(LABEL = '{args.name}'), and run: SAM-CREPE\.venv\Scripts\python SAM-CREPE/prepare_inputs.py crepe --inputs {out}")

# The crepe command: run CREPE on each window's crepe_input.wav, saving crepe_act.npz and crepe_f0.csv.
def run_crepe(args):
    inputs = Path(args.inputs)
    # the settings cut saved (or the defaults), with any --set overrides
    settings = Settings.load(inputs / "settings.json") if (inputs / "settings.json").exists() else Settings()
    settings = settings.with_overrides(args.set)
    items = sorted(p.parent for p in inputs.glob("*/crepe_input.wav")) # every window folder
    # skip windows that already have CREPE's output, unless --redo
    todo =[p for p in items if args.redo or not (p / "crepe_act.npz").exists()]
    print(f"CREPE on {len(todo)} of {len(items)} windows (about 45 s each on the CPU)")
    for k, item in enumerate(todo, start=1):
        # CREPE's predict.py on this window with the CREPE settings, saving every pitch bin's scores
        cmd = [sys.executable, str(CREPE / "predict.py"), str(item / "crepe_input.wav"),
               "--model_path", str(CREPE / settings.crepe_model), "--compress", str(settings.crepe_compress),
               "--step", str(settings.crepe_step_s), "--high_pass", str(settings.crepe_high_pass_hz),
               "--decoder", settings.crepe_decoder, "--no_characterisation", "--no_print", "--save_activations"]
        done = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if done.returncode != 0:
            sys.exit(f"CREPE failed on {item.name}:\n{done.stderr[-2000:]}")
        # rename predict.py's outputs to the names the later steps read
        (item / "crepe_input_act.npz").replace(item / "crepe_act.npz")
        (item / "crepe_input_f0.csv").replace(item / "crepe_f0.csv")
        print(f"  {k}/{len(todo)} {item.name}")
    print("done")

# The collect command: copy each window's confidence map from Colab's download into its folder, if it fits.
def collect(args):
    inputs = Path(args.inputs)
    items = {p.parent.name: p.parent for p in inputs.glob("*/sam_input.wav")} # window name: its folder
    maps = Path(args.maps)
    found = {} # window name: the bytes of its confidence map
    # --maps is either a folder of <window>_conf.npz or the zip from Colab
    if maps.is_dir():
        for p in maps.rglob("*_conf.npz"):
            found[p.name[:-len("_conf.npz")]] = p.read_bytes()
    else:
        with zipfile.ZipFile(maps) as z:
            for name in z.namelist():
                if name.endswith("_conf.npz"):
                    found[Path(name).name[:-len("_conf.npz")]] = z.read(name)
    missing, bad = [], []
    # each window needs a map with one column per SAM-Whistle frame of its sam_input.wav
    for stem, item in sorted(items.items()):
        if stem not in found:
            missing.append(stem)
            continue
        conf = np.load(io.BytesIO(found[stem]))["conf"]
        frames = (sf.info(str(item / "sam_input.wav")).frames - SAM_N_FFT) // SAM_HOP + 1
        if conf.shape[1] != frames:
            bad.append(f"{stem}: map has {conf.shape[1]} frames, window has {frames}")
            continue
        (item / "conf_map.npz").write_bytes(found[stem]) # saved unchanged as conf_map.npz
    # report what was added, missing, or rejected
    print(f"{len(items) - len(missing) - len(bad)} of {len(items)} maps added to {inputs}")
    if missing:
        print(f"no map for {len(missing)}: {', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}")
    if bad:
        print("maps that do not fit their window (not added):\n  " + "\n  ".join(bad))

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True) # one subcommand per step: cut, crepe, collect
    # cut's options
    c = sub.add_parser("cut", help="cut the audio into windows and write an inputs folder")
    c.add_argument("--folder", help=f"folder of recordings, each .wav with its Raven selection table "
                                    f"(default: {RECORDINGS}); used when --recording is not given")
    c.add_argument("--recording", help="one recording (.wav) instead of the folder")
    c.add_argument("--selections", help="with --recording: its Raven selection table, one window per selection "
                                        "(default: the whole file)")
    c.add_argument("--name", required=True, help="the inputs folder is SAM-CREPE/inputs_<name>")
    c.add_argument("--only", nargs="*", type=int, help="only these selection numbers")
    c.add_argument("--exclude", nargs="*", help="inputs folders whose windows' selections are left out "
                                                "(e.g. an earlier SAM-CREPE\\inputs_<name>, so its selections are not cut again)")
    c.add_argument("--channel", type=int, default=0, help="which channel of the recording (default 0)")
    c.add_argument("--set", action="append", metavar="NAME=VALUE", help="override a step 1 setting in settings.py")
    c.add_argument("--replace", action="store_true", help="delete and redo an existing inputs folder of the same name")
    # crepe's options
    r = sub.add_parser("crepe", help="run CREPE on every window of an inputs folder")
    r.add_argument("--inputs", required=True, help="the inputs folder made by cut")
    r.add_argument("--set", action="append", metavar="NAME=VALUE", help="override a CREPE setting in settings.py")
    r.add_argument("--redo", action="store_true", help="also redo windows that already have crepe_act.npz")
    # collect's options
    m = sub.add_parser("collect", help="add the confidence maps from Colab to an inputs folder")
    m.add_argument("--inputs", required=True, help="the inputs folder made by cut")
    m.add_argument("--maps", required=True, help="conf_maps_<label>.zip from Colab (or a folder of <window>_conf.npz)")
    args = ap.parse_args()
    # run the chosen command
    if args.command == "cut":
        cut(args)
    elif args.command == "crepe":
        run_crepe(args)
    else:
        collect(args)

if __name__ == "__main__":
    main()
