"""Step 4 of SAM-CREPE: one figure per audio file, comparing the analysts, SAM-Whistle, CREPE and SAM-CREPE.

Four panels over the same spectrogram (of the audio SAM-Whistle ran on, sam_input.wav):
    Analysts      the analysts' traces alone (dashed black), from the references folder
    SAM-Whistle   SAM-Whistle without repair: a run folder made with every step off (default R0, the published
                  settings, minlen 150)
    CREPE         CREPE's top pick at every 2 ms moment where its score is at least crepe_threshold (settings.py):
                  darker dots for higher scores
    SAM-CREPE     a SAM-CREPE run folder (default R6), with the points join added (orange circles) and extend added
                  (yellow triangles), from its _provenance.csv
In every panel the analysts' traces are drawn dashed for reference. Contours on an analyst's whistle are drawn in
their own colour each, so the number of pieces is visible; other contours in thin red; a short bar marks each
contour's start and end, so separate contours can be told from one joined contour. "On an analyst's whistle" is
the scorer's rule: of the contour's points inside the analyst's span, at least half within 350 Hz of the trace.
Without references (new recordings), the Analysts panel is left empty and every contour is drawn in blue.

Reads the run folders, the inputs folder they used (their inputs_used.txt) and the references folder (only this
step reads the analysts' traces; the method never does). Writes only SAM-CREPE/outputs/<name>/ (default
figures_<sam-run>_<run>), one PNG per audio file; refuses if it exists, unless --replace.

Run with SAM-CREPE's environment, from the repo root, for example:
    SAM-CREPE\\.venv\\Scripts\\python SAM-CREPE\\figures.py
    SAM-CREPE\\.venv\\Scripts\\python SAM-CREPE\\figures.py --sam-run R0 --run R6 --stems T08_1_sel_026
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg") # draw to files only, no window (must be set before pyplot is imported)
import matplotlib.pyplot as plt # noqa: E402
from matplotlib.lines import Line2D # noqa: E402
import numpy as np # noqa: E402
from scipy.io import wavfile # noqa: E402
from scipy.signal import spectrogram # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)) # so the SAM-CREPE modules below can be imported
from settings import Settings # noqa: E402
from store import OUTPUTS, output_dir # noqa: E402
from helpers.paths import DEFAULT_INPUTS, DEFAULT_REFERENCES # noqa: E402
from helpers.model_outputs import file_offsets, load_model_outputs # noqa: E402
from helpers.references import load_references # noqa: E402
from helpers.silbido_io import read_silbido # noqa: E402

TOL_HZ = 350 # the scorer's matching tolerance (MEASURES.md)
# colours: the categorical palette used in all SAM-CREPE figures
CONTOUR_COLOURS = ["#2a78d6", "#e87ba4", "#6250d6", "#1baf7a"]
STRAY_COLOUR, CREPE_COLOUR = "#e34948", "#1baf7a"
JOIN_COLOUR, EXTEND_COLOUR = "#eb6834", "#eda100"
INK, MUTED = "#0b0b0b", "#52514e"
JOIN_SOURCES = {"map", "crepe", "harmonic", "interpolated"} # provenance sources of points added by join

def on_whistle(points, analysts):
    """The scorer's rule: points inside an analyst's span, at least half of them within TOL_HZ of the trace."""
    for t, f in analysts.values():
        inside = (points[:, 0] >= t[0]) & (points[:, 0] <= t[-1]) # the contour's points within this analyst's time span
        # at least half of those points within TOL_HZ of the trace (interpolated at the points' times)
        if inside.any() and np.mean(np.abs(points[inside, 1] - np.interp(points[inside, 0], t, f)) <= TOL_HZ) >= 0.5:
            return True
    return False

def load_run(run_dir, stem, offset_s):
    """The run's contours on the recording's clock, each with its points' sources (or None)."""
    if not (run_dir / f"{stem}.bin").exists():
        return [] # no detections for this file
    # shift each contour's times onto the recording's clock
    contours = [np.column_stack([w[:, 0] + offset_s, w[:, 1]]) for w in read_silbido(run_dir / f"{stem}.bin")]
    prov = run_dir / f"{stem}_provenance.csv"
    if not prov.exists():
        return [(c, None) for c in contours] # no provenance file: points' sources unknown
    by_whistle = defaultdict(list) # whistle index: the source of each of its points, in order
    for r in csv.DictReader(open(prov, newline="")):
        by_whistle[int(r["whistle"])].append(r["source"])
    return [(c, by_whistle.get(k)) for k, c in enumerate(contours)]

# Which marker a point gets from its provenance source: "extend", "join", or None (an original SAM-Whistle point).
def source_kind(source):
    if source.startswith("extend"):
        return "extend"
    return "join" if source in JOIN_SOURCES else None

# Draw a run's contours on one panel, with start/end bars and join/extend markers. Returns how many contours.
def draw_contours(ax, contours, analysts, t0):
    k = 0 # counts contours on a whistle, so each gets its own colour
    for points, sources in contours:
        tt, ff = points[:, 0] - t0, points[:, 1] / 1000 # seconds from the window start, kHz
        # on an analyst's whistle (or every contour, without references): thick, one colour each
        if not analysts or on_whistle(points, analysts):
            colour = CONTOUR_COLOURS[min(k, len(CONTOUR_COLOURS) - 1)] if analysts else CONTOUR_COLOURS[0]
            k += 1
            ax.plot(tt, ff, c=colour, lw=2.0, zorder=4)
        else: # any other contour: thin red
            colour = STRAY_COLOUR
            ax.plot(tt, ff, c=colour, lw=1.0, zorder=2)
        # short bars at the contour's start and end
        ax.scatter(tt[[0, -1]], ff[[0, -1]], s=36, c=colour, marker="|", linewidths=1.6, zorder=6)
        # mark points added by join (circles) and extend (triangles), when provenance covers every point
        if sources and len(sources) == len(points):
            kinds = np.array([source_kind(s) for s in sources], dtype=object)
            for kind, c, marker in (("join", JOIN_COLOUR, "o"), ("extend", EXTEND_COLOUR, "^")):
                sel = kinds == kind
                if sel.any():
                    ax.scatter(tt[sel], ff[sel], s=16, c=c, marker=marker, edgecolors="white", linewidths=0.4, zorder=5)
    return len(contours)

def run_inputs(run_dir):
    """The inputs folder a run read (its inputs_used.txt), or else DEFAULT_INPUTS (helpers/paths.py)."""
    used = run_dir / "inputs_used.txt"
    return Path(used.read_text().strip()) if used.exists() else DEFAULT_INPUTS

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sam-run", default="R0", help="run folder with SAM-Whistle unrepaired (default R0)")
    ap.add_argument("--run", default="R6", help="SAM-CREPE run folder (default R6)")
    ap.add_argument("--references", default=str(DEFAULT_REFERENCES), help="folder of analysts' traces (default SAM-CREPE/references)")
    ap.add_argument("--stems", nargs="*", help="only these audio files (default: all in the SAM-CREPE run)")
    ap.add_argument("--name", help="output folder in SAM-CREPE/outputs (default figures_<sam-run>_<run>)")
    ap.add_argument("--replace", action="store_true", help="delete and redo an existing figures folder of the same name")
    args = ap.parse_args()

    # the two run folders to compare
    sam_dir, run_dir = OUTPUTS / args.sam_run, OUTPUTS / args.run
    for d in (sam_dir, run_dir):
        if not d.is_dir():
            sys.exit(f"run folder not found: {d}")
    # the SAM-CREPE run's own settings (for crepe_threshold), or the defaults
    settings = Settings.load(run_dir / "settings.json") if (run_dir / "settings.json").exists() else Settings()
    inputs = run_inputs(run_dir)
    references = Path(args.references)
    stems = args.stems or sorted(p.stem for p in run_dir.glob("*.bin")) # every file the SAM-CREPE run has contours for
    out = output_dir(args.name or f"figures_{args.sam_run}_{args.run}", replace=args.replace)

    for stem in stems:
        folder = inputs / stem
        off = file_offsets(folder) # each input file's offset to the recording's clock
        analysts = load_references(stem, references) # {trace: (times, freqs)}; empty without references
        sr, x = wavfile.read(folder / "sam_input.wav")
        # spectrogram with SAM-Whistle's frame size and hop (8 ms frames every 2 ms at 192 kHz)
        f_hz, t_s, sxx = spectrogram(x.astype(np.float64), fs=sr, nperseg=1536, noverlap=1536 - 384, window="hann")
        t0 = off["sam_input.wav"] # window start: every time is drawn relative to it
        db = 10 * np.log10(sxx + 1e-20) # power in dB (the tiny floor avoids log of zero)
        # CREPE's top-scoring pitch bin at each frame, kept where its score reaches crepe_threshold
        crepe = load_model_outputs(stem, inputs).crepe
        top = np.argmax(crepe.values, axis=1)
        score = crepe.values[np.arange(len(top)), top]
        keep = score >= settings.crepe_threshold
        c_t, c_f, c_s = crepe.t[keep] - t0, crepe.f[top][keep], score[keep]
        # frequency range to show: around the analysts' traces, or a default band
        if analysts:
            all_f = np.concatenate([f for _, f in analysts.values()])
            f_lo, f_hi = max(2000, all_f.min() - 3000), min(50000, all_f.max() + 4000)
        else:
            f_lo, f_hi = 2000, 25000
        band = (f_hz >= f_lo) & (f_hz <= f_hi)
        vmin, vmax = np.percentile(db[band], [60, 99.9]) # grey scale from the shown band's levels

        # four panels side by side, sharing the frequency axis
        titles =["Analysts", f"SAM-Whistle ({args.sam_run})", f"CREPE (top pick, score {settings.crepe_threshold:g} or more)",
                  f"SAM-CREPE ({args.run})"]
        fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.0), sharey=True)
        for ax, title in zip(axes, titles):
            # faint spectrogram background, then the analysts' traces dashed on top
            ax.pcolormesh(t_s, f_hz[band] / 1000, db[band], cmap="Greys", vmin=vmin, vmax=vmax, shading="auto",
                          rasterized=True, alpha=0.55)
            for t, f in analysts.values():
                ax.plot(t - t0, f / 1000, ls=(0, (4, 3)), lw=1.4 if title == "Analysts" else 1.2, c=INK, zorder=3)
            ax.set_title(title, fontsize=9, color=INK, loc="left")
            ax.set_xlabel("s", fontsize=8, color=MUTED)
            ax.tick_params(labelsize=7, colors=MUTED)
            ax.set_ylim(f_lo / 1000, f_hi / 1000)
        if not analysts:
            axes[0].text(0.5, 0.5, "no analyst traces", transform=axes[0].transAxes, ha="center", color=MUTED, fontsize=9)
        # fill the panels: SAM-Whistle's contours, CREPE's dots (more opaque for higher scores), SAM-CREPE's contours
        n_sam = draw_contours(axes[1], load_run(sam_dir, stem, off["sam_detections.bin"]), analysts, t0)
        axes[2].scatter(c_t, c_f / 1000, s=5, c=CREPE_COLOUR, alpha=np.clip(0.25 + c_s, 0.25, 1.0), lw=0, zorder=4)
        n_run = draw_contours(axes[3], load_run(run_dir, stem, off["sam_detections.bin"]), analysts, t0)
        axes[1].set_title(f"{titles[1]}: {n_sam} contours", fontsize=9, color=INK, loc="left")
        axes[3].set_title(f"{titles[3]}: {n_run} contours", fontsize=9, color=INK, loc="left")
        axes[0].set_ylabel("kHz", fontsize=8, color=MUTED)
        # one legend for all panels, along the bottom
        legend = [Line2D([], [], c=INK, ls=(0, (4, 3)), lw=1.4, label="analysts"),
                  Line2D([], [], c=CONTOUR_COLOURS[0], lw=2, label="contour on a whistle (one colour each)"),
                  Line2D([], [], c=STRAY_COLOUR, lw=1.0, label="other contour"),
                  Line2D([], [], ls="", marker="|", c=INK, markersize=8, mew=1.6, label="contour start / end"),
                  Line2D([], [], ls="", marker="o", mfc=CREPE_COLOUR, mec=CREPE_COLOUR, markersize=4, label="CREPE top pick"),
                  Line2D([], [], ls="", marker="o", mfc=JOIN_COLOUR, mec="white", label="added by join"),
                  Line2D([], [], ls="", marker="^", mfc=EXTEND_COLOUR, mec="white", label="added by extend")]
        fig.legend(handles=legend, loc="lower center", ncol=7, fontsize=8, frameon=False)
        # figure title: the file's name and its categories from sample.json
        info = json.loads((folder / "sample.json").read_text())
        category = ", ".join(info.get("categories", []))
        fig.suptitle(stem + (f" ({category})" if category else ""), fontsize=11, color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0.07, 1, 0.95)) # leave room for the legend below and the title above
        fig.savefig(out / f"{stem}.png", dpi=130)
        plt.close(fig)
        print(f"{stem}: written")
    print(f"written: {out}")

if __name__ == "__main__":
    main()
