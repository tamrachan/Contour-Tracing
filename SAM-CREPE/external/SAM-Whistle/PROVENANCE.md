# SAM-Whistle, the files SAM-CREPE's stage A uses

This folder is a subset of SAM-Whistle's Python package, holding only the files whose code runs when `SAM-CREPE/get_fragments.py` builds the graph (through `SAM-CREPE/helpers/graph_search.py`). It was made from the full copy in `SAM-CREPE/external/SAM-Whistle/`, so that the graph search can be run elsewhere without SAM-Whistle's model code and its dependencies.

Source: https://github.com/xixinzhang/sam_whistle, commit 8b48829203a1427e985dddae2619d862ad1395cd (2025-10-30, "fix bug of visualization shape"), copied on 2026-11-13 into `SAM-CREPE/external/SAM-Whistle/`; this subset made from that copy on 2026-11-19.

## Files

Copied unchanged:

- `sam_whistle/__init__.py`
- `sam_whistle/config.py`: `TonalConfig` and its defaults
- `sam_whistle/evaluate/__init__.py` (empty)
- `sam_whistle/evaluate/tonal_extraction/__init__.py`: starts Java and loads the silbido classes
- `sam_whistle/evaluate/tonal_extraction/jar/` (`silbido-classes-1.1-beta2.jar`, `Jama-1.0.3.jar`, `commons-math3-3.2.jar`): the graph code
- `sam_whistle/utils/audio.py`: loading the audio and making the spectrogram
- `sam_whistle/utils/tone.py`: peak picking

## Changes from the published code

- `sam_whistle/evaluate/tonal_extraction/tonal_tracker.py`: removed the line `from sam_whistle.model import *`. The model classes are only used in `TonalTracker`'s inference methods (`sam_inference` and the other model methods), which stage A never calls because it reads a confidence map that was already made. Calling one of those methods in this copy raises `NameError`.
- `sam_whistle/utils/__init__.py`: new file importing only `audio` and `tone`. The published one also imports `graph`, `prompt`, `visualize`, `anno` and `eval`, which stage A does not use and which need further packages.

Left out: `model/`, `datasets/`, `main.py`, `evaluate/eval_conf.py`, `evaluate/eval_tonal.py`, the other `utils/` modules (including `write_binary.py`, which `SAM-CREPE/finalise.py` uses from the full copy), `pyproject.toml` and `README.md`.

## Running it

Needs `numpy`, `scipy`, `matplotlib`, `tqdm`, `tyro`, `torch` (CPU build is enough), `torchaudio`, `librosa` and `jpype1`, and a JDK with `JAVA_HOME` set.

This subset has not yet been checked against the full copy. Before relying on it, run stage A with it and confirm the fragments match experiment 3's contours (`get_fragments.py` reports this in `fragments.csv`).

## Licence

The upstream repository has no licence file. Check with its authors before sharing this copy outside the VIP.
