"""Every location SAM-CREPE uses - all in one place.

Inside SAM-CREPE:
    DEFAULT_INPUTS      SAM-CREPE/inputs/: the model outputs stage A and stage B read, one folder per audio file
                        (layout in model_outputs.py). It holds the development set (the 33 selections), copied
                        from results/test_inputs/sars/ without the analysts' traces. Use --inputs to point stage A
                        at any other folder of the same layout; stage B then follows stage A.
    inputs_<set>/       other input sets made by prepare_inputs.py (step 1), in the same layout
    DEFAULT_REFERENCES  SAM-CREPE/references/: the analysts' traces, read only by figures.py (step 4), never by the method
    OUTPUTS             SAM-CREPE/outputs/: every run writes only to its own folder here (store.py)
    SAM_WHISTLE         SAM-CREPE/external/SAM-Whistle/: the copy of SAM-Whistle's code (see its PROVENANCE.md)
    CREPE               SAM-CREPE/external/CREPE/: CREPE's predict.py and the bottlenose model (see its PROVENANCE.md)
    COLAB               SAM-CREPE/colab/: the notebook that makes SAM-Whistle's confidence maps on a GPU

Outside SAM-CREPE (optional):
    DEFAULT_REFERENCE   experiment 3's graph search runs, used by get_fragments.py only to check its fragments
                        (the check is skipped if the folder is not there).
"""
from pathlib import Path

SAM_CREPE = Path(__file__).resolve().parents[1] # this file is SAM-CREPE/helpers/paths.py
DEFAULT_INPUTS = SAM_CREPE / "inputs_sars_T08"
DEFAULT_REFERENCES = SAM_CREPE / "references" # analysts' traces, read only by figures.py
CREPE = SAM_CREPE / "external" / "CREPE" # CREPE's predict.py and model (see its PROVENANCE.md)
COLAB = SAM_CREPE / "colab" # the notebook that makes SAM-Whistle's maps
OUTPUTS = SAM_CREPE / "outputs"
SAM_WHISTLE = SAM_CREPE / "external" / "SAM-Whistle"
SAM_WHISTLE_WRITER = SAM_WHISTLE / "sam_whistle" / "utils" / "write_binary.py"

DEFAULT_REFERENCE = SAM_CREPE / "baselines"
