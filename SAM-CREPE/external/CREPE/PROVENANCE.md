# CREPE, copied for SAM-CREPE

This folder holds what SAM-CREPE's step 1 needs to compute CREPE's scores: a modified version of `predict.py`, the bottlenose model `model_only-0_bottlenose_dolphins.pth` and the repository's `README.md`. 

Modified `predict.py`: [Tamra Chan's fork](https://github.com/tamrachan/bioacoustic_F0_estimation) adds `--save_activations` (every pitch bin's score for every frame, saved as `<name>_act.npz`) to the [original code](https://gitlab.lis-lab.fr/paul.best/f0_estimation) by Paul Best. The model is the one trained on bottlenose dolphin whistles from the paper's experiments (`paper_experiments/crepe_weights/` in the original repository).

Left out: the repository's other models, paper experiments, data, GUI and virtual environment.