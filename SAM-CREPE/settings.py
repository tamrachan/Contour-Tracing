"""Every setting of the SAM-CREPE pipeline - nothing in the pipeline is hard-coded.

Change a default by editing it below, or override any setting for one run on the command line:
    --set max_join_gap_ms=150 --set extend_rule=crepe_required
Every run saves the settings it used (settings.json in its output folder), so a result can always be traced
back to the exact settings that produced it, and runs with different values can be compared.

Where each default comes from:
    P  the team's manual tracing protocol
    S  SAM-Whistle's own setting
    D  measured in our data (experiments 2 and 3)
    T  decided by me (Tamra)
    G  a starting guess, to be tested
"""
import json
from dataclasses import asdict, dataclass, fields, replace

FRAME_S = 0.002   # S: SAM-Whistle and CREPE both produce one frame every 2 ms

EXTEND_RULES = ("map_carry", "crepe_required", "map_only")
PEAK_CHOICES = ("nearest", "strongest")
RANK_BY = ("support", "cost")
FILL_METHODS = ("linear", "pchip")
MERGE_KEEP = ("map", "longer")
EXTEND_PREDICTIONS = ("line", "free")


@dataclass(frozen=True)
class Settings:
    # step 1: inputs (prepare_inputs.py): cutting the audio and running CREPE
    window_min_s: float = 3.006            # S: shortest window cut around a selection: one SAM-Whistle block, 1500 frames
                                           #    (SAM-Whistle fails on shorter audio; scripts/data_prep/make_sam_input.py)
    window_margin_s: float = 0.5           # T: audio kept either side of a selection longer than window_min_s allows for
    sam_sample_rate: int = 192000          # S: SAM-Whistle's model was trained on 192 kHz audio
    crepe_model: str = "model_only-0_bottlenose_dolphins.pth"  # D: the bottlenose model in external/CREPE/ (PLAN.md step 3)
    crepe_compress: float = 20             # D: CREPE's --compress (whistles slowed into CREPE's range), as for our CREPE runs
    crepe_step_s: float = 0.002            # D: CREPE's --step: one frame every 2 ms, like SAM-Whistle's map
    crepe_high_pass_hz: float = 500        # D: CREPE's --high_pass
    crepe_decoder: str = "weighted_argmax" # D: CREPE's --decoder

    # stage A: fragments (SAM-Whistle's graph search on the saved confidence maps)
    fragment_minlen_ms: int = 50           # T: shortest piece kept. The protocol's 100 ms minimum applies to the finished
                                           #    whistle (finalise), so shorter pieces are kept for joining. D: experiment 3
                                           #    found 50 and 100 equally good; SAM-Whistle publishes 150. To experiment with.
    fragment_maxgap_ms: int = 50           # S: longest silence one piece may bridge; D: longer was worse in experiment 3
    map_threshold: float = 0.5             # S: confidence a map peak needs, in the graph search and as the map's vote
    peak_min_gap_rows: int = 2             # S: map peaks closer than this many rows are merged (TonalConfig.peak_dis_thr)
    skip_click_frames: bool = True         # S: ignore map frames that look like a click (as SAM-Whistle's graph search does)
    click_extra_fraction: float = 0.01     # S: a frame is a click if its number of peaks jumps by more than this share
                                           #    of the map's 361 rows (i.e. 4 or more extra peaks) (TonalConfig.broadband)
    peak_choice: str = "nearest"           # T: when two map peaks are in the window, take the one "nearest" the expected
                                           #    frequency (stays on the same whistle) or the "strongest" (to test later)

    # evidence: "is the whistle at time t within the window around the expected frequency?"
    crepe_threshold: float = 0.1           # D: CREPE's top pick counts only at or above this score (step 4 convention)
    window_min_hz: float = 500             # S: search window half-width is at least this (SAM-Whistle peak_tolerance_Hz)
    window_slope_frames: float = 1.0       # G: or the expected slope times this many frames, if that is wider

    # step 2: clean
    clean: bool = True
    tone_min_ms: float = 100               # D: a piece longer than this whose frequency changes less than
    tone_max_range_hz: float = 1000        #    tone_max_range_hz is a steady tone (step 4 definition)
    harmonic_tolerance: float = 0.08       # D: within 8% of 2x or 3x an overlapping piece is a harmonic (step 4 definition)
    harmonic_max_multiple: int = 5         # T, D: harmonics checked: 2x up to this many times the lower piece. Up to 5 removed
                                           #    45 more pieces, none on a whistle; above 5 almost nothing changes
    harmonic_min_overlap_ms: float = 20    # G: the two pieces must overlap in time for at least this long
    check_lone_harmonics: bool = True      # T: a piece with no lower partner is checked for a fundamental at 1/2, 1/3 of its
                                           #    frequency (map or CREPE); if found, it is a harmonic too
    lone_harmonic_min_share: float = 0.5   # G: ... found at this share of the piece's moments or more
    use_harmonic_evidence: bool = True     # T: a discarded harmonic still shows the whistle is there: its frequency divided
                                           #    by its multiple counts as evidence in join and extend (evidence.py), with
                                           #    the same yes/no vote as a map peak. To experiment with: on / off
    ignore_tone_bands: bool = True         # D: observations on a steady tone are not evidence of a whistle (the join figure showed
                                           #    paths travelling along the ~20 and ~27.6 kHz tones)
    tone_band_hz: float = 300              # G: an observation within this of a removed tone's frequency is ignored, over the whole window
    remove_tone_band_fragments: bool = True  # T, D: also remove kept fragments that sit flat (within tone_max_range_hz) inside a tone
                                           #    band: pieces of a tone too short to be caught as one. Join routed whistles up to
                                           #    such pieces at ~20 to 21 kHz (sel 032, 051; outputs/figures_R0_R1_R4m_R6)
    recording_tone_bands: bool = True      # D: also block tones found in other selections of the same recording: in the
                                           #    development recording a ~20 kHz tone runs throughout but is only traced as a
                                           #    tone in some selections
    tone_min_selections: int = 3           # G: ... if found as a tone in at least this many selections (so a one-off flat
                                           #    sound, e.g. at 6 kHz where whistles often turn, is not blocked everywhere)
    flat_harmonics_as_tones: bool = True   # D: a removed harmonic that is flat gives no harmonic evidence and its frequency is
                                           #    blocked like a tone (a flat 20.8 kHz line created false evidence at 10.4 kHz)
    flat_harmonic_min_ms: float = 40       # G: ... flat = lasting at least this long within tone_max_range_hz
    observations_min_run: int = 3          # G: n. An observation only counts in a join route if at least n of the m moments in some
    observations_run_window: int = 3       # G: m.  window of m consecutive moments around it have an observation on the same track.
                                           #    Whistles leave runs of observations, noise leaves single dots. n = m = 3 means
                                           #    "3 in a row"; n = 3, m = 5 also keeps a faint whistle caught only every 2nd
                                           #    or 3rd moment (sel 051). To experiment with
    observations_track_hz: float = 500     # G: "on the same track": within this many Hz per moment apart (the search window;
                                           #    real whistles change less than 400 Hz per moment 99% of the time)

    # step 3: join
    join: bool = True
    join_use_crepe: bool = True            # False = join on the map alone (run R3); True = map and CREPE (run R4)
    max_join_gap_ms: float = 200           # D, G: observed gaps 46 to 138 ms; to be swept
    max_join_overlap_ms: float = 10        # P: pieces may overlap by at most 10 ms
    fit_ms: float = 20                     # G: a piece's direction is a straight line through its last (or first) 20 ms
    max_unsupported_ms: float = 20         # P: a silence shorter than 20 ms is still one whistle: the path may skip at most
                                           #    this long between two observations
    max_slope_hz_per_s: float = 550000     # D: the fastest change between two points of a path (550 kHz/s; the analysts'
                                           #    traces never exceed 544, and 99% of the time stay under 200)
    freq_tolerance_hz: float = 125         # S: allowed on top of the slope limit: one row of the map
    max_jump_ratio: float = 2.0            # T: two ADJACENT points of a whistle may not differ by an octave or more (a whistle
                                           #    never suddenly jumps onto a harmonic). A and B themselves may differ by more
    max_sustained_slope_hz_per_s: float = 320000  # D: a join route may not climb or fall faster than this, sustained over
    sustained_slope_window_ms: float = 20  #    sustained_slope_window_ms. The analysts' traces reach 318 kHz/s over 20 ms at their
                                           #    99th percentile (max 353); routes climbing a click (sel 011) went faster.
                                           #    max_slope_hz_per_s limits each step; this limits a run of steps. 0 = off
    skip_cost: float = 1.0                 # G: path cost of each 2 ms moment skipped without an observation
    travel_cost_per_khz: float = 1.0       # G: path cost of each kHz the path moves up or down, so smooth paths win over
                                           #    detours (a detour up to a tone and back costs twice its height)
    allow_steps: bool = False              # P, D: allow "step" joins (an abrupt jump from A's end to B's start, ignoring the
                                           #    slope limit but not max_jump_ratio). Off: all 5 step candidates in the
                                           #    33 selections were wrong. To experiment with
    step_max_gap_ms: float = 20            # P: a step: the next piece starts within 20 ms
    same_graph_bonus: float = 0.0          # G: cost removed for pieces cut at the same crossing (0 = off)
    fill_method: str = "linear"            # T: how moments skipped between two observations of a route are filled in: "linear" (a
                                           #    straight line; best on the analysts' traces, points 5 ms apart) or "pchip"
                                           #    (a smooth curve through the neighbouring observations that never overshoots them;
                                           #    may suit skips of up to 20 ms better). Filled points are marked
                                           #    "interpolated". To compare
    rank_by: str = "support"               # D: which competing join wins: "support" (most of the gap covered by observations, then
                                           #    cheapest) or "cost" (cheapest per moment; let short joins to other sounds win)

    # step 3b: merge parallel duplicates (optional)
    merge_duplicates: bool = True          # T, D: merge two whistles that are the same whistle traced twice side by side.
                                           #    In the 33 selections 57 to 81% of the breaks between pieces were such
                                           #    parallel traces, all on one ridge of the map (the join research check,
                                           #    docs/join_research_check.md in the project repository)
    merge_max_hz: float = 350              # S: the two traces count as the same whistle where they are within this
                                           #    (SAM-Whistle's matching tolerance)
    merge_min_overlap_ms: float = 10       # P: ... and they overlap in time for longer than this (shorter overlaps are joins)
    merge_min_share: float = 0.8           # G: ... and they are within merge_max_hz at this share of the overlap or more
    merge_same_graph_only: bool = True     # D: ... and they share a SAM-Whistle graph (all 17 parallel pairs did), so a
                                           #    second dolphin's whistle passing close by is not merged
    merge_keep: str = "map"                # T: which trace is kept where the two overlap: "map" (the one higher on the map's
                                           #    ridge, i.e. with the higher average confidence score) or "longer". To compare

    # step 4: extend
    extend: bool = True
    extend_rule: str = "crepe_required"    # T, D: crepe_required: CREPE must support every added moment; map_carry: the map
                                           #    alone may continue for map_carry_ms after CREPE's last support; map_only:
                                           #    CREPE is ignored. On the 33 selections crepe_required and map_carry gave the
                                           #    same coverage (0.90), with 3 instead of 29 points past whistle ends
                                           #    (2026-11-13). The others: to experiment with
    map_carry_ms: float = 10               # D, G: the map stays on just past real whistle ends (its frames are 8 ms)
    extend_map_fallback: bool = False      # T, D: if CREPE has not agreed at the whistle's tip, may the map alone still extend it
                                           #    (for at most map_carry_ms)? Off: only 3 of 44 missed starts had map evidence
                                           #    and no CREPE, while past ends that are already right the map fires at 28% of
                                           #    the first 10 ms (CREPE 7%). To experiment with
    extend_stop_unsupported_ms: float = 20 # P: stop after 20 ms without support (a gap of more than 20 ms makes two whistles)
    max_extension_ms: float = 150          # D: how far one end may grow; missed starts and ends observed up to 114 ms
    extend_observations_min_run: int = 3   # G: n and m of the "n of m" rule (see observations_min_run), for extend only. At
    extend_observations_run_window: int = 3  #  a whistle's edge single stray dots are the main risk. n = 1 switches it off
    extend_prediction: str = "free"        # T, D: where extend looks next: "line" (the search window around the straight line
                                           #    through the last fit_ms) or "free" (evidence near the line first, as with "line";
                                           #    only if there is none, as widely as the slope limit allows, so it can follow a
                                           #    turn such as the hook many whistles start with). On the 33 selections "free"
                                           #    improved coverage in 9 (sel 006: 0.89 to 0.98), worsened none, and halved the
                                           #    median missed start (10 to 5.5 ms) (2026-11-13)
    extend_stop_at_other: bool = True      # T, D: stop extending where another whistle is at the same moment within window_min_hz
                                           #    of the expected frequency. Whistles may overlap in time; this only stops one
                                           #    running onto another's trace. On the 33 selections all 40 stops were alongside
                                           #    parallel duplicates (none at two crossing whistles). To experiment with: off
    max_whistle_ms: float = 0              # P, T: extend never makes a whistle longer than this. The protocol gives no maximum
                                           #    (only the 100 ms minimum), so 0 = no limit. For reference: the longest analyst
                                           #    trace in the 33 selections is 1,376 ms

    # step 5: finalise
    finalise: bool = True
    min_whistle_ms: float = 100            # P: whistles shorter than this are dropped
    finalise_recheck_tones: bool = True    # T: apply the tone rule again to finished whistles, so short pieces of a steady
                                           #    tone (under tone_min_ms, so not caught by clean) joined into a long one are removed

    def __post_init__(self):
        if self.extend_rule not in EXTEND_RULES:
            raise ValueError(f"extend_rule must be one of {EXTEND_RULES}, not {self.extend_rule!r}")
        if self.observations_run_window < self.observations_min_run:
            raise ValueError("observations_run_window (m) must be at least observations_min_run (n)")
        if self.extend_observations_run_window < self.extend_observations_min_run:
            raise ValueError("extend_observations_run_window (m) must be at least extend_observations_min_run (n)")
        if self.fill_method not in FILL_METHODS:
            raise ValueError(f"fill_method must be one of {FILL_METHODS}, not {self.fill_method!r}")
        if self.rank_by not in RANK_BY:
            raise ValueError(f"rank_by must be one of {RANK_BY}, not {self.rank_by!r}")
        if self.peak_choice not in PEAK_CHOICES:
            raise ValueError(f"peak_choice must be one of {PEAK_CHOICES}, not {self.peak_choice!r}")
        if self.extend_prediction not in EXTEND_PREDICTIONS:
            raise ValueError(f"extend_prediction must be one of {EXTEND_PREDICTIONS}, not {self.extend_prediction!r}")
        if self.merge_keep not in MERGE_KEEP:
            raise ValueError(f"merge_keep must be one of {MERGE_KEEP}, not {self.merge_keep!r}")
        if not 1 < self.max_jump_ratio <= 2:
            raise ValueError("max_jump_ratio must be above 1 and at most 2 (an octave)")

    def with_overrides(self, pairs):
        """A copy with "name=value" overrides applied, each converted to that setting's type."""
        types = {f.name: f.type for f in fields(self)}
        changes = {}
        for pair in pairs or []:
            name, _, text = pair.partition("=")
            if name not in types:
                raise ValueError(f"unknown setting {name!r}; settings are: {', '.join(types)}")
            kind = types[name]
            if kind in (bool, "bool"):
                if text.lower() not in ("true", "false", "1", "0"):
                    raise ValueError(f"{name} must be true or false, not {text!r}")
                changes[name] = text.lower() in ("true", "1")
            elif kind in (int, "int"):
                changes[name] = int(text)
            elif kind in (float, "float"):
                changes[name] = float(text)
            else:
                changes[name] = text
        return replace(self, **changes)

    def save(self, path):
        path.write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, path):
        return cls(**json.loads(path.read_text()))
