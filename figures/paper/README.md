# Paper figures

Main-paper filenames follow the compiled figure order:
1. figP1_observation_ladder
2. figP2_offset_and_scope
3. figP3_ladder_of_auditors_qwen05b
4. figP4_detection_vs_jailbreak
5. figP5_aware_attacker_single_layer

Other saved assets use the appendix_ prefix. This includes supplementary and parked figures; PGD and CAA depth plots are retained but not included in Section 4.1 or newly inserted into the appendix.

Generation: .venv/bin/python src/plot_paper.py paper1 (or paper2 through paper5). Use appendix for supplementary assets. PDF, PNG, and data/caption sidecars follow the same stem.
