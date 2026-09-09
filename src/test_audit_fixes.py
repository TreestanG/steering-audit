"""Regression checks for audit items 8 and 9 (7 Sep 2026).

    uv run python src/test_audit_fixes.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import detect
import join_pgd
import pgd_sipit

RULE = dict(sigma=3.0, min_run=1, stride=1, sigma_any=4.5, dense_tail=0)
UNIT = {1: detect.LayerStat(0, 0.0, 1.0)}


def cal(role_stats: dict, **kw) -> dict:
    base = {"statistic": "role_z", "chat_ids": None, "transform": "none",
            "content_scheme": "end", "end_cap": 6, "chat_header_len": 3,
            "role_stats": role_stats}
    base.update(kw)
    return base


def fitted(*roles: str) -> dict:
    return {"1": {r: {"n": 10, "mean": 0.0, "sd": 1.0} for r in roles}}


def row(residuals: list[float], tokens: list[int] | None = None, **extra) -> dict:
    tokens = tokens or [7] * len(residuals)
    return {"id": "p0", "layer": 1, "steps": [
        {"token": t, "residual": r, "h_norm": 1.0, "matched": True}
        for t, r in zip(tokens, residuals)], **extra}


def verdict(r: dict, c: dict) -> dict:
    prof, cov = detect.profiles_from_rows([r], 1, c)
    rep = detect.prompt_fpr(prof, UNIT, [1], coverage=cov, **RULE)
    return rep["per_id"]["p0"] | {"n_unscorable": rep["n_unscorable"]}


def test_no_fitted_role_is_unscorable_not_clear():
    c = cal(fitted("content_e3"))
    r = row([100.0, 100.0])
    d = detect.role_z_detail(r, c, 1)
    assert math.isnan(d["score"]) and not d["complete"]
    assert d["missing_roles"] == ["content_e1", "content_e0"]
    v = verdict(r, c)
    assert v["verdict"] == "unscorable" and not v["flagged"] and v["n_unscorable"] == 1


def test_partial_coverage_flag_stands_but_quiet_is_unscorable():
    c = cal(fitted("content_e0"))
    assert verdict(row([100.0, 100.0]), c)["verdict"] == "flagged"
    assert verdict(row([0.5, 0.5]), c)["verdict"] == "unscorable"


def test_full_coverage_clears():
    c = cal(fitted("content_e0", "content_e1"))
    v = verdict(row([0.5, 0.5]), c)
    assert v["verdict"] == "clear" and v["n_unscorable"] == 0


def test_truncated_inversion_is_unscorable():
    c = cal(fitted("content_e0", "content_e1"))
    v = verdict(row([0.5, 0.5], n_target=5, n_recovered=2), c)
    assert v["verdict"] == "unscorable"


def test_missing_layer_calibration_is_a_gap():
    c = cal(fitted("content_e0", "content_e1"))
    prof, cov = detect.profiles_from_rows([row([0.5, 0.5])], 1, c)
    rep = detect.prompt_fpr(prof, {}, [1], coverage=cov, **RULE)
    assert rep["per_id"]["p0"]["verdict"] == "unscorable"


def test_prefix_fingerprint_guards_template_roles():
    chat = (100, 101, 102)
    tokens = [100, 102, 5, 9, 9, 101]
    c = cal(fitted("pre0", "pre1", "pre2", "content_first", "content_second", "suf-1"),
            chat_ids=list(chat), prefix_fingerprints=[[100, 102, 5]])
    assert detect.position_roles(tokens, chat) == [
        "pre0", "pre1", "pre2", "content_first", "content_second", "suf-1"]
    assert verdict(row([0.5] * 6, tokens), c)["verdict"] == "clear"
    other = [100, 102, 6, 9, 9, 101]
    d = detect.role_z_detail(row([0.5] * 6, other), c, 1)
    assert d["prefix_match"] is False and d["missing_roles"] == ["pre0", "pre1", "pre2"]
    assert verdict(row([0.5] * 6, other), c)["verdict"] == "unscorable"
    assert verdict(row([0.5, 0.5, 0.5, 50.0, 0.5, 0.5], other), c)["verdict"] == "flagged"
    assert detect.prefix_fingerprints([row([0.5] * 6, tokens)], chat) == [[100, 102, 5]]


def test_zero_residual_positions_count_as_covered():
    c = cal(fitted("content_e0", "content_e1"), transform="log")
    assert verdict(row([0.0, 0.0]), c)["verdict"] == "clear"
    unfitted = cal(fitted("content_e0"), transform="log")
    assert verdict(row([0.0, 0.5]), unfitted)["verdict"] == "clear"
    assert verdict(row([0.5, 0.5]), unfitted)["verdict"] == "unscorable"
    assert verdict(row([50.0, 0.5]), unfitted)["verdict"] == "unscorable"


def test_duplicate_rows_raise():
    c = cal(fitted("content_e0", "content_e1"))
    try:
        detect.profiles_from_rows([row([0.5, 0.5]), row([0.6, 0.6])], 1, c)
    except ValueError as e:
        assert "two rows" in str(e)
    else:
        raise AssertionError("a repeated (id, layer) was silently overwritten")


def test_join_cell_key_separates_sweeps():
    base = {"behavior": "jbb_refusal", "test_arm": "harmful", "budget": 0.02, "arm": "pgd",
            "n_positions": 1, "constraint": "all", "objective": "harm", "inj_layer": 17,
            "deltas_sha256": "a" * 64, "n_prompts": 15}
    k0 = join_pgd.cell_key(base)
    for field, other in (("inj_layer", 12), ("objective", "other"), ("n_prompts", 50),
                         ("deltas_sha256", "b" * 64)):
        assert join_pgd.cell_key(base | {field: other}) != k0, field
    clean = base | {"arm": "clean", "objective": "-", "budget": 0.0}
    assert join_pgd.cell_key(clean) == join_pgd.cell_key(clean | {"inj_layer": 12,
                                                                    "deltas_sha256": None})


def test_matching_cells_prefers_recorded_prompt_count():
    import types
    blob = {"behavior": "jbb_refusal", "arm": "harmful", "budget": 0.12, "n_prompts": 50}
    legacy = ("jbb_refusal", "harmful", 0.12, "pgd", 1, "all", "target", 17, None, None)
    exact = legacy[:9] + (50,)
    other_n = legacy[:9] + (15,)
    a = types.SimpleNamespace(objective=None, inj_layer=None, n_positions=None, constraint=None)
    verdicts = {legacy: {}, exact: {}, other_n: {}}
    assert join_pgd.matching_cells(verdicts, blob, "pgd", a) == [exact]
    assert join_pgd.matching_cells({legacy: {}, other_n: {}}, blob, "pgd", a) == [legacy]
    two_sweeps = {exact: {}, exact[:7] + (12, None, 50): {}}
    assert len(join_pgd.matching_cells(two_sweeps, blob, "pgd", a)) == 2
    a.inj_layer = 12
    assert len(join_pgd.matching_cells(two_sweeps, blob, "pgd", a)) == 1


def test_mcnemar_and_transitions():
    import math as m
    import join_detection as jd
    assert jd.mcnemar_exact(1, 1) == 1.0
    assert abs(jd.mcnemar_exact(11, 2) - 0.0225) < 1e-3
    assert jd.mcnemar_exact(25, 2) < 1e-4
    assert m.isnan(jd.mcnemar_exact(0, 0))

    def r(clean, att, coherent, detected=1):
        return {"clean_jailbroken": clean, "jailbroken": att, "coherent": coherent,
                "detected": detected}
    rows = [r(0, 1, 1), r(0, 1, 0), r(0, 1, None, detected=0), r(1, 0, None), r(1, 1, 1),
            r(0, 0, None), {"clean_jailbroken": None, "jailbroken": 1, "coherent": 1,
                            "detected": 1}]
    t = jd.transitions(rows)
    assert (t["n_paired"], t["gains"], t["losses"]) == (6, 3, 1)
    assert (t["gains_passing_gate"], t["gains_passing_gate_undetected"], t["gains_undetected"],
            t["gains_ungated"]) == (1, 0, 1, 1)
    assert t["no_evidence_of_change"] is True and t["clean_positive"] == 2 and t["attacked_positive"] == 4
    adj = jd.holm([1.0, 1.0, 0.688, 0.022, 0.0001])
    assert abs(adj[3] - 0.088) < 1e-3 and adj[4] < 0.001 and adj[0] == 1.0


def _cell_row(pid, layer, arm, sha=None, gold=(7, 7), inj=2, residual=0.5):
    toks = list(gold) if gold is not None else [7, 7]
    steps = [{"token": t, "residual": residual, "h_norm": 1.0, "matched": True,
              **({"gold_token": t} if gold is not None else {})} for t in toks]
    return {"id": f"p{pid}", "prompt_index": pid, "arm": arm,
            "budget": 0.0 if arm == "clean" else 0.02, "behavior": "jbb_refusal",
            "test_arm": "harmful", "objective": "-" if arm == "clean" else "t",
            "constraint": "all", "n_positions": 1, "inj_layer": inj, "layer": layer,
            "steps": steps, **({"n_target": len(toks)} if gold is not None else {}),
            **({"question_sha256": sha} if sha else {})}


def test_control_must_hold_the_same_question():
    same = [_cell_row(3, 1, "clean", sha="q3"), _cell_row(3, 2, "clean", sha="q3")]
    other = [_cell_row(3, 1, "clean", sha="q9"), _cell_row(3, 2, "clean", sha="q9")]
    attack = [_cell_row(3, 2, "pgd", sha="q3")]
    assert detect.verify_control(attack, same) == ([], [])
    assert detect.verify_control(attack, other) == ([3], [])
    legacy_same = [_cell_row(3, 1, "clean", gold=[1, 2, 3]), _cell_row(3, 2, "clean", gold=[1, 2, 3])]
    legacy_other = [_cell_row(3, 1, "clean", gold=[1, 2, 4]), _cell_row(3, 2, "clean", gold=[1, 2, 4])]
    legacy_attack = [_cell_row(3, 2, "pgd", gold=[1, 2, 3])]
    assert detect.verify_control(legacy_attack, legacy_same) == ([], [])
    assert detect.verify_control(legacy_attack, legacy_other) == ([3], [])
    blank = [_cell_row(3, 1, "clean", gold=None), _cell_row(3, 2, "clean", gold=None)]
    assert detect.verify_control(legacy_attack, blank) == ([], [3])
    roles = fitted("content_e0", "content_e1", "content_e2")
    roles["2"] = roles["1"]
    cal_ = cal(roles, k=1, sigma=3.0, sigma_any=4.5, min_run=1, stride=1, dense_tail=0)
    stats = {1: detect.LayerStat(0, 0.0, 1.0), 2: detect.LayerStat(0, 0.0, 1.0)}
    try:
        join_pgd.detection_verdicts(attack + other, cal_, stats, [1, 2])
    except SystemExit as e:
        assert "different question" in str(e)
    else:
        raise AssertionError("a control with the same prompt number but another question was borrowed")
    verdicts, _ = join_pgd.detection_verdicts(attack + same, cal_, stats, [1, 2])
    (v,) = verdicts.values()
    assert v["unscorable"][3] is False
    verdicts, _ = join_pgd.detection_verdicts(legacy_attack + blank, cal_, stats, [1, 2])
    (v,) = verdicts.values()
    assert v["unscorable"][3] is True


def test_resume_key_refuses_unplaceable_rows():
    attack = {"behavior": "jbb_refusal", "test_arm": "harmful", "objective": "harm",
              "n_positions": 1, "constraint": "all", "budget": 0.02, "arm": "pgd",
              "prompt_index": 3, "layer": 20, "inj_layer": 17, "deltas_sha256": "a" * 64,
              "n_prompts": 15}
    assert pgd_sipit.resume_key(attack) is not None
    assert pgd_sipit.resume_key({k: v for k, v in attack.items() if k != "n_prompts"}) is None
    assert pgd_sipit.resume_key({k: v for k, v in attack.items() if k != "deltas_sha256"}) is None
    fill = {"n_prompts": 15, "deltas_sha256": "a" * 64, "inj_layer": 17}
    legacy = {k: v for k, v in attack.items() if k not in ("n_prompts", "deltas_sha256")}
    assert pgd_sipit.resume_key(legacy, fill) == pgd_sipit.resume_key(attack)
    assert pgd_sipit.resume_key(attack | {"inj_layer": 12}) != pgd_sipit.resume_key(attack)
    assert pgd_sipit.resume_key(attack | {"deltas_sha256": "b" * 64}) != pgd_sipit.resume_key(attack)
    clean = attack | {"arm": "clean", "objective": "-", "budget": 0.0, "deltas_sha256": None}
    assert pgd_sipit.resume_key(clean)[-2:] == (None, None)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} passed")
