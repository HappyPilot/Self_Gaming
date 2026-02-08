from agents.utils import hf_vljepa_probe_eval as probe


def test_extract_probe_terms_from_semantic_fields():
    row = {
        "video_scenario": "ball rolling down ramp",
        "expected_inference": "velocity increase over time",
        "failure_mode": "constant speed guess",
    }
    terms = probe.extract_probe_terms(row)
    assert "velocity" in terms
    assert "increase" in terms
    assert "over" not in terms


def test_evaluate_probe_rows_coverage():
    rows = [
        {"expected_inference": "velocity increase over time"},
        {"expected_behavior": "pause or reroute"},
    ]
    samples = [
        {"scene": {"text": ["velocity increased after motion"], "objects": []}, "actions": []},
        {"scene": {"text": ["idle"], "objects": []}, "actions": []},
    ]
    report = probe.evaluate_probe_rows(rows, samples)
    assert report["rows_total"] == 2
    assert report["rows_matched"] == 1
    assert report["coverage_pct"] == 50.0
    assert report["unmatched_indices"] == [1]
