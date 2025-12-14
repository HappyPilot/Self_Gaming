import unittest

from agents.curriculum_status import (
    DeathCurriculumStatus,
    compute_death_curriculum_status,
)


class FakeMemRPC:
    def __init__(self, responses):
        self.responses = responses

    def query(self, payload, timeout=1.0):
        mode = payload.get("mode")
        value = self.responses.get(mode)
        return {"value": value}


def _build_recent(deaths, resurrects, other):
    entries = [{"delta": "hero_dead"} for _ in range(deaths)]
    entries += [{"delta": "hero_resurrected"} for _ in range(resurrects)]
    entries += [{"delta": "npc_dialog"} for _ in range(other)]
    return entries


class CurriculumStatusTests(unittest.TestCase):
    def test_ready_status(self):
        responses = {
            "rules": [{"text": "rule"}],
            "recent_critical": _build_recent(10, 8, 0),
            "calibration_events": [{"x_norm": 0.5, "y_norm": 0.8}],
        }
        mem = FakeMemRPC(responses)
        status = compute_death_curriculum_status(
            mem,
            min_resurrects=5,
            min_success_rate=0.6,
            min_calibrations=1,
        )
        self.assertTrue(status.ready)
        self.assertEqual(status.reason, "ready")
        self.assertEqual(status.death_count, 10)
        self.assertEqual(status.resurrect_count, 8)

    def test_low_success_rate(self):
        responses = {
            "rules": [],
            "recent_critical": _build_recent(10, 2, 0),
            "calibration_events": [{"x_norm": 0.5, "y_norm": 0.8}],
        }
        mem = FakeMemRPC(responses)
        status = compute_death_curriculum_status(
            mem,
            min_resurrects=2,
            min_success_rate=0.5,
            min_calibrations=1,
        )
        self.assertFalse(status.ready)
        self.assertEqual(status.reason, "low_success_rate")

    def test_no_calibration(self):
        responses = {
            "rules": [],
            "recent_critical": _build_recent(5, 5, 0),
            "calibration_events": [],
        }
        mem = FakeMemRPC(responses)
        status = compute_death_curriculum_status(
            mem,
            min_resurrects=1,
            min_success_rate=0.5,
            min_calibrations=1,
        )
        self.assertFalse(status.ready)
        self.assertEqual(status.reason, "no_calibration")


if __name__ == "__main__":
    unittest.main()

