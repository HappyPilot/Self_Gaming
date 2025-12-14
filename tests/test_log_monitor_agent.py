import tempfile
from pathlib import Path

from agents import log_monitor_agent as lm


def write_lines(path: Path, lines):
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line + "\n")


def test_log_monitor_detects_training_failures():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        log_file = base / "train_manager.log"
        write_lines(
            log_file,
            [
                "2025-01-01T00:00:00 INFO train job",
                "2025-01-01T00:00:01 ERROR job_failed job_id=1",
                "Traceback: boom",
                "loss nan detected",
            ],
        )
        other_log = base / "vision_agent.log"
        write_lines(
            other_log,
            [
                "2025-01-01 WARNING slow frame rate",
                "2025-01-01 INFO recovered",
            ],
        )
        lm.LOG_DIRS[:] = [base]
        lm.TRAINING_SOURCES.add("train_manager")
        agent = lm.LogMonitorAgent()
        summary, alerts = agent.scan_once()
        assert "train_manager" in summary
        assert summary["train_manager"]["errors"] >= 1
        assert summary["vision_agent"]["warnings"] >= 1
        issues = {alert["issue"] for alert in alerts}
        assert "train_job_failed" in issues
        assert "loss_nan" in issues or "nan_values" in issues
        # second pass should not resurface identical alerts due to offsets
        summary2, alerts2 = agent.scan_once()
        assert not alerts2
        assert summary2["train_manager"]["lines"] == summary["train_manager"]["lines"]
