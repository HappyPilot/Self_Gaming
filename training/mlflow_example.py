"""Example MLflow logging for params/metrics/artifacts."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

try:
    import mlflow
except Exception as exc:  # noqa: BLE001
    raise SystemExit(f"mlflow is not installed: {exc}")


TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "self_gaming")
RUN_NAME = os.getenv("MLFLOW_RUN_NAME")


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.log_param("agent", "policy_agent")
        mlflow.log_param("backend", "reflex_stub")
        mlflow.log_metric("reward", 0.42)
        mlflow.log_metric("latency_ms", 15.7)
        payload = {"timestamp": time.time(), "note": "sample artifact"}
        artifact_path = Path("tmp_mlflow_artifact.json")
        artifact_path.write_text(json.dumps(payload), encoding="utf-8")
        mlflow.log_artifact(str(artifact_path))
        try:
            artifact_path.unlink()
        except FileNotFoundError:
            pass
    print("MLflow run logged")


if __name__ == "__main__":
    main()
