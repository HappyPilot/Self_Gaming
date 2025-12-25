MLflow tracking
===============

This repo ships a small helper to run a local MLflow UI backed by a file store.

Start MLflow
------------

```bash
tools/mlflow_start.sh
```

By default, runs use `/mnt/ssd/mlflow` when available and fall back to
`./mlruns` otherwise. The UI listens on port 5001.

Log a sample run
---------------

```bash
python training/mlflow_example.py
```

Environment variables:

- `MLFLOW_TRACKING_URI` (default: http://127.0.0.1:5001)
- `MLFLOW_EXPERIMENT` (default: self_gaming)
- `MLFLOW_RUN_NAME` (optional)
