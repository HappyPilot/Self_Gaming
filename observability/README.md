Observability stack
===================

This stack provides a Prometheus exporter that converts MQTT metrics into
Prometheus metrics, plus a bundled Prometheus + Grafana setup.

Quick start
-----------

Run alongside the main compose (so the exporter can reach `mq` by default):

```bash
docker compose -f docker-compose.yml -f docker-compose.observability.yml up -d
```

If you run the observability stack alone, set MQTT_HOST to a reachable broker
address (for example, the host IP or `host.docker.internal` on Docker Desktop):

```bash
MQTT_HOST=127.0.0.1 docker compose -f docker-compose.observability.yml up -d
```

Optional profiles
-----------------

- `infra_metrics` enables node-exporter for host memory/CPU metrics.
- `gpu_metrics` enables dcgm-exporter for NVIDIA GPU metrics (requires `runtime: nvidia`).

Example:

```bash
PROMETHEUS_CONFIG=prometheus.full.yml docker compose -f docker-compose.observability.yml --profile infra_metrics --profile gpu_metrics up -d
```

Grafana defaults to `admin/admin` and is available on port 3000.
