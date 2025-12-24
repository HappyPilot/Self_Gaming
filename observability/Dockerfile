FROM python:3.11-slim

WORKDIR /app
COPY observability/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY observability /app/observability
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "-m", "observability.mqtt_exporter"]
