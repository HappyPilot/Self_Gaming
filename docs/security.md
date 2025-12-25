# Security Notes

MQTT authentication
-------------------

To enable username/password authentication for Mosquitto, generate a password
file in `mosquitto/passwords` (keep this file out of git):

```bash
docker run --rm -it -v "$(pwd)/mosquitto:/mosquitto" eclipse-mosquitto:2 \
  mosquitto_passwd -c /mosquitto/passwords <username>
```

Start the secure broker override:

```bash
docker compose -f docker-compose.yml -f docker-compose.mqtt-secure.yml up -d mq
```

Ensure `mosquitto/passwords` exists before starting; Mosquitto will refuse to
start without it when `allow_anonymous false` is configured.

If your services run with host networking, keep `MQTT_HOST=127.0.0.1`.
For bridged networks, connect via the published port (use the host IP or
`host.docker.internal` on Docker Desktop).

Clients can set credentials via environment variables:

```bash
export MQTT_USERNAME=<username>
export MQTT_PASSWORD=<password>
```

The repo's `sitecustomize.py` applies these credentials automatically for
Python agents that use `paho-mqtt`.

Optional TLS
------------

Place certificates in `mosquitto/certs`:

- `ca.crt`
- `server.crt`
- `server.key`

The TLS listener config is stored in `mosquitto/tls.conf` and is mounted into
`mosquitto/conf.d` when the TLS override is enabled.

Then enable the TLS override:

```bash
docker compose -f docker-compose.yml -f docker-compose.mqtt-secure.yml \
  -f docker-compose.mqtt-tls.yml up -d mq
```

TLS listeners will be on port 8883. Ensure your MQTT clients enable TLS when
connecting to port 8883 and trust the CA if verification is enabled.
