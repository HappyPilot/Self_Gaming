# Config Profiles

This repo uses env profiles for Docker Compose.

- Defaults: `config/defaults.env`
- Profiles: `config/jetson.env`, `config/mac.env`, `config/production.env`

## Select a profile
```bash
export SG_PROFILE=jetson
```

Examples:
```bash
export SG_PROFILE=mac
export SG_PROFILE=production
```
Notes:
- `mac` profile uses `OCR_BACKEND=easyocr` and `OCR_FORCE_CPU=1`.
- `production` keeps defaults and expects explicit overrides.

## Local overrides (not committed)
If you need host-specific endpoints, copy `config/local.env.example` to
`config/local.env` and pass it via `SG_LOCAL_ENV_FILE` (use an absolute path):
```bash
export SG_LOCAL_ENV_FILE="$(pwd)/config/local.env"
```

## LLM endpoint overrides
Point Jetson agents at a remote LLM server by overriding these:
```bash
LLM_ENDPOINT=http://10.0.0.230:11434/v1/chat/completions
TEACHER_LOCAL_ENDPOINT=http://10.0.0.230:11434/v1/chat/completions
```

## OCR language defaults
`OCR_LANGS` defaults to `en` to reduce OCR latency and false positives.
Enable more languages by overriding `OCR_LANGS` in your local env file.
