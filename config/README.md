# Config Profiles

This repo uses env profiles for Docker Compose.

- Defaults: `config/defaults.env`
- Profiles: `config/jetson.env`, `config/mac.env`, `config/production.env`
- Select a profile with `SG_PROFILE`:
```bash
export SG_PROFILE=jetson
```

## Jetson execution
- Docker runs on Jetson: `10.0.0.68` (SSH available)
- All tests and compose sanity checks run on Jetson
- Repo remote: `git@github.com:HappyPilot/Self_Gaming.git`
