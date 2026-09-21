# Onboarding

## Welcome

This document walks new teammates through getting local development running
for the Acme platform. Plan for about 30 minutes.

## 1. Get access

Access is granted through Google SSO. Ask your manager to add you to the
`acme-devs` group in the Google Workspace console. After that, sign in at
`https://login.acme.dev` once so the identity provider provisions your
account.

## 2. Install the toolchain

The repo requires Python 3.12, `uv`, and `make`. Run `make setup` which
creates the virtual environment, installs dependencies, and copies the
example config into `~/.config/acme/`.

## 3. Seed the local database

`make db` starts Postgres and Redis via docker compose and applies migrations.
Then `make seed` loads a small fixture dataset so local development has
realistic data to work with.

## 4. Run the app

Start the API with `make dev`. It listens on `http://localhost:8000` with
auto-reload enabled. Verify with `curl http://localhost:8000/healthz`.