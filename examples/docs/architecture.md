# Architecture

## Overview

The Acme platform is a three-tier web application: a FastAPI backend, a
Postgres primary database, and a Redis-backed task queue for background work.

## Backend

The API layer is a FastAPI application in `src/api/`. It exposes REST
endpoints under `/v1/` and uses SQLAlchemy for data access. Authentication
is JWT-based and validated against the Google SSO provider.

## Database

Postgres is the system of record. The schema is versioned with Alembic
migrations stored in `migrations/`. The `accounts`, `projects`, and `billing`
tables are the core of the data model. A nightly job streams a sanitized copy
to a read replica used for analytics.

## Background processing

Long-running work (email delivery, invoice generation, webhook dispatch) is
pushed onto a Redis queue and processed by the `worker` service. The worker
retries failed jobs with exponential backoff, up to five attempts, before
moving them to a dead-letter queue for manual review.

## Frontend

The dashboard is a single-page React application served as static files from
the `web` container and proxied through the API.