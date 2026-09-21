# Deployment

## Deploying to production

All services run as Docker containers and are managed by `docker compose`.
The canonical build is produced by `./scripts/deploy.sh`, which builds images,
runs migrations, and rolls out the update.

## The deploy script

`./scripts/deploy.sh` performs three steps:

1. Builds the application image and tags it with the git commit SHA.
2. Applies pending database migrations (`make migrate`) against the primary
   Postgres instance.
3. Performs a rolling update of the `web` and `worker` services so there is
   no downtime.

## Rolling updates and rollback

The `web` service runs three replicas behind a load balancer. The deploy
script drains and replaces one replica at a time. If health checks fail for
more than 30 seconds, the deploy aborts and the previous image tag is
restored automatically.

## Environment configuration

Secrets live in `~/.config/acme/secrets.env` on the deploy host and are
injected into containers at runtime. Do not commit secrets to the repository.

## Health checks

`GET /healthz` returns `200 OK` when the service can reach Postgres, Redis,
and the worker queue. The load balancer polls this endpoint every 5 seconds.