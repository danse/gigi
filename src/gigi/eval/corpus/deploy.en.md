# Production deployment

## Release process

Deployments run from a tagged commit. The CI pipeline builds the container
image and pushes it to the internal registry; a human then promotes the image
to the production environment.

## Steps

1. Tag the release: `git tag vX.Y.Z` and push.
2. Run `./scripts/deploy.sh production` from the ops laptop.
3. Watch the dashboard; the deploy script waits for the new pod to become
   healthy before reporting success.

## Rollback

If the new version misbehaves, roll back in seconds with
`./scripts/deploy.sh production --rollback`. The script keeps the previous
image and re-pins the deployment to it, so no rebuild is needed. Rollbacks are
logged to the operations channel.