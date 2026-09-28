# CI pipeline

Every pull request runs the pipeline in three stages: lint, test, build.

Stage one lints the changed files with ruff and checks formatting. Stage two
runs the unit test suite with pytest under the oldest supported Python. Stage
three builds the container image and pushes it to the registry under
`pr-<number>`.

Merges to `main` run the same stages plus smoke tests against a staging
deploy. Failed stages block the merge; a green pipeline is the only entry
criterion. Artifacts from `main` are kept for 14 days.