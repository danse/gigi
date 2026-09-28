# Backup and retention

Every night at 02:00 UTC a cron job snapshots the production database and the
object store. Snapshots are stored in the `s3://gigi-backups` bucket, encrypted
with the ops KMS key.

Retention is 45 days. Nightly snapshots older than that expire automatically;
there is no separate weekly or monthly archive. The restore playbook lives in
the runbook: `restore-db.sh` takes a snapshot id and spins up a temporary
instance to validate it.

Once a quarter a human restores the oldest snapshot to confirm the process
still works end to end.