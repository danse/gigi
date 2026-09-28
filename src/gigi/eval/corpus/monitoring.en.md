# Monitoring and alerting

## Dashboards

Grafana hosts the main dashboards, one per service. The ops dashboard shows
request rate, error rate and p95 latency for the last hour, refreshed every 30
seconds. The platform dashboard covers CPU, memory and disk of the k8s nodes.

## Alerts

Alert rules live in the `prometheus-rules` repo. A new alert should sound only
for symptoms, not causes, and must include a runbook link and a severity
label (`critical`, `warning` or `page`). `page` alerts page the on-call during
business hours only.

## On-call shifts

Shifts rotate weekly on Friday at noon. During your shift you own the pager:
acknowledge an activation within 15 minutes or the alert escalates to the whole
platform team. Handoff notes go in the ops channel.