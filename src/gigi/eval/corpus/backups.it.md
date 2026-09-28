# Backup e conservazione

Ogni notte alle 02:00 UTC un cron job scatta istantanee del database di
produzione e dell'object store. Le istantanee vengono salvate nel bucket
`s3://gigi-backup`, cifrate con la chiave KMS delle operazioni.

La conservazione è di 45 giorni. Le istantanee notturne più vecchie scadono in
automatico; non esistono archivi settimanali o mensili separati. Il playbook di
ripristino è nel runbook: `ripristina-db.sh` accetta un id di istantanea e
avvia un'istanza temporanea per verificarla.

Una volta a trimestre qualcuno ripristina l'istantanea più vecchia per
confermare che il processo funzioni ancora dall'inizio alla fine.