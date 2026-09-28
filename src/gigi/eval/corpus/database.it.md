# Migrazioni del database

Le modifiche allo schema del database viaggiano come migrazioni SQL versate nel
repo `migrations`. Ogni migrazione ha un numero progressivo e due file:
`up.sql` e `down.sql`.

Per applicare le migrazioni in sospeso esegui `make migrate` nell'ambiente di
destinazione. Lo script confronta la migrazione più recente applicata con i
file presenti e applica in ordine quelle mancanti, dentro una transazione.

Se una migrazione fallisce, il comando segna il database come "in migrazione";
esegui `make migrate-status` e poi `make migrate --rollback` per annullare
l'ultima migrazione completata. Non cancellare mai una migrazione già applicata
in produzione.