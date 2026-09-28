# Rilascio in produzione

## Procedura

Ogni versione parte da un commit con tag. La pipeline compila l'immagine del
contenitore e la carica nel registro interno; poi una persona la promuove
all'ambiente di produzione.

## Passaggi

1. Crea il tag: `git tag vX.Y.Z` e fai push.
2. Esegui `./scripts/deploy.sh produzione` dal portatile di operazioni.
3. Guarda la dashboard; lo script attende che il nuovo pod sia sano prima di
   dichiarare il successo.

## Rollback

Se la nuova versione dà problemi, torna indietro in pochi secondi con
`./scripts/deploy.sh produzione --rollback`. Lo script conserva l'immagine
precedente e la riassegna al deployment, senza ricompilare nulla. I rollback
vengono registrati nel canale delle operazioni.