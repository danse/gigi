# Publicación de releases

## Proceso

Cada release sale de una rama etiquetada. El pipeline genera la imagen del
contenedor y la sube al registro interno; después, una persona promociona la
imagen al entorno de producción.

## Pasos

1. Crea la etiqueta: `git tag vX.Y.Z` y haz push.
2. Ejecuta `./scripts/publicar.sh produccion` desde el portátil de operaciones.
3. Comprueba el panel; el script espera a que el nuevo pod esté sano antes de
   informar del éxito.

## Reversión

Si la versión nueva falla, revierte en segundos con `./scripts/publicar.sh
produccion --revertir`. Se conserva la imagen anterior y se vuelve a fijar la
aplicación a ella, sin reconstruir nada. Las reversiones quedan registradas en
el canal de operaciones.