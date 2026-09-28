# Publicació del lloc web

El lloc web públic es publica des de la branca `main` del repositori `web`,
mitjançant el pipeline de publicació. Cada commit a `main` genera l'HTML
estàtic i el puja al CDN amb una validació de contingut prèvia.

Les pàgines noves es creen amb la plantilla del repositori i han de passar la
revisió d'enllaços i d'accessibilitat. La publicació s'anuncia al canal
`#web`.

La reversió és immediata: es torna a publicar la build anterior des del panell
de control, sense tocar el codi.