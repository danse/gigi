# Inventari de sistemes

Els entorns de producció i pre-producció es documenten a l'inventari
`inventari.yaml`: host, funció, propietari i data d'alta de cada servidor.

Cada trimestre es revisa l'inventari per confirmar que no hi hagi servidors
obsolets d'un servei donat de baixa. Els hosts que no apareixen a l'inventari
es desconnecten de la xarxa en 30 dies.

Els canvis d'infraestructura es fan mitjançant la plataforma
d'aprovisionament, mai directament al servidor. Els accessos privilegiats
requereixen un permís temporal justificat al tiquet.