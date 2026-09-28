# Configuración de la VPN

La VPN corporativa funciona con WireGuard. Para configurarla, instala el
cliente, importa el perfil desde el portal de red y conecta: la subred interna
(10.0.0.0/8) queda alcanzable sin más ajustes.

El perfil personal usa tu cuenta corporativa; el acceso a la red interna queda
registrado en el registro de auditoría. Desconecta la VPN al finalizar el día:
las sesiones inactivas expiran a las 12 horas igualmente.

Para incluir un equipo nuevo en el DNS interno, abre una solicitud en el portal
de red con el nombre, el responsable y el entorno.