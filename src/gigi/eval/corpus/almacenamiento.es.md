# Almacenamiento de archivos

Los archivos de negocio viven en buckets de object storage: uno por entorno
(`dev`, `staging`, `prod`) y por equipo. Las claves de acceso se rotan cada 90
días.

Las subidas pasan por la pasarela `upload.internal`, que valida el tipo y el
tamaño máximo (2 GB) y firma la ubicación. Los descargables se sirven con URL
firmada que caduca en 15 minutos.

La política de retención del bucket `prod` es de 7 años; los demás buckets
borran archivos con más de 90 días de antigüedad.