# Compilación del frontend

El frontend se construye con Vite. `npm run build` genera los estáticos en
`dist/`, listos para servir desde el CDN con un hash en el nombre de cada
archivo.

Durante el desarrollo, `npm run dev` levanta el servidor local con
actualización en caliente. Los estilos usan variables CSS definidas en
`src/theme.css`; no se escribe CSS en línea.

Los cambios visuales pasan por revisión en el entorno de pre-producción antes
de llegar a producción. El paquete final debe sumar menos de 200 KB
descomprimido.