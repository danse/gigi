# Alertas y notificaciones

Las alertas se definen en el repositorio `prometheus-rules`, en archivos YAML
por servicio. Cada regla fija una condición, un nivel (crítico, aviso o
paging) y un enlace al runbook.

Las notificaciones llegan al canal `#alertas` de Slack y, para los niveles de
paging, al teléfono del guardia. Un aviso que no se responde en 15 minutos
escala a todo el equipo de plataforma.

Antes de crear una alerta nueva, comprueba que no duplique otra existente y que
la condición esté respaldada por una métrica real. Cada alerta debe tener un
dueño asignado.