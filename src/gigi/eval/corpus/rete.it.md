# Accesso alla rete

La VPN aziendale usa WireGuard. Per accedere installa il client, importa il
file di configurazione che trovi nel portale e connettiti: la rete interna
(10.0.0.0/8) diventa raggiungibile senza altre impostazioni.

Gli utenti esterni usano la VPN solo per la gestione degli incidenti; il resto
del traffico passa dai proxy di perimetro. La policy richiede di scollegare la
VPN a fine giornata: le sessioni inattive scadono comunque dopo 12 ore.

Per un nuovo host registrato nel DNS interno, apri una richiesta nel portale
rete indicando nome, responsabile e ambiente.