# Benchmark de moderación

## Para qué sirve

Cada entrenamiento de mmBERT-base deja un checkpoint nuevo en `models/`. El benchmark los mide a todos con el mismo examen: mensajes en español, portugués e inglés con la respuesta correcta ya etiquetada. El resultado (`bench_out/scoreboard.md`) muestra qué tan bien separa cada checkpoint los mensajes que hay que moderar de los que no, por fuente y por categoría, para decidir cuál conviene tener activo.

La parte más importante del examen es la **suite propia** (`suites/es_casino.csv`): mensajes como los que vamos a ver en el chat del casino, con lunfardo, jerga de póker y trampas. Una fila cuenta recién cuando una persona revisó sus etiquetas.

## Categorías

| Columna | Es 1 cuando… | Ejemplos |
|---|---|---|
| `insult` | insulta, ridiculiza o acosa a una persona | "sos un manco", "callate gil", "p3lotud0" |
| `threat` | amenaza con daño o violencia, o con escrachar | "te voy a romper la cara", "sé dónde vivís" |
| `identity_hate` | ataca a gente por un rasgo protegido (lista abajo) | "bolita de mierda", "andá a lavar los platos" |
| `sexual_harassment` | comentarios, pedidos o avances sexuales no deseados hacia una persona | "mandame fotos", "pasame tu número, linda" |
| `profanity` | tiene puteadas o lenguaje vulgar, dirigido o no | "qué mano de mierda", "la puta madre" |

**Rasgos protegidos** (`identity_hate`): origen étnico o color de piel, nacionalidad, religión, género, orientación sexual, identidad de género y discapacidad. Los insultos por clase social ("villero", "planero"), por el cuerpo ("gorda", "enano") o por la edad ("viejo choto") son `insult`, no `identity_hate`.

**`target`**: a quién ataca el mensaje, si es dañino.
- `player`: a una persona concreta ("sos un manco").
- `group`: a un grupo ("los bolivianos son…").
- `none`: a nadie, o el mensaje no es dañino.

**Valores:** `1` sí, `0` no, `?` no estamos seguros o no nos ponemos de acuerdo (no se evalúa), vacío = sin revisar.

### Casos difíciles

- **Trato amistoso:** "boludo, vení", "negro, arrancamos", "gordo, buena mano" van con todo en `0`. El mismo término usado con odio ("negro de mierda") es `identity_hate=1`.
- **Contar un insulto no es insultar:** "me dijo trolo, ya lo reporté" va en `0`.
- **Negaciones:** "no sos un idiota" va en `0`.
- **Puteadas sin destinatario:** "qué día de mierda" lleva `profanity=1` y el resto en `0`.
- **Autocrítica:** "soy un manco" va en `0`.
- **Plata y apuestas:** "all in 500", "pago x2", "gg wp" van con todo en `0`. Ojo con los números que parecen leetspeak.
- **Lo ofuscado cuenta igual:** "p e l o t u d o" o "tr0l0" se etiquetan como si estuvieran bien escritos.
- **Mezcla de idiomas:** "nice hand manco" es `insult=1`.
- **Sarcasmo:** "qué genio que sos" va en `0`, salvo que ridiculice a alguien de forma clara.

## Cómo revisar la suite

1. Abrí `suites/es_casino.csv` en Excel o Google Sheets. Está separado por `;`.
2. En cada fila, leé el texto, corregí las categorías y `target`, y poné `reviewed` en `1`. Las sugerencias vienen de un generador: no las des por buenas.
3. Si una fila no tiene sentido, borrala.
4. **Sumá mensajes reales de memoria:** al menos 50 frases que hayas visto en chats de juegos o de póker, con `author` en `human`. Son las más valiosas, y el reporte las separa de las generadas. Cada fila nueva necesita un `id` único (por ejemplo `es-h001`, `es-h002`…), `lang` en `es`, `author` en `human` y `reviewed` en `1`. Sin `reviewed` en `1` el benchmark la ignora, y si falta alguno de los otros da un error que dice en qué línea.
5. Guardá como CSV (UTF-8). Sirve tanto con `;` como con `,`. Ojo: Excel convierte en fórmula un texto que empieza con `=`, `+` o `-`; si te pasa, poné un espacio adelante.
6. **Segunda opinión:** que otra persona revise una copia de las mismas ~150 filas sin mirar la primera revisión, y compará las dos:
   `uv run python scripts/bench.py agreement suites/es_casino.csv copia.csv`
   Si el kappa de una categoría da menos de 0.6, el criterio no está claro: hablenlo y ajusten esta guía.

## Cómo correr

```bash
uv sync
uv run python scripts/bench.py sources                     # cuántos ejemplos hay por fuente y categoría
uv run python scripts/bench.py run                         # evalúa todos los checkpoints y escribe bench_out/scoreboard.md
uv run python scripts/bench.py run --checkpoints ft-20261007-184309,ft-20261008-120000   # solo esos
uv run python scripts/bench.py run --include-unreviewed    # vista PRELIMINAR con filas sin revisar
uv run python scripts/bench.py latency --device cpu --threads 4   # latencia en CPU con 4 hilos, como un server chico
```

`--batch-size` (8 por defecto) controla cuántos mensajes se procesan juntos; en una GPU de 4 GB, 8 es más rápido que 32. Si la GPU se queda sin memoria, bajalo a 4.

## Checkpoints

Sin `--checkpoints`, el benchmark evalúa todos los checkpoints completos de `models/` (los que `scripts/activate_model.py --list` no marca como incompletos). Cada uno aparece con el nombre de su carpeta, y el scoreboard dice cuál es el activo.

Casi todos los ejemplos en español con los que se entrena también son filas de la suite, y cada checkpoint guarda con cuáles entrenó. Por eso se sacan del examen los textos con los que se entrenó **cualquiera** de los checkpoints comparados, para todos: así todos se miden con los mismos mensajes y ninguno con algo que memorizó. El scoreboard dice cuántos se sacaron.

Las predicciones quedan guardadas en `bench_out/predictions/`, una carpeta por checkpoint. El entrenamiento también guarda ahí las suyas, así que un checkpoint recién entrenado no se vuelve a evaluar. Si revisaste o corregiste filas de la suite, solo se vuelven a evaluar esas. Con `--bootstrap 200` el cálculo de intervalos es más rápido.

## Cómo leer el scoreboard

- **AUROC [IC 95%]**: probabilidad de que el checkpoint puntúe más alto un mensaje dañino que uno sano. 0.5 es tirar una moneda y 1 es perfecto. El intervalo muestra cuánto puede variar por azar.
- **R · FPR**: cada checkpoint tiene un umbral, elegido para que marque por error al 5% de los mensajes sanos de la partición de calibración. R es qué parte de los dañinos detecta con ese umbral y FPR qué parte de los sanos marca en cada fuente. Si el FPR de una fuente se aleja mucho del 5%, ese umbral no se traslada bien a ese tipo de texto.
- **Diferencia contra el mejor**: si el intervalo incluye 0, no hay evidencia de que el mejor sea mejor de verdad.
- **pocos datos**: menos de 30 ejemplos de alguna clase; no sacar conclusiones.
- **suite-es/human vs. suite-es/generated**: si un checkpoint anda mucho mejor con las frases generadas, probablemente aprendió el estilo del generador más que el del chat real.
- **targets_player** (si un mensaje dañino ataca a una persona concreta): el modelo no tiene esa salida, así que la columna dice n/a. La suite igual etiqueta `target`.

## Fuentes y licencias

| Fuente | Idioma | Licencia |
|---|---|---|
| suite-es | es | propia |
| HateCheck multilingüe (es, pt) y HateCheck (en) | es/pt/en | CC BY 4.0 |
| ToLD-Br | pt | CC BY-SA 4.0 |
| OLID-BR | pt | CC BY 4.0 |

Las públicas se bajan de Hugging Face en una revisión fija. Quedaron afuera por licencia o acceso: OffendES (pide aceptar términos y su licencia se contradice), HatEval y HateBR (no comerciales). Los datos con los que se entrena, y sus licencias, están en [`finetuning.md`](finetuning.md).
