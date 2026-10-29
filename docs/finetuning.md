# Ajustar el modelo (fine-tuning)

## La idea en cuatro frases

El modelo de la app es **mmBERT-base**, un modelo de lenguaje multilingüe chico (307M parámetros, licencia MIT) al que le agregamos una salida por categoría (insulto, amenaza, odio por identidad, acoso sexual y puteada) y otra de sentimiento. De fábrica no sabe nada de moderación: el ajuste le muestra miles de ejemplos con la respuesta correcta y acomoda sus pesos para que acierte. Después se lo evalúa con mensajes que **nunca vio**, para saber si de verdad aprendió y no solo memorizó. Solo se usa el modelo nuevo si le gana al que está activo.

## El circuito

1. **Juntar ejemplos etiquetados**, por tres vías que terminan en el mismo dataset (`data/training/examples.json`):
   - **Chat:** debajo de cada mensaje, "¿Es insulto? Sí / No".
   - **Pantalla de entrenamiento** (`/training.html`): muestra de a uno los ejemplos pendientes, con la sugerencia y lo que opina el modelo hoy. Atajos: `1` insulto, `2` no insulto, `3` descartar.
   - **Importar un archivo:** `uv run python scripts/import_examples.py archivo.csv` (columnas `text` y, opcionalmente, `label` con `insult` o `clean`).
2. **Entrenar:** `uv run python scripts/finetune.py`. En la 3080 Ti tarda unos 20 minutos la primera vez (14 son de etiquetar el sentimiento) y unos 5 las siguientes.
3. **Leer el reporte.** Si el modelo nuevo es mejor, queda activado. Reiniciá el server para usarlo.

## Con qué datos aprende

- **ToLD-Br** (portugués): todos los mensajes dañinos y 6000 sanos, de los que no están en el benchmark.
- **OLID-BR** (portugués): su partición de entrenamiento.
- **Tus ejemplos en español** del grupo de entrenamiento, repetidos 3 veces para que pesen más:
  - si el mismo texto está en la suite, se usan las etiquetas por categoría de la suite;
  - si no, "insulto" dice solo que es un insulto, y "no insulto" dice que no es insulto, amenaza ni odio. Puteada y acoso quedan sin dato.

Lo que no se sabe (una categoría sin dato) no cuenta ni a favor ni en contra: el modelo no aprende nada de esa casilla.

Ningún texto de ToLD-Br u OLID-BR repite uno del benchmark. Tus ejemplos sí pueden estar en la suite: el checkpoint guarda cuáles usó, y el benchmark los saca del examen.

## Datos, modelos y licencias

Todo corre en nuestras máquinas: ningún mensaje sale a una API externa. Los datasets y modelos de Hugging Face se bajan en una revisión fija.

| Qué | Para qué | Licencia |
|---|---|---|
| [mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base) | el modelo que se ajusta y que usa la app | MIT |
| [ToLD-Br](https://huggingface.co/datasets/mteb/told-br) | entrenamiento | CC BY-SA 4.0 |
| [OLID-BR](https://huggingface.co/datasets/dougtrajano/olid-br) | entrenamiento | CC BY 4.0 |
| `data/training/examples.json` (con los ejemplos generados de `datasets/generated_rioplatense_v1.jsonl`) | entrenamiento | propia |
| [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | etiqueta el sentimiento de los textos de entrenamiento; la app no lo usa | Apache-2.0 |
| [DynaSent](https://github.com/cgpotts/dynasent), ronda 2, test (`data/sentiment/`) | evaluar el sentimiento en inglés | CC BY 4.0 |
| `data/sentiment/suite_sentiment_draft.csv` | evaluar el sentimiento en español (borrador sin revisar) | propia |

Las fuentes del examen (la suite, HateCheck, ToLD-Br y OLID-BR) y sus licencias están en [`benchmark.md`](benchmark.md). Sin los archivos de `data/sentiment/`, el reporte no evalúa el sentimiento.

## El sentimiento

No hay un dataset de sentimiento en español o portugués con licencia que permita uso comercial (TASS, InterTASS, TweetSentBR y BRIGHTER no la tienen). Por eso **Qwen3-4B**, un LLM abierto, etiqueta el sentimiento de cada texto de entrenamiento una sola vez, en la GPU. Solo se usa para eso: el modelo que corre en la app es mmBERT. Las etiquetas quedan en `data/training/sentiment_labels.jsonl`, y al volver a entrenar solo se etiquetan los textos nuevos.

Son etiquetas automáticas que nadie revisó. Para entrenar sin etiquetar textos nuevos: `--skip-sentiment-labels` (esos textos no cuentan para el sentimiento).

## Criterio de etiquetado

- **Insulto:** agresión dirigida a una persona. Insultos ("sos un pelotudo"), slurs y amenazas ("te voy a romper la cara").
- **No insulto:** todo lo demás. Sarcasmo ("qué genio que sos"), puteadas sin destinatario ("qué día de mierda"), trato amistoso ("boludo, vení"), negaciones ("no sos un idiota") y hablar sobre insultos ("me dijo forro y me dolió").

Ser consistente importa más que la cantidad: el modelo aprende exactamente el criterio que reflejan tus etiquetas. Ante la duda, "Descartar". Las categorías finas (amenaza, odio, acoso) se etiquetan en la suite: ver [`benchmark.md`](benchmark.md).

## Cómo leer el reporte

| Línea | Qué significa |
|---|---|
| Umbral de flag | el puntaje desde el que se tapa un mensaje. Se elige para que marque solo al 5% de los mensajes sanos de calibración del chat en español |
| AUROC por fuente | qué tan bien separa los mensajes dañinos de los sanos en cada fuente del benchmark, sin los textos con los que entrenó. 0.50 es azar y 1.00 es perfecto |
| Detecta / marca | con ese umbral, qué parte de los mensajes dañinos del chat en español tapa y qué parte de los sanos tapa por error |
| Sentimiento (macro-F1) | qué tan bien acierta el sentimiento, en español (un borrador sin revisar) y en inglés (DynaSent). 1.00 es perfecto |
| Contra el modelo activo | los dos modelos medidos con los mismos mensajes, que ninguno vio |

El "flag" es el puntaje más alto entre insulto, amenaza y odio por identidad. La puteada sola y el acoso sexual no tapan el mensaje.

El modelo nuevo se activa solo si:

- separa mejor el chat en español que el modelo activo (AUROC más alto), sin empeorar el sentimiento en más de 0.05;
- o, si no hay un modelo activo, si su AUROC en el chat en español llega al menos a 0.75.

## Por qué hay tres grupos de ejemplos

Cada ejemplo cae, según su texto, en uno de tres grupos fijos:

- **entrenamiento (70%):** de acá aprende;
- **calibración (10%)** y **prueba (20%):** no se usan para aprender.

El mismo texto siempre cae en el mismo grupo, así una frase repetida nunca está a la vez en entrenamiento y en el examen. El umbral y las métricas salen del benchmark, que sigue la misma idea: calibración para el umbral, evaluación para los números.

## Cuántos ejemplos hacen falta

- **Mínimo para entrenar:** 20 insultos y 20 no insultos en el grupo de entrenamiento.
- **Recomendado:** unos 800 en total, con muchos casos difíciles (negaciones, "boludo" amistoso, sarcasmo, amenazas).

Lo que más falta hoy son **amenazas, odio por identidad y acoso sexual** revisados: el modelo aprendió con 93 amenazas y ningún caso de acoso.

## Opciones

```
uv run python scripts/finetune.py --seed 1                    # otra semilla, para ver cuánto varía
uv run python scripts/finetune.py --no-activate               # guardar sin ponerlo en uso
uv run python scripts/finetune.py --skip-sentiment-labels     # no etiquetar el sentimiento de textos nuevos
```

| Variable | Valor por defecto | Para qué |
|---|---|---|
| `DUTY` | `0.6` | qué parte del tiempo trabaja la GPU; el resto descansa para no calentarse |
| `COOLDOWN_S` | `120` | pausa en segundos entre etiquetar el sentimiento y entrenar |
| `FLAG_THRESHOLD` | el del modelo | en el server, reemplaza el umbral que eligió el entrenamiento |
| `MODEL_DEVICE` | `cpu` | en el server, dónde corre el modelo (`cuda` para la GPU) |

Entrena en la GPU si hay una. El modelo que guarda corre en CPU: en un server de 4 hilos tarda unos 20 ms por mensaje.

## Comparar checkpoints y volver atrás

```
uv run python scripts/activate_model.py --list    # modelos entrenados, su AUROC en el chat en español y su umbral (* = activo)
uv run python scripts/bench.py run                # todos los checkpoints con el mismo examen: bench_out/scoreboard.md
uv run python scripts/activate_model.py models/ft-20261007-184309
```

Cada entrenamiento se guarda en su propia carpeta dentro de `models/`. Ninguno se pisa. Los números de `--list` salen de la evaluación de cada entrenamiento; para compararlos con exactamente los mismos mensajes, usá `bench.py run` (ver [`benchmark.md`](benchmark.md)).

## Qué pasa por dentro

- **Un solo modelo, una sola pasada:** mmBERT lee el mensaje una vez y devuelve ocho números: una probabilidad independiente por categoría (un mensaje puede insultar y putear a la vez) y una distribución de sentimiento.
- **Pérdida:** entropía cruzada binaria por categoría, donde las casillas sin dato no cuentan, más entropía cruzada contra las probabilidades de sentimiento de Qwen.
- **Receta:** 2 épocas, lotes de 32, tasa de aprendizaje 4e-5 con calentamiento, textos de hasta 128 tokens, bf16 en la GPU.
- **Evaluación:** con las mismas fuentes, particiones y métricas que el benchmark. Las predicciones quedan en `bench_out/predictions/`, así el benchmark no las vuelve a calcular.
- **Checkpoint:** `models/ft-<fecha>/` guarda el modelo, el tokenizer y `training_meta.json`, con el umbral, las métricas, la decisión y los textos de la app con los que entrenó.
