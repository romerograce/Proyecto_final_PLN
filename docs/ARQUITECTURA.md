# Arquitectura del sistema

## Diagrama del pipeline

```
Texto crudo (reseña de cine)
        │
        ▼
┌─────────────────────┐
│  Preprocesamiento    │  spaCy (es_core_news_sm)
│  - Limpieza          │  - minúsculas, URLs, menciones, emojis
│  - Tokenización       │  - protección de negaciones y palabras
│  - Lematización       │    de sentimiento (bien/mal/mejor/peor...)
└──────────┬───────────┘
           │
           ├──────────────────────┐
           ▼                      ▼
┌─────────────────────┐  ┌─────────────────────┐
│  Modelado de temas    │  │  Representación       │
│  LDA + bigramas       │  │  TF-IDF (clásico)     │
│  (gensim)              │  │  Tokenizer BETO        │
│  6 temas               │  │  (neuronal)             │
└──────────┬───────────┘  └──────────┬───────────┘
           │                          │
           │              ┌───────────┴───────────┐
           │               ▼                        ▼
           │      ┌──────────────┐      ┌──────────────────┐
           │      │ Naive Bayes    │      │ BETO (fine-tuning) │
           │      │ (MultinomialNB)│      │ (transformers)      │
           │      └───────┬──────┘      └─────────┬────────┘
           │              │                          │
           │              └────────────┬────────────┘
           │                            ▼
           │                  Comparación de métricas
           │                  (accuracy, F1, matriz de confusión)
           │                            │
           └────────────┬───────────────┘
                         ▼
              ┌─────────────────────┐
              │   Interfaz (Gradio)   │
              │   sentimiento + tema  │
              │   + recomendación      │
              └─────────────────────┘
```

## Decisiones de diseño

### 1. Dos textos preprocesados distintos, según el modelo

Se generan dos versiones del texto preprocesado:

- **`texto_procesado`**: limpio, lematizado, sin stopwords (excepto negaciones y
  palabras de sentimiento). Usado para TF-IDF/Naive Bayes y para LDA, donde la
  reducción de vocabulario ayuda a que el modelo bag-of-words generalice mejor.
- **`texto_transformer`**: limpieza ligera (URLs, menciones), sin lematizar ni
  quitar stopwords. Usado para BETO, ya que el modelo pre-entrenado aprendió a
  interpretar el lenguaje natural completo; lematizar o quitar palabras
  destruiría información que el transformer sabe aprovechar.

### 2. Protección de palabras de sentimiento en el preprocesamiento

Se detectó que la lista de stopwords por defecto de spaCy en español incluye
palabras con fuerte carga de sentimiento ("bien", "mal", "buena", "mejor",
"peor"), probablemente heredadas de expresiones idiomáticas. Se excluyeron
explícitamente de la eliminación de stopwords, junto con las negaciones
("no", "nunca", "sin"), ya que ambos grupos son señales críticas para la
tarea de análisis de sentimiento.

### 3. LDA con bigramas y filtrado agresivo del diccionario

La primera versión de LDA (solo unigramas, filtro de diccionario permisivo)
produjo temas con coherencia c_v = 0.276 y palabras muy genéricas repetidas
entre temas ("poco", "mejor", "momento"). Se mejoró agregando:
- Detección de bigramas (`gensim.models.phrases`), capturando expresiones
  como "efecto_especial" o "sin_duda".
- Filtro de palabras genéricas específico para LDA (palabras útiles para
  sentimiento pero no para temas, como "mucho", "bien", "momento").
- Filtrado más estricto del diccionario (`no_below=10, no_above=0.35`).

Esto elevó la coherencia a c_v = 0.328 y produjo temas más interpretables.

### 4. Comparación clásico vs. neuronal

Se entrenaron ambos modelos sobre la misma partición de datos
(80% entrenamiento / 20% prueba, estratificada por clase). BETO se
entrenó con 1, 2 y 3 épocas para observar la curva de convergencia
(46.6% → 50.8% → 54.5% accuracy), documentándose la corrida final de
3 épocas por restricciones de tiempo de cómputo. Ver `README.md` para
la tabla comparativa completa.

### 5. Interfaz desacoplada de los objetos entrenados

`interfaz_gradio.py` recibe los modelos y objetos auxiliares (preprocesador,
clasificador, modelo LDA, diccionario) como parámetros explícitos de la
función `crear_interfaz(...)`, en vez de depender de variables globales.
Esto permite que el módulo se importe y reutilice sin depender del espacio
de nombres del notebook donde se ejecuta.

## Stack tecnológico

| Componente | Tecnología |
|---|---|
| Preprocesamiento de texto | spaCy (`es_core_news_sm`) |
| Modelado de temas | gensim (LDA + Phrases) |
| Clasificación clásica | scikit-learn (TF-IDF + MultinomialNB) |
| Clasificación neuronal | HuggingFace Transformers (BETO) |
| Interfaz | Gradio |
| Entorno de entrenamiento | Google Colab (GPU T4) |
