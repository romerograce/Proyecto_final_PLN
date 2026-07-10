# 🎬 Análisis de Críticas de Cine en Español

Proyecto final del módulo de Procesamiento de Lenguaje Natural — Maestría en Inteligencia Artificial, Universidad Católica Boliviana "San Pablo".

Sistema de punta a punta que analiza reseñas de cine en español: identifica **temas** predominantes (LDA), predice **sentimiento** (positivo/negativo/neutral) comparando un modelo clásico (Naive Bayes) contra uno neuronal (BETO), y expone todo a través de una **interfaz interactiva**.

```
Preprocesamiento → LDA (temas) → TF-IDF/BETO → Sentimiento → Interfaz (Gradio)
```

---

## 📊 Dataset

- **Fuente**: [`us-lsi/muchocine`](https://huggingface.co/datasets/us-lsi/muchocine) (HuggingFace)
- **Tamaño**: 3,872 reseñas de películas, escritas originalmente en español (no traducidas)
- **Etiquetas**: rating de 0 a 4, mapeado a 3 clases de sentimiento (negativo / neutral / positivo)

```python
from datasets import load_dataset
ds = load_dataset("us-lsi/muchocine", revision="refs/convert/parquet")
```

## 🗂️ Estructura del repositorio

```
proyecto-analisis-cine/
├── src/
│   ├── __init__.py
│   ├── preprocesamiento.py         Limpieza, tokenización, lematización (spaCy)
│   ├── modelado_temas.py           LDA con bigramas (gensim)
│   ├── clasificador_clasico.py     TF-IDF + Naive Bayes
│   ├── clasificador_beto.py        Fine-tuning de BETO (transformers)
│   └── interfaz_gradio.py          Interfaz de uso final
├── notebooks/
│   └── proyecto_completo.ipynb     Notebook de Colab con el pipeline completo, ya ejecutado
├── data/
│   └── README.md                   El dataset se carga públicamente desde HuggingFace
├── docs/
│   ├── ARQUITECTURA.md             Diagrama y decisiones de diseño
│   └── resultados/
│       ├── matriz_confusion_nb.png
│       ├── matriz_confusion_beto.png
│       └── interfaz_funcionando.png
└── README.md
```

## 🚀 Cómo ejecutar

Todo el pipeline está pensado para correr en **Google Colab** (requiere GPU para el fine-tuning de BETO).

1. Abre `notebooks/proyecto_completo.ipynb` en Colab.
2. Ejecuta las celdas en orden (instalación → datos → preprocesamiento → LDA → clasificación → interfaz).
3. Al llegar a la sección de interfaz, se genera un enlace público de Gradio (`*.gradio.live`) para interactuar con el sistema.

## 🧠 Pipeline

| Etapa | Técnica | Detalle |
|---|---|---|
| Preprocesamiento | spaCy (`es_core_news_sm`) | Limpieza, lematización, protección de negaciones y palabras de sentimiento |
| Modelado de temas | LDA + bigramas (gensim) | 6 temas, coherencia c_v = 0.33 |
| Sentimiento (clásico) | TF-IDF + Naive Bayes | Línea base |
| Sentimiento (neuronal) | BETO (`dccuchile/bert-base-spanish-wwm-uncased`) | Fine-tuning, 3 épocas |
| Interfaz | Gradio | Modo "una opinión": pega una reseña → sentimiento + tema + recomendación |

## 📈 Resultados: clásico vs. neuronal

| Métrica | Naive Bayes | BETO |
|---|---|---|
| **Accuracy** | **59.0%** | 54.5% |
| F1 negativo | 0.65 | 0.61 |
| F1 neutral | 0.42 | 0.45 |
| F1 positivo | 0.66 | 0.57 |

**Hallazgo principal**: en este proyecto, Naive Bayes superó a BETO. BETO mostró una tendencia de mejora consistente con más épocas de entrenamiento (46.6% → 50.8% → 54.5%), lo que sugiere que con más recursos computacionales probablemente superaría al modelo clásico — pero dentro de las limitaciones de tiempo/cómputo del proyecto, el modelo simple resultó más eficiente. Ver `informe.pdf` para el análisis completo.

La clase **"neutral"** fue consistentemente la más difícil de clasificar en ambos modelos, lo cual es esperable: el lenguaje de una reseña "neutral" suele mezclar elogios y críticas en el mismo texto.

## 🎓 Temas identificados por LDA

1. Actuación / interpretación actoral
2. Cine policial / género negro
3. Drama familiar / social español
4. Acción / sagas de Hollywood
5. Terror / comedia de bajo presupuesto
6. Cine de autor / lenguaje visual

## 🛠️ Tecnologías

- **Preprocesamiento**: spaCy
- **Modelado de temas**: gensim (LDA)
- **Clasificación clásica**: scikit-learn (TF-IDF + Naive Bayes)
- **Clasificación neuronal**: HuggingFace Transformers (BETO)
- **Interfaz**: Gradio
- **Entorno**: Google Colab (GPU T4)

## 👤 Autor

[Tu nombre] — Maestría en Inteligencia Artificial, UCB "San Pablo"
