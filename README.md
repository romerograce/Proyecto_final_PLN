# Análisis de Críticas de Cine en Español

## Descripción del proyecto

Este proyecto fue desarrollado como trabajo final del módulo de **Procesamiento de Lenguaje Natural (PLN)** de la **Maestría en Inteligencia Artificial** de la Universidad Católica Boliviana "San Pablo".

El objetivo principal es analizar reseñas de películas escritas en español para identificar dos aspectos importantes:

* El **tema principal** del que habla la reseña mediante un modelo de *Topic Modeling* (LDA).
* El **sentimiento** expresado por el usuario (positivo, neutral o negativo) utilizando dos enfoques diferentes: un modelo clásico (Naive Bayes) y un modelo basado en redes neuronales (BETO).

Finalmente, se desarrolló una interfaz sencilla en **Gradio** para que cualquier usuario pueda escribir una reseña y obtener el análisis de manera interactiva.

---

## Flujo general del proyecto

El proceso realizado durante el desarrollo del proyecto fue el siguiente:

```
Carga del dataset
        ↓
Preprocesamiento del texto
        ↓
Modelado de temas (LDA)
        ↓
Clasificación de sentimiento
(Naive Bayes y BETO)
        ↓
Interfaz en Gradio
```

---

## Dataset utilizado

Para este proyecto se utilizó el conjunto de datos **MuchoCine**, disponible públicamente en Hugging Face.

**Características del dataset**

* Fuente: us-lsi/muchocine
* Idioma: Español
* Total de reseñas: 3.872
* Calificaciones originales: de 0 a 4 estrellas

Las calificaciones fueron agrupadas en tres categorías de sentimiento:

* Negativo
* Neutral
* Positivo

El dataset se carga directamente desde Hugging Face mediante la librería `datasets`, por lo que no es necesario descargar archivos manualmente.

---

## Estructura del proyecto

```
proyecto-analisis-cine/
│
├── src/
│   ├── preprocesamiento.py
│   ├── modelado_temas.py
│   ├── clasificador_clasico.py
│   ├── clasificador_beto.py
│   └── interfaz_gradio.py
│
├── notebooks/
│   └── proyecto_completo.ipynb
│
├── docs/
│   └── resultados/
│
├── data/
│   └── README.md
│
└── README.md
```

Cada archivo contiene una parte específica del desarrollo, desde el procesamiento de texto hasta la construcción de la interfaz final.

---

## Desarrollo del proyecto

### 1. Preprocesamiento

Antes de entrenar los modelos fue necesario limpiar las reseñas.

Las principales tareas realizadas fueron:

* Conversión del texto a minúsculas.
* Eliminación de caracteres especiales.
* Eliminación de palabras vacías (*stopwords*).
* Tokenización.
* Lematización utilizando **spaCy**.
* Conservación de negaciones importantes para el análisis de sentimiento.

El objetivo fue obtener textos más limpios sin perder información relevante.

---

### 2. Modelado de temas

Para descubrir los temas principales presentes en las reseñas se utilizó el algoritmo **LDA (Latent Dirichlet Allocation)** implementado con la librería **gensim**.

También se generaron **bigramas**, permitiendo que expresiones frecuentes fueran tratadas como una sola unidad.

Después de realizar varias pruebas se trabajó con **6 temas**, obteniendo una coherencia aproximada de **0.33**.

Los temas encontrados corresponden principalmente a:

* Actuación
* Cine policial
* Drama
* Acción
* Terror y comedia
* Cine de autor

---

### 3. Clasificación de sentimiento

Se compararon dos enfoques distintos.

#### Modelo clásico

Se utilizó:

* TF-IDF como representación del texto.
* Multinomial Naive Bayes como clasificador.

Este modelo sirve como línea base y requiere pocos recursos computacionales.

#### Modelo neuronal

También se entrenó el modelo **BETO**, una versión de BERT entrenada específicamente para español.

El entrenamiento se realizó durante **3 épocas** utilizando GPU en Google Colab.

---

## Resultados obtenidos

Los resultados finales fueron los siguientes:

| Modelo      | Accuracy  |
| ----------- | --------- |
| Naive Bayes | **59.0%** |
| BETO        | 54.5%     |

Aunque inicialmente se esperaba que BETO obtuviera un mejor rendimiento, el modelo clásico consiguió una mayor precisión en este proyecto.

Durante el entrenamiento se observó que BETO fue mejorando en cada época, por lo que probablemente un entrenamiento más largo o con mayor capacidad computacional habría permitido obtener mejores resultados.

También se observó que la categoría **neutral** fue la más difícil de clasificar para ambos modelos.

---

## Interfaz

Como parte final del proyecto se desarrolló una interfaz utilizando **Gradio**.

El usuario solamente debe escribir una reseña de una película y el sistema devuelve:

* El sentimiento identificado.
* El tema al que pertenece la reseña.
* Una recomendación basada en la clasificación obtenida.

Esto permite utilizar el modelo de forma sencilla sin necesidad de ejecutar código.

---

## Tecnologías utilizadas

* Python
* spaCy
* gensim
* scikit-learn
* Transformers (BETO)
* Gradio
* Google Colab

---

## Cómo ejecutar el proyecto

1. Abrir el notebook `proyecto_completo.ipynb` en Google Colab.
2. Ejecutar las celdas en el orden establecido.
3. Esperar a que finalice el entrenamiento de los modelos.
4. Ejecutar la última sección para iniciar la interfaz de Gradio.
5. Abrir el enlace generado para probar el sistema.

---

## Conclusiones

Durante este proyecto fue posible aplicar diferentes técnicas de Procesamiento de Lenguaje Natural para analizar reseñas de películas en español.

Además de construir un sistema funcional, el trabajo permitió comparar un modelo tradicional con un modelo basado en aprendizaje profundo, observando sus ventajas y limitaciones dentro de un entorno con recursos computacionales limitados.

---

## Autor

**Grace Linda Romero Arancibia**
