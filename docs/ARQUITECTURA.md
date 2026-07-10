# Arquitectura del sistema

## Descripción general

El sistema fue desarrollado siguiendo un flujo de procesamiento que inicia con una reseña de cine escrita en español y finaliza mostrando al usuario el sentimiento identificado y el tema principal de la opinión.

A lo largo del proyecto se implementaron diferentes etapas de procesamiento de lenguaje natural, utilizando tanto técnicas tradicionales como modelos basados en aprendizaje profundo.

## Flujo del sistema

```text
Reseña de cine
      │
      ▼
Preprocesamiento del texto
      │
      ├───────────────┐
      ▼               ▼
Modelado de temas   Clasificación de sentimiento
 (LDA)              (Naive Bayes y BETO)
      │               │
      └───────┬───────┘
              ▼
      Comparación de resultados
              │
              ▼
       Interfaz en Gradio
```

---

# Descripción de cada etapa

## 1. Preprocesamiento

La primera etapa consiste en preparar las reseñas antes de utilizarlas para entrenar los modelos.

Durante este proceso se realizaron tareas como:

* Conversión del texto a minúsculas.
* Eliminación de caracteres innecesarios.
* Eliminación de enlaces y menciones.
* Tokenización.
* Lematización mediante **spaCy**.
* Conservación de palabras importantes para el análisis de sentimiento, como las negaciones.

El objetivo fue reducir el ruido presente en los textos y conservar únicamente la información relevante para el análisis.

---

## 2. Representación del texto

No todos los modelos utilizan el texto de la misma manera, por lo que fue necesario preparar dos versiones diferentes.

### Texto para Naive Bayes y LDA

Para estos modelos se utilizó un texto más limpio, eliminando palabras poco informativas y aplicando lematización.

Esta representación facilita que los modelos tradicionales encuentren patrones con mayor facilidad.

### Texto para BETO

En el caso del modelo BETO, solamente se realizó una limpieza básica.

No se eliminaron stopwords ni se aplicó lematización, ya que este tipo de modelos fue entrenado para comprender el lenguaje en su forma natural y puede aprovechar mejor el contexto completo.

---

## 3. Modelado de temas

Para identificar los principales temas presentes en las reseñas se utilizó el algoritmo **Latent Dirichlet Allocation (LDA)** implementado con la librería **gensim**.

Además, se incorporó la generación de **bigramas**, permitiendo reconocer expresiones frecuentes como una sola unidad.

Después de realizar varias pruebas se decidió trabajar con **6 temas**, ya que ofrecían un equilibrio adecuado entre interpretación y coherencia.

---

## 4. Clasificación de sentimiento

El proyecto compara dos enfoques distintos para clasificar el sentimiento de una reseña.

### Modelo clásico

El primer modelo utiliza:

* TF-IDF para representar el texto.
* Multinomial Naive Bayes como clasificador.

Este enfoque es sencillo, rápido de entrenar y sirve como referencia para comparar otros modelos más complejos.

### Modelo neuronal

El segundo enfoque utiliza **BETO**, un modelo basado en la arquitectura BERT entrenado específicamente para español.

Se realizó un proceso de ajuste (*fine-tuning*) durante tres épocas utilizando Google Colab con GPU.

---

## 5. Comparación de modelos

Ambos modelos fueron entrenados utilizando la misma división de datos:

* 80% para entrenamiento.
* 20% para pruebas.

Posteriormente se compararon diferentes métricas de evaluación, entre ellas:

* Accuracy.
* Precision.
* Recall.
* F1-score.
* Matriz de confusión.

Esta comparación permitió analizar el desempeño de ambos enfoques bajo las mismas condiciones.

---

## 6. Interfaz del sistema

Como etapa final se desarrolló una interfaz utilizando **Gradio**.

La interfaz permite que el usuario escriba una reseña y obtenga automáticamente:

* El sentimiento detectado.
* El tema identificado por LDA.
* Una recomendación generada a partir del resultado obtenido.

De esta manera, el proyecto puede utilizarse sin necesidad de ejecutar directamente el código.

---

# Decisiones tomadas durante el desarrollo

Durante el desarrollo del proyecto se tomaron algunas decisiones importantes.

* Se utilizaron dos versiones del texto preprocesado, ya que los modelos tradicionales y los modelos tipo Transformer requieren diferentes formas de representación.

* Se conservaron palabras como **"no"**, **"nunca"**, **"bien"**, **"mal"** y otras expresiones relacionadas con el sentimiento, porque eliminarlas podía afectar el desempeño del clasificador.

* Se incorporaron bigramas en el modelo LDA para obtener temas más claros y fáciles de interpretar.

* Se evaluaron tanto un modelo clásico como un modelo basado en aprendizaje profundo con el fin de comparar su rendimiento utilizando el mismo conjunto de datos.

* La interfaz fue desarrollada como un módulo independiente para facilitar su reutilización y permitir que el sistema pueda ejecutarse sin modificar el código principal.

---

# Herramientas utilizadas

| Componente               | Herramienta                      |
| ------------------------ | -------------------------------- |
| Lenguaje de programación | Python                           |
| Preprocesamiento         | spaCy                            |
| Modelado de temas        | gensim                           |
| Clasificación clásica    | scikit-learn                     |
| Clasificación neuronal   | Hugging Face Transformers (BETO) |
| Interfaz                 | Gradio                           |
| Entorno de desarrollo    | Google Colab (GPU T4)            |

---

# Resumen de la arquitectura

La arquitectura implementada integra técnicas tradicionales de Procesamiento de Lenguaje Natural con modelos modernos basados en Transformers.

El flujo desarrollado permite procesar una reseña desde su limpieza inicial hasta la presentación de resultados mediante una interfaz interactiva, facilitando tanto el análisis de temas como la clasificación automática del sentimiento.

