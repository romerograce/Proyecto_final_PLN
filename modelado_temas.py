"""
modelado_temas.py
------------------
Modelado de temas (LDA) sobre el corpus de reseñas ya preprocesado.

Requiere que ya exista una columna de texto preprocesado (tokens separados
por espacio), tal como la genera Preprocesador.pipeline_completo() en
preprocesamiento.py.

Uso típico:

    from modelado_temas import ModeladorTemas

    modelador = ModeladorTemas(num_temas=6)
    modelador.entrenar(df["texto_procesado"])
    modelador.mostrar_temas()

    # Tema dominante de cada reseña, para agregar como columna:
    df["tema_id"] = modelador.temas_por_documento(df["texto_procesado"])
"""

import gensim
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel


class ModeladorTemas:
    """
    Envoltorio simple sobre gensim LdaModel para modelado de temas.

    Parámetros
    ----------
    num_temas : int
        Número de tópicos a descubrir. Si no lo tienes claro, usa
        buscar_num_temas_optimo() primero para elegirlo con base en
        coherencia, en vez de adivinar.
    passes : int
        Número de pasadas de entrenamiento sobre el corpus.
    """

    def __init__(self, num_temas=6, passes=10, random_state=42):
        self.num_temas = num_temas
        self.passes = passes
        self.random_state = random_state
        self.diccionario = None
        self.corpus_bow = None
        self.modelo = None

    # ------------------------------------------------------------------
    def _preparar_corpus(self, textos_procesados):
        """
        textos_procesados: Serie/lista de strings, cada uno con tokens
        separados por espacio (salida de pipeline_completo()).
        """
        tokens_por_doc = [t.split() for t in textos_procesados]

        self.diccionario = corpora.Dictionary(tokens_por_doc)
        # Filtra extremos: palabras que aparecen en menos de 5 documentos,
        # o en más del 50% del corpus (demasiado genéricas para dar señal de tema)
        self.diccionario.filter_extremes(no_below=5, no_above=0.5)

        self.corpus_bow = [self.diccionario.doc2bow(tokens) for tokens in tokens_por_doc]
        return tokens_por_doc

    # ------------------------------------------------------------------
    def entrenar(self, textos_procesados):
        """Entrena el modelo LDA sobre la columna de texto preprocesado."""
        tokens_por_doc = self._preparar_corpus(textos_procesados)

        self.modelo = LdaModel(
            corpus=self.corpus_bow,
            id2word=self.diccionario,
            num_topics=self.num_temas,
            passes=self.passes,
            random_state=self.random_state,
            alpha="auto",
            eta="auto",
        )
        return self.modelo

    # ------------------------------------------------------------------
    def mostrar_temas(self, num_palabras=10):
        """Imprime las palabras más representativas de cada tema."""
        if self.modelo is None:
            raise ValueError("Primero llama a entrenar().")

        for idx, tema in self.modelo.print_topics(num_topics=self.num_temas, num_words=num_palabras):
            palabras = [p.split("*")[1].strip().replace('"', "") for p in tema.split(" + ")]
            print(f"Tema {idx}: {', '.join(palabras)}")

    # ------------------------------------------------------------------
    def temas_por_documento(self, textos_procesados):
        """
        Devuelve, para cada documento, el ID del tema dominante (el de
        mayor probabilidad). Útil para agregar como columna al DataFrame.
        """
        if self.modelo is None:
            raise ValueError("Primero llama a entrenar().")

        temas_dominantes = []
        for texto in textos_procesados:
            bow = self.diccionario.doc2bow(texto.split())
            distribucion = self.modelo.get_document_topics(bow)
            if len(distribucion) == 0:
                temas_dominantes.append(-1)  # documento sin tema claro (muy corto/vacío)
            else:
                tema_top = max(distribucion, key=lambda x: x[1])[0]
                temas_dominantes.append(tema_top)
        return temas_dominantes

    # ------------------------------------------------------------------
    def calcular_coherencia(self, textos_procesados):
        """
        Calcula la coherencia c_v del modelo actual. Valores más altos
        (típicamente 0.3-0.6+) indican temas más interpretables.
        """
        if self.modelo is None:
            raise ValueError("Primero llama a entrenar().")

        tokens_por_doc = [t.split() for t in textos_procesados]
        coherence_model = CoherenceModel(
            model=self.modelo,
            texts=tokens_por_doc,
            dictionary=self.diccionario,
            coherence="c_v",
        )
        return coherence_model.get_coherence()


# ----------------------------------------------------------------------
def buscar_num_temas_optimo(textos_procesados, rango=range(3, 11), passes=5, random_state=42):
    """
    Entrena varios modelos LDA con distinto número de tópicos y devuelve
    una lista de (num_temas, coherencia) para que elijas el mejor punto
    en vez de adivinar el número de tópicos.

    Ejemplo:
        resultados = buscar_num_temas_optimo(df["texto_procesado"])
        for n, c in resultados:
            print(n, c)
    """
    resultados = []
    for n in rango:
        modelador = ModeladorTemas(num_temas=n, passes=passes, random_state=random_state)
        modelador.entrenar(textos_procesados)
        coherencia = modelador.calcular_coherencia(textos_procesados)
        resultados.append((n, coherencia))
        print(f"num_temas={n} -> coherencia={coherencia:.4f}")
    return resultados


# ----------------------------------------------------------------------
# Prueba rápida con datos sintéticos, para validar que el módulo corre
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import random

    random.seed(42)

    vocab_terror = ["terror", "miedo", "asesino", "sangre", "oscuro", "grito", "tension"]
    vocab_comedia = ["risa", "comedia", "divertido", "gracioso", "humor", "actor", "chiste"]
    vocab_drama = ["drama", "emocion", "llanto", "profundo", "historia", "personaje", "trama"]

    textos_sinteticos = []
    for _ in range(60):
        vocab = random.choice([vocab_terror, vocab_comedia, vocab_drama])
        doc = " ".join(random.choices(vocab, k=15))
        textos_sinteticos.append(doc)

    import pandas as pd
    serie = pd.Series(textos_sinteticos)

    print("--- Búsqueda de número óptimo de temas ---")
    resultados = buscar_num_temas_optimo(serie, rango=range(2, 6), passes=5)

    print("\n--- Entrenando modelo final con 3 temas ---")
    modelador = ModeladorTemas(num_temas=3, passes=10)
    modelador.entrenar(serie)
    modelador.mostrar_temas()

    print("\n--- Coherencia del modelo final ---")
    print(modelador.calcular_coherencia(serie))

    print("\n--- Tema dominante de los primeros 5 documentos ---")
    print(modelador.temas_por_documento(serie)[:5])
