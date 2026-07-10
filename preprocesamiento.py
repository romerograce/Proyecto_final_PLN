"""
preprocesamiento.py
--------------------
Pipeline de preprocesamiento de texto en español, reutilizable en todo
el proyecto (LDA, TF-IDF/Naive Bayes, y preparación de texto para BETO).

Uso típico:

    from preprocesamiento import Preprocesador

    prep = Preprocesador()
    texto_limpio = prep.limpiar("¡Qué película más buena! Vi https://x.com y me encantó :)")
    tokens = prep.tokenizar_y_lematizar(texto_limpio)

    # Para aplicar sobre una columna de un DataFrame:
    df["texto_procesado"] = df["review_body"].apply(prep.pipeline_completo)
"""

import re
import unicodedata

import spacy


class Preprocesador:
    """
    Pipeline de preprocesamiento de texto en español.

    Parámetros
    ----------
    modelo_spacy : str
        Nombre del modelo de spaCy a cargar (por defecto 'es_core_news_sm').
    conservar_negaciones : bool
        Si True, evita eliminar palabras de negación ('no', 'nunca', 'sin', etc.)
        aunque estén en la lista de stopwords. Esto es importante para el
        análisis de sentimientos, donde "no me gustó" pierde su significado
        si se elimina el "no".
    lematizar : bool
        Si True, aplica lematización (recomendado para LDA y TF-IDF).
        Para BETO/transformers normalmente NO se lematiza: el propio
        tokenizer del modelo pre-entrenado maneja el texto crudo/normalizado.
    """

    NEGACIONES = {
        "no", "nunca", "jamás", "jamas", "tampoco", "ni", "sin", "nadie",
        "nada", "ningún", "ningun", "ninguna", "ninguno",
    }

    # IMPORTANTE: la lista de stopwords en español de spaCy incluye, por
    # defecto, varias palabras con fuerte carga de sentimiento (probablemente
    # arrastradas de expresiones como "de buenas a primeras" o "a mal tiempo,
    # buena cara"). Si no se protegen, el pipeline elimina justo las palabras
    # más informativas para clasificar sentimiento. Verificado con:
    #   'buena', 'buen', 'bien', 'mal', 'mejor', 'peor' -> True (son stopwords)
    PALABRAS_SENTIMIENTO = {
        "bien", "mal", "buen", "buena", "buenas", "buenos",
        "mejor", "mejores", "peor", "peores",
        "mucho", "muchos", "muy", "poco", "pocos", "poca", "pocas",
        "más", "mas", "menos",
    }

    def __init__(self, modelo_spacy="es_core_news_sm", conservar_negaciones=True,
                 conservar_sentimiento=True, lematizar=True):
        self.nlp = spacy.load(modelo_spacy, disable=["parser", "ner"])
        self.conservar_negaciones = conservar_negaciones
        self.conservar_sentimiento = conservar_sentimiento
        self.lematizar = lematizar

        # Ajustamos las stopwords de spaCy para no perder negaciones ni
        # palabras con carga de sentimiento/intensidad
        self.stopwords = set(self.nlp.Defaults.stop_words)
        if self.conservar_negaciones:
            self.stopwords -= self.NEGACIONES
        if self.conservar_sentimiento:
            self.stopwords -= self.PALABRAS_SENTIMIENTO

    # ------------------------------------------------------------------
    # Limpieza de texto crudo
    # ------------------------------------------------------------------
    def limpiar(self, texto: str) -> str:
        """
        Limpieza básica de texto crudo:
        - minúsculas
        - quita URLs
        - quita menciones (@usuario) y hashtags (deja la palabra del hashtag)
        - quita HTML residual
        - normaliza espacios y puntuación repetida
        - conserva tildes y ñ (importantes en español)
        """
        if not isinstance(texto, str):
            return ""

        texto = texto.lower()

        # HTML residual
        texto = re.sub(r"<[^>]+>", " ", texto)

        # URLs
        texto = re.sub(r"https?://\S+|www\.\S+", " ", texto)

        # Menciones (@usuario) -> se eliminan
        texto = re.sub(r"@\w+", " ", texto)

        # Hashtags: se conserva la palabra, se quita el símbolo
        texto = re.sub(r"#(\w+)", r"\1", texto)

        # Emojis y caracteres no imprimibles fuera del rango básico latino
        texto = re.sub(
            r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U0001F1E6-\U0001F1FF]",
            " ",
            texto,
        )

        # Repetición excesiva de puntuación o letras (ej: "buenisimaaaaa" -> "buenisima")
        texto = re.sub(r"(.)\1{2,}", r"\1", texto)

        # Quitar caracteres que no sean letras (incluye tildes/ñ), números o espacios
        texto = re.sub(r"[^a-záéíóúñü0-9\s]", " ", texto)

        # Espacios múltiples
        texto = re.sub(r"\s+", " ", texto).strip()

        return texto

    # ------------------------------------------------------------------
    # Tokenización + stopwords + lematización
    # ------------------------------------------------------------------
    def tokenizar_y_lematizar(self, texto_limpio: str) -> list[str]:
        """
        Recibe texto ya limpio (ver limpiar()) y devuelve una lista de
        tokens finales: sin stopwords, lematizados (si self.lematizar=True).
        """
        doc = self.nlp(texto_limpio)

        tokens = []
        for token in doc:
            if token.is_space or token.is_punct:
                continue
            palabras_protegidas = self.NEGACIONES | self.PALABRAS_SENTIMIENTO
            if len(token.text) < 2 and token.text not in palabras_protegidas:
                continue
            if token.text in self.stopwords:
                continue

            palabra = token.lemma_ if self.lematizar else token.text
            tokens.append(palabra)

        return tokens

    # ------------------------------------------------------------------
    # Pipeline completo: texto crudo -> texto procesado (string)
    # ------------------------------------------------------------------
    def pipeline_completo(self, texto: str) -> str:
        """
        Limpieza + tokenización + lematización, devuelto como un solo
        string separado por espacios (listo para TfidfVectorizer o gensim
        con .split()).
        """
        limpio = self.limpiar(texto)
        tokens = self.tokenizar_y_lematizar(limpio)
        return " ".join(tokens)

    # ------------------------------------------------------------------
    # Variante para modelos tipo BETO: limpieza ligera, SIN lematizar
    # ------------------------------------------------------------------
    def limpiar_para_transformer(self, texto: str) -> str:
        """
        Limpieza mínima pensada para alimentar el tokenizer de BETO u
        otro transformer: solo normaliza espacios/URLs/menciones, pero
        NO lematiza ni quita stopwords (el modelo pre-entrenado ya
        aprendió a manejar el texto natural, incluidas las stopwords).
        """
        if not isinstance(texto, str):
            return ""

        texto = re.sub(r"<[^>]+>", " ", texto)
        texto = re.sub(r"https?://\S+|www\.\S+", " ", texto)
        texto = re.sub(r"@\w+", " ", texto)
        texto = re.sub(r"\s+", " ", texto).strip()
        return texto


# ----------------------------------------------------------------------
# Función auxiliar: mapeo de rating (0-4, escala del dataset muchocine)
# a sentimiento de 3 clases.
# ----------------------------------------------------------------------
def rating_a_sentimiento(rating: int) -> str:
    """
    Mapea el star_rating del dataset muchocine (escala 0-4) a una
    etiqueta de sentimiento de 3 clases.

    0, 1 -> negativo
    2    -> neutral
    3, 4 -> positivo
    """
    if rating <= 1:
        return "negativo"
    elif rating == 2:
        return "neutral"
    else:
        return "positivo"


# ----------------------------------------------------------------------
# Prueba rápida al ejecutar el archivo directamente
# ----------------------------------------------------------------------
if __name__ == "__main__":
    prep = Preprocesador()

    ejemplo = (
        "¡¡Qué película más buenaaaaa!! No me la esperaba para nada. "
        "Vi el trailer en https://youtube.com/algo y ya sabía que sería un exitazo 😍😍 "
        "@usuario tenías razón, el guión es buenísimo."
    )

    limpio = prep.limpiar(ejemplo)
    print("Texto limpio:\n", limpio)

    tokens = prep.tokenizar_y_lematizar(limpio)
    print("\nTokens (lematizados, sin stopwords):\n", tokens)

    print("\nPipeline completo (para TF-IDF/LDA):\n", prep.pipeline_completo(ejemplo))

    print("\nLimpieza ligera (para BETO):\n", prep.limpiar_para_transformer(ejemplo))

    print("\nMapeo de ratings de ejemplo:")
    for r in [0, 1, 2, 3, 4]:
        print(f"  rating={r} -> {rating_a_sentimiento(r)}")

    # Verificación explícita: palabras clave de sentimiento NO deben perderse
    print("\n--- Verificación de palabras de sentimiento protegidas ---")
    prueba_sentimiento = "la actuación fue muy buena pero el guión estuvo mal"
    tokens_prueba = prep.tokenizar_y_lematizar(prep.limpiar(prueba_sentimiento))
    print("Texto:", prueba_sentimiento)
    print("Tokens:", tokens_prueba)
    assert "buena" in tokens_prueba or "buen" in tokens_prueba, "¡Se perdió la palabra de sentimiento 'buena'!"
    assert "mal" in tokens_prueba, "¡Se perdió la palabra de sentimiento 'mal'!"
    print("OK: las palabras de sentimiento se conservan correctamente.")
