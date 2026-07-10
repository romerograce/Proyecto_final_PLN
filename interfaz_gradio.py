"""
interfaz_gradio.py
--------------------
Interfaz de uso final del sistema de análisis de críticas de cine.
Combina: preprocesamiento -> sentimiento (Naive Bayes) -> tema (LDA) -> recomendación.

IMPORTANTE: este script asume que ya existen en el entorno (mismo notebook
de Colab) las siguientes variables/objetos, generados en pasos anteriores:
    - prep              (instancia de Preprocesador)
    - clf_nb             (instancia de ClasificadorClasico, ya entrenado)
    - modelo_lda          (LdaModel de gensim, ya entrenado)
    - diccionario         (corpora.Dictionary de gensim)
    - bigram_model        (Phraser de gensim)
    - quitar_genericas    (función)
    - nombres_temas       (dict id_tema -> nombre)

Si alguno no existe en tu sesión, este script lo indicará con un error
claro al momento de correr la función de predicción.
"""

import gradio as gr


def recomendacion_desde_sentimiento(sentimiento):
    """Traduce el sentimiento predicho a una recomendación simple."""
    if sentimiento == "positivo":
        return "✅ Recomendada"
    elif sentimiento == "negativo":
        return "❌ No recomendada"
    else:
        return "➖ Depende del gusto (opiniones mixtas)"


def crear_funcion_analisis(prep, clf_nb, modelo_lda, diccionario, bigram_model, quitar_genericas, nombres_temas):
    """
    Crea la función de análisis "cerrando" sobre los objetos pasados como
    argumento, en vez de depender de variables globales del módulo. Esto
    evita el problema de que las variables definidas en el notebook de
    Colab no sean visibles dentro de este archivo importado.
    """

    def analizar_resena(texto_resena):
        if not texto_resena or not texto_resena.strip():
            return "⚠️ Por favor pega una reseña", "", ""

        texto_proc = prep.pipeline_completo(texto_resena)
        sentimiento_pred = clf_nb.predecir(texto_proc)

        texto_lda = quitar_genericas(texto_proc)
        tokens_lda = bigram_model[texto_lda.split()]
        bow = diccionario.doc2bow(tokens_lda)
        distribucion_temas = modelo_lda.get_document_topics(bow)

        if len(distribucion_temas) == 0:
            tema_nombre = "No se pudo determinar (texto muy corto)"
        else:
            tema_id = max(distribucion_temas, key=lambda x: x[1])[0]
            tema_nombre = nombres_temas.get(tema_id, f"Tema {tema_id}")

        recomendacion = recomendacion_desde_sentimiento(sentimiento_pred)

        return sentimiento_pred.upper(), tema_nombre, recomendacion

    return analizar_resena


# ----------------------------------------------------------------------
# Construcción de la interfaz Gradio
# ----------------------------------------------------------------------
def crear_interfaz(prep, clf_nb, modelo_lda, diccionario, bigram_model, quitar_genericas, nombres_temas):
    """
    Recibe todos los objetos ya entrenados como parámetros explícitos.

    Uso en Colab:
        app = crear_interfaz(prep, clf_nb, modelo_lda, diccionario,
                              bigram_model, quitar_genericas, nombres_temas)
        app.launch(share=True, debug=True)
    """
    analizar_resena = crear_funcion_analisis(
        prep, clf_nb, modelo_lda, diccionario, bigram_model, quitar_genericas, nombres_temas
    )

    ejemplos = [
        ["Una obra maestra del cine de terror, con actuaciones brillantes y un guion que te mantiene en tensión de principio a fin. Totalmente recomendable."],
        ["Aburrida de principio a fin. El guion no tiene sentido y las actuaciones son pésimas. Una pérdida de tiempo."],
        ["Es una película que ni fu ni fa. Tiene momentos interesantes pero en general se siente repetitiva y sin mucha personalidad."],
    ]

    with gr.Blocks(title="Análisis de Críticas de Cine") as demo:
        gr.Markdown("# 🎬 Sistema de Análisis de Críticas de Cine en Español")
        gr.Markdown(
            "Pega una crítica de película y el sistema analizará su **sentimiento**, "
            "el **tema/género** predominante (vía LDA), y si es **recomendable**."
        )

        with gr.Row():
            entrada = gr.Textbox(
                label="Pega aquí la crítica de cine",
                placeholder="Escribe o pega una reseña de película...",
                lines=6,
            )

        boton = gr.Button("Analizar", variant="primary")

        with gr.Row():
            salida_sentimiento = gr.Textbox(label="Sentimiento detectado")
            salida_tema = gr.Textbox(label="Tema / género predominante")
            salida_recomendacion = gr.Textbox(label="¿Recomendada?")

        boton.click(
            fn=analizar_resena,
            inputs=entrada,
            outputs=[salida_sentimiento, salida_tema, salida_recomendacion],
        )

        gr.Examples(
            examples=ejemplos,
            inputs=entrada,
        )

    return demo


if __name__ == "__main__":
    app = crear_interfaz()
    app.launch(share=True, debug=True)
