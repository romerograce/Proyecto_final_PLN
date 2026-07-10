"""
clasificador_clasico.py
------------------------
Clasificador clásico de sentimiento: TF-IDF + Naive Bayes.
Sirve como línea base para comparar contra el modelo neuronal (BETO).
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


class ClasificadorClasico:
    """
    Envoltorio sobre TfidfVectorizer + MultinomialNB.

    Uso típico:
        clf = ClasificadorClasico()
        clf.entrenar(df["texto_procesado"], df["sentimiento"])
        clf.evaluar()
        clf.matriz_confusion()
    """

    def __init__(self, max_features=5000, ngram_range=(1, 2), test_size=0.2, random_state=42):
        self.vectorizador = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        self.modelo = MultinomialNB()
        self.test_size = test_size
        self.random_state = random_state
        self.clases = None

    def entrenar(self, textos, etiquetas):
        X_train, X_test, y_train, y_test = train_test_split(
            textos, etiquetas,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=etiquetas,
        )

        X_train_tfidf = self.vectorizador.fit_transform(X_train)
        X_test_tfidf = self.vectorizador.transform(X_test)

        self.modelo.fit(X_train_tfidf, y_train)

        self.X_test_tfidf = X_test_tfidf
        self.y_test = y_test
        self.y_pred = self.modelo.predict(X_test_tfidf)
        self.clases = sorted(etiquetas.unique()) if hasattr(etiquetas, "unique") else sorted(set(etiquetas))

        return self.modelo

    def evaluar(self):
        acc = accuracy_score(self.y_test, self.y_pred)
        print(f"Exactitud (accuracy): {acc:.4f}\n")
        print("Reporte de clasificación:")
        print(classification_report(self.y_test, self.y_pred))
        return acc

    def matriz_confusion(self, guardar_como=None):
        cm = confusion_matrix(self.y_test, self.y_pred, labels=self.clases)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=self.clases, yticklabels=self.clases)
        plt.xlabel("Predicción")
        plt.ylabel("Real")
        plt.title("Matriz de confusión - Naive Bayes")
        plt.tight_layout()
        if guardar_como:
            plt.savefig(guardar_como)
        plt.show()

    def predecir(self, texto_procesado_individual):
        """Predice el sentimiento de un solo texto ya preprocesado."""
        vector = self.vectorizador.transform([texto_procesado_individual])
        return self.modelo.predict(vector)[0]


# ----------------------------------------------------------------------
# Prueba rápida con datos sintéticos
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("prueba_sentimiento.csv")

    clf = ClasificadorClasico()
    clf.entrenar(df["texto_procesado"], df["sentimiento"])
    clf.evaluar()

    print("\nPredicción de ejemplo:", clf.predecir("pelicula bueno actuacion excelente"))
